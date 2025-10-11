import os
import re
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from typing import List, Tuple
from pre_process import Preprocessor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import conll2003.conlleval as eval

logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_embeddings_from_bin_gz(file_path: str) -> KeyedVectors:
    """Load Word2Vec embeddings from a compressed binary file."""
    logger.info("Loading Word2Vec embeddings...")
    embeddings = KeyedVectors.load_word2vec_format(file_path, binary=True)
    logger.info(
        f"Loaded {len(embeddings)} word vectors with {embeddings.vector_size} dimensions"
    )
    return embeddings


class VanillaRNN(nn.Module):

    def __init__(
        self,
        preWeights,
        layer_mode: str = "RNN",
        fine_tune=False,
        bidirect=False,
        hiddenSize=256,
        n_layers=1,
        num_classes=10,
    ):
        super(VanillaRNN, self).__init__()

        # parameters
        vocab_size, dim_size = preWeights.shape
        num_directions = 2 if bidirect else 1

        # embedding Layer
        # Because in tensor([[42, 17, 108, ...]]) Pytorch will
        # automatically do self.embedding[42], self.embedding[17], self.embedding[108], ...
        # So we have to let the index match our word_to_index
        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.embedding.load_state_dict({"weight": torch.FloatTensor(preWeights)})

        self.embedding.weight.requires_grad = fine_tune

        if layer_mode == "RNN":
            self.nn = nn.RNN(
                input_size=dim_size,
                hidden_size=hiddenSize,
                num_layers=n_layers,
                bidirectional=bidirect,
                batch_first=True,
            )
        elif layer_mode == "LSTM":
            self.nn = nn.LSTM(
                input_size=dim_size,
                hidden_size=hiddenSize,
                num_layers=n_layers,
                bidirectional=bidirect,
                batch_first=True,
            )
        elif layer_mode == "GRU":
            self.nn = nn.GRU(
                input_size=dim_size,
                hidden_size=hiddenSize,
                num_layers=n_layers,
                bidirectional=bidirect,
                batch_first=True,
            )
        else:
            raise ValueError("Invalid layer_mode variable.")
        self.fc = nn.Linear(hiddenSize * num_directions, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        nn_out, _ = self.nn(embeds)
        logits = self.fc(nn_out)
        return logits


def transform_data(
    data: List[Tuple[List[str], List[str]]],
    word_to_index,
    target_to_idx,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform data into tensors of indices."""
    X = []
    Y = []
    for sentence, targets in data:
        x_indices = [
            word_to_index.get(word, word_to_index["<PAD>"]) for word in sentence
        ]
        y_indices = [target_to_idx.get(tag, target_to_idx["<pad>"]) for tag in targets]
        X.append(x_indices)
        Y.append(y_indices)
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def get_embedding_matrix(word_to_index, embeddings, embedding_dim):
    vocab_size = len(word_to_index)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_to_index.items():
        if word in embeddings:
            # We change the row of value to embedding by the word_to_index
            embedding_matrix[idx] = embeddings[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix

def train_RNN(
    preWeights,
    train_loader,
    fine_tune: bool = False,
    layer_mode: str = "RNN",
    bidirect: bool = False,
    device: torch.device = torch.device("mps"),
    loss_record: list[list[float]] = [],
    hidden_size = 256,
    n_layers = 1,
    pad_tag_id = 0,
    num_classes = 10,
    learning_rate = 0.0001,
    loss_delta = 1e-3
):
    nn_model = VanillaRNN(
        preWeights=preWeights,
        layer_mode=layer_mode,
        fine_tune=fine_tune,
        bidirect=bidirect,
        hiddenSize=hidden_size,
        n_layers=n_layers,
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.Adam(nn_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_id)

    direction = "bidirectional" if bidirect else "unidirectional"
    logger.info(f"Start training for Vanilla{layer_mode} {direction}:")
    loss_record.append([200.0, 100.0])
    epoch = 0
    while loss_record[-1][-2] - loss_record[-1][-1] > loss_delta:
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = nn_model(inputs)

            predicted = torch.argmax(logits, dim=2) # Shape: [batch, seq_len]
            
            # Create a mask to ignore padding tokens in accuracy calculation
            # We don't want to penalize the model for predictions on padding
            mask = (labels != pad_tag_id)
            
            # Compare predictions to true labels where mask is True
            total_correct += (predicted[mask] == labels[mask]).sum().item()
            
            # The total number of samples is the number of non-padded tokens
            total_samples += mask.sum().item()

            logits = logits.view(-1, num_classes)
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_record[-1].append(avg_loss)

        epoch_acc = (total_correct / total_samples) * 100
        logger.info(
            f"Epoch {epoch+1}, "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Accuracy: {epoch_acc:.2f}%"
        )
        epoch += 1
    logger.info(f"Vanilla{layer_mode} {direction} {"embbed" if fine_tune else ""} training done")
    loss_record[-1] = loss_record[-1][2:]
    x = range(len(loss_record[-1]))
    plt.plot(x, loss_record[-1], label=f"{layer_mode} {direction} {"embbed" if fine_tune else ""}")

    save_path = f"./results/train/{layer_mode}_{direction}{"_embbed" if fine_tune else ""}.pth"
    torch.save(nn_model.state_dict(), save_path)
    logger.info(f"Model weights saved to {save_path}")

def eval_RNN(
    preWeights,
    valid_loader,
    processor: Preprocessor,
    layer_mode: str = "RNN",
    bidirect: bool = False,
    device: torch.device = torch.device("mps"),
    hidden_size = 256,
    n_layers = 1,
    num_classes = 10,
    fine_tune: bool = False
):
    nn_model = VanillaRNN(
        preWeights=preWeights,
        layer_mode=layer_mode,
        bidirect=bidirect,
        hiddenSize=hidden_size,
        n_layers=n_layers,
        num_classes=num_classes,
    ).to(device)

    direction = "bidirectional" if bidirect else "unidirectional"
    file_name = f"{layer_mode}_{direction}{"_embbed" if fine_tune else ""}"
    saved_weights_path = f"./results/train/{file_name}.pth"
    nn_model.load_state_dict(torch.load(saved_weights_path))
    logger.info(f"Generating validation file for {file_name}...")

    nn_model.eval()
    
    txt = f"./results/valid/{file_name}.txt"
    with open(txt, "w") as f:
        with torch.no_grad():
            idx_to_word = processor.idx_to_word
            idx_to_target = processor.idx_to_target
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Get model predictions
                logits = nn_model(inputs)
                predicted = torch.argmax(logits, dim=2)

                for i in range(inputs.shape[0]): # For each sentence
                    input_words = [idx_to_word[idx.item()] for idx in inputs[i]]
                    real_tags = [idx_to_target[idx.item()] for idx in labels[i]]
                    predicted_tags = [idx_to_target[int(idx.item())] for idx in predicted[i]]
                    
                    # Write results for one sentence
                    for word, r_tag, p_tag in zip(input_words, real_tags, predicted_tags):
                        # Don't write out the padding
                        # NOTE have to remove predicted tag <pad> too or it'll break the eval code
                        if word != "<PAD>" and p_tag != "<pad>":
                            f.write(f"{word} {r_tag} {p_tag}\n")
    
    logger.info(f"Generated {file_name}")
    logger.info(eval.evaluate_conll_file(open(txt, "r")))

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    cpu_count = os.cpu_count() or 4
    logger.info(f"Number of CPU cores available: {cpu_count}")

    processor = Preprocessor()
    train = processor.load_data("conll2003/train.txt")
    valid = processor.load_data("conll2003/valid.txt")
    test = processor.load_data("conll2003/test.txt")

    logger.info(
        f"Train size: {len(train)}, Valid size: {len(valid)}, Test size: {len(test)}"
    )

    train = transform_data(train, processor.word_to_idx, processor.target_to_idx)
    valid = transform_data(valid, processor.word_to_idx, processor.target_to_idx)
    test = transform_data(test, processor.word_to_idx, processor.target_to_idx)

    X_train, y_train = train[0], train[1]
    X_valid, y_valid = valid[0], valid[1]
    X_test, y_test = test[0], test[1]

    # For 14000 samples, 2000 batches means 14000/2000 = 7 ~ 8 batch size
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=cpu_count // 2,
        pin_memory=True
    )

    # Load Google's pre-trained Word2Vec embeddings
    embeddings = load_embeddings_from_bin_gz("GoogleNews-vectors-negative300.bin.gz")

    preWeights = get_embedding_matrix(
        processor.word_to_idx, embeddings, embeddings.vector_size
    )
    logger.info(f"Embedding matrix shape: {preWeights.shape}")

    hidden_size = 256
    n_layers = 1
    pad_tag_id = processor.target_to_idx["<pad>"]
    num_classes = len(processor.target_to_idx)
    learning_rate = 1e-4
    loss_delta = 1e-1 # Stop if training stops improving after this threshold

    os.makedirs("./results/train", exist_ok=True)
    os.makedirs("./results/valid", exist_ok=True)
    os.makedirs("./results/test", exist_ok=True)

    # Train 6 combinations of RNN hidden layers and uni/bi-directional
    loss_record = []
    combo = []
    
    for layer_mode in ["RNN", "LSTM", "GRU"]:
        for bidirect in [False, True]:
            combo.append((layer_mode, bidirect))
            train_RNN(
                preWeights,
                train_loader,
                False, # Freeze weights for the first 6 models
                layer_mode,
                bidirect,
                device,
                loss_record,
                hidden_size,
                n_layers,
                pad_tag_id,
                num_classes,
                learning_rate,
                loss_delta
            )
    
    # Pick the best performing one and train again while also updating embeddings
    # TODO Bonus point
    avg_losses = [rec[-1] for rec in loss_record]
    min_loss_i = avg_losses.index(min(avg_losses))
    train_RNN(
        preWeights,
        train_loader,
        True, # Update weights while training
        combo[min_loss_i][0],
        combo[min_loss_i][1],
        device,
        loss_record,
        hidden_size,
        n_layers,
        pad_tag_id,
        num_classes,
        learning_rate,
        loss_delta
    )

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(f"Training loss curves")
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(f"./results/train/train_loss_curve.png")
    logger.info("Loss curve saved. Check ./results/train/train_loss_curve.png")

    validation_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=cpu_count // 2,
        pin_memory=True
    )
    for mode, bidirect in combo:
        eval_RNN(
            preWeights,
            valid_loader,
            processor,
            mode,
            bidirect,
            device,
            hidden_size,
            n_layers,
            num_classes
        )
    
    eval_RNN(
        preWeights,
        valid_loader,
        processor,
        combo[min_loss_i][0],
        combo[min_loss_i][1],
        device,
        hidden_size,
        n_layers,
        num_classes
    )

if __name__ == "__main__":
    main()
