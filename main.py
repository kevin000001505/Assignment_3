import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from typing import List, Tuple
from pre_process import Preprocessor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import conll2003.conlleval as eval
from model import VanillaRNN
from utils import (
    get_embedding_matrix,
    transform_data,
    load_embeddings_from_bin_gz,
    configuration,
)

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


def calculate_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, pad_tag_id: int
) -> float:
    total_correct = 0
    total_samples = 0

    predicted = torch.argmax(logits, dim=1)  # Shape: [batch * seq_len]

    # Create a mask to ignore padding tokens in accuracy calculation
    # We don't want to penalize the model for predictions on padding
    mask = labels != pad_tag_id

    # Compare predictions to true labels where mask is True
    total_correct += (predicted[mask] == labels[mask]).sum().item()

    # The total number of samples is the number of non-padded tokens
    total_samples += mask.sum().item()
    epoch_acc = (total_correct / total_samples) * 100
    return epoch_acc


def train_RNN(
    preWeights: torch.Tensor,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    fine_tune: bool = False,
    layer_mode: str = "RNN",
    bidirect: bool = False,
    device: torch.device = torch.device("mps"),
    loss_record: list[list[float]] = [],
    hidden_size=256,
    n_layers=1,
    pad_tag_id=0,
    num_classes=10,
    learning_rate=0.0001,
    loss_delta=1e-3,
    lr_step_size: int = 20,
    lr_gamma: float = 0.5,
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
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_id)

    direction = "bidirectional" if bidirect else "unidirectional"
    logger.info(f"Start training for Vanilla{layer_mode} {direction}:")
    loss_record.append([200.0, 100.0])
    epoch = 0

    # Early stopping condition: stop if the improvement in loss is less than loss_delta
    while loss_record[-1][-2] - loss_record[-1][-1] > loss_delta:
        nn_model.train()
        epoch_loss = 0.0
        val_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = nn_model(inputs)

            logits = logits.view(-1, num_classes)
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        nn_model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = nn_model(inputs)

                outputs = outputs.view(-1, num_classes)
                labels = labels.view(-1)

                # This return only float values, not tensors
                val_loss += criterion(outputs, labels).item()

        epoch_acc = calculate_accuracy(outputs, labels, pad_tag_id)

        avg_loss = epoch_loss / len(train_loader)

        # Change to validation loss for early stopping
        loss_record[-1].append(val_loss)

        # Step the LR scheduler based on epoch and log current LR
        scheduler.step()
        logger.info(
            f"Epoch {epoch+1}, "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Val Accuracy: {epoch_acc:.6f}% "
            f"Val Loss: {val_loss/len(valid_loader):.4f} "
        )
        epoch += 1
    logger.info(
        f"Vanilla{layer_mode} {direction} {"embbed" if fine_tune else ""} training done"
    )
    loss_record[-1] = loss_record[-1][2:]
    x = range(len(loss_record[-1]))
    plt.plot(
        x,
        loss_record[-1],
        label=f"{layer_mode} {direction} {"embbed" if fine_tune else ""}",
    )

    save_path = (
        f"./results/train/{layer_mode}_{direction}{"_embbed" if fine_tune else ""}.pth"
    )
    torch.save(nn_model.state_dict(), save_path)
    logger.info(f"Model weights saved to {save_path}")


def eval_RNN(
    preWeights: torch.Tensor,
    test_loader: DataLoader,
    processor: Preprocessor,
    layer_mode: str = "RNN",
    bidirect: bool = False,
    device: torch.device = torch.device("mps"),
    hidden_size=256,
    n_layers=1,
    num_classes=10,
    fine_tune: bool = False,
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
    logger.info(f"Generating test file for {file_name}...")

    nn_model.eval()

    txt = f"./results/test/{file_name}.txt"
    with open(txt, "w") as f:
        with torch.no_grad():
            idx_to_word = processor.idx_to_word
            idx_to_target = processor.idx_to_target
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Get model predictions
                logits = nn_model(inputs)
                predicted = torch.argmax(logits, dim=2)

                for i in range(inputs.shape[0]):  # For each sentence
                    input_words = [idx_to_word[idx.item()] for idx in inputs[i]]
                    real_tags = [idx_to_target[idx.item()] for idx in labels[i]]
                    predicted_tags = [
                        idx_to_target[int(idx.item())] for idx in predicted[i]
                    ]

                    # Write results for one sentence
                    for word, r_tag, p_tag in zip(
                        input_words, real_tags, predicted_tags
                    ):
                        # Don't write out the padding
                        # NOTE have to remove predicted tag <pad> too or it'll break the eval code
                        if word != "<PAD>" and p_tag != "<pad>":
                            f.write(f"{word} {r_tag} {p_tag}\n")

    logger.info(f"Generated {file_name}")
    logger.info(eval.evaluate_conll_file(open(txt, "r")))


def main():
    device, cpu_count = configuration()
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
        batch_size=64,
        shuffle=True,
        num_workers=cpu_count // 2,
        pin_memory=True if device.type == "cuda" else False,
    )

    validation_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(
        validation_dataset,
        batch_size=512,  # This is for validation so just increase until full memory usage
        shuffle=True,
        num_workers=cpu_count // 2,
        pin_memory=True if device.type == "cuda" else False,
    )

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,  # This is for testing so just increase until full memory usage
        shuffle=True,
        num_workers=cpu_count // 2,
        pin_memory=True if device.type == "cuda" else False,
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
    loss_delta = 1e-5  # Stop if training stops improving after this threshold

    os.makedirs("./results/train", exist_ok=True)
    os.makedirs("./results/test", exist_ok=True)

    # Train 6 combinations of RNN hidden layers and uni/bi-directional
    loss_record = []

    # store [Model, bidirectional]
    combo = []

    for layer_mode in ["RNN", "LSTM", "GRU"]:
        for bidirect in [False, True]:
            combo.append((layer_mode, bidirect))
            train_RNN(
                preWeights,
                train_loader,
                valid_loader,
                False,  # Freeze weights for the first 6 models
                layer_mode,
                bidirect,
                device,
                loss_record,
                hidden_size,
                n_layers,
                pad_tag_id,
                num_classes,
                learning_rate,
                loss_delta,
            )

    # Pick the best performing one and train again while also updating embeddings
    # TODO Bonus point
    avg_losses = [rec[-1] for rec in loss_record]
    min_loss_i = avg_losses.index(min(avg_losses))
    train_RNN(
        preWeights,
        train_loader,
        valid_loader,
        True,  # Update weights while training
        combo[min_loss_i][0],
        combo[min_loss_i][1],
        device,
        loss_record,
        hidden_size,
        n_layers,
        pad_tag_id,
        num_classes,
        learning_rate,
        loss_delta,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training loss curves")
    plt.legend()
    plt.xscale("log")
    plt.grid(True)
    plt.savefig("./results/train/train_loss_curve.png")
    logger.info("Loss curve saved. Check ./results/train/train_loss_curve.png")

    for mode, bidirect in combo:
        eval_RNN(
            preWeights,
            test_loader,
            processor,
            mode,
            bidirect,
            device,
            hidden_size,
            n_layers,
            num_classes,
        )

    logger.info(
        f"Evaluating the best model after fine-tuning embeddings on test set: Model: {combo[min_loss_i][0]} {'bidirectional' if combo[min_loss_i][1] else 'unidirectional'}"
    )


if __name__ == "__main__":
    main()
