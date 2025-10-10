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
        preTrain=True,
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
        # Because in tensor([[42, 17, 108, ...]]) Pytorch will automatically do self.embedding[42], self.embedding[17], self.embedding[108], ...
        # So we have to let the index match our word_to_index
        self.embedding = nn.Embedding(vocab_size, dim_size)
        self.embedding.load_state_dict({"weight": torch.FloatTensor(preWeights)})

        if preTrain:
            self.embedding.weight.requires_grad = False

        self.rnn = nn.RNN(
            input_size=dim_size,
            hidden_size=hiddenSize,
            num_layers=n_layers,
            bidirectional=bidirect,
            batch_first=True,
        )
        self.fc = nn.Linear(hiddenSize * num_directions, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, hidden = self.rnn(embeds)
        logits = self.fc(rnn_out)
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

    train = transform_data(train, processor.word_to_index, processor.target_to_idx)
    valid = transform_data(valid, processor.word_to_index, processor.target_to_idx)
    test = transform_data(test, processor.word_to_index, processor.target_to_idx)

    X_train, y_train = train[0], train[1]
    X_valid, y_valid = valid[0], valid[1]
    X_test, y_test = test[0], test[1]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=2000, shuffle=True, num_workers=cpu_count // 2
    )

    # Load Google's pre-trained Word2Vec embeddings
    embeddings = load_embeddings_from_bin_gz("GoogleNews-vectors-negative300.bin.gz")

    preWeights = get_embedding_matrix(
        processor.word_to_index, embeddings, embeddings.vector_size
    )
    logger.info(f"Embedding matrix shape: {preWeights.shape}")

    hidden_size = 256
    n_layers = 1
    pad_tag_id = processor.target_to_idx["<pad>"]
    num_classes = len(processor.target_to_idx)
    learning_rate = 0.0001
    num_epochs = 6000

    rnn = VanillaRNN(
        preWeights=preWeights,
        preTrain=True,
        bidirect=False,
        hiddenSize=hidden_size,
        n_layers=n_layers,
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_id)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = rnn(inputs)

            logits = logits.view(-1, num_classes)
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch+1}, Avg Loss: {epoch_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    main()
