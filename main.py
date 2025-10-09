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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

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
        self.output = nn.Softmax(dim=-1)

    def forward(self, x):
        embeds = self.embedding(x)
        rnn_out, hidden = self.rnn(embeds)
        logits = self.fc(rnn_out)
        return self.output(logits)


def transform_data(
    data: List[Tuple[List[str], List[str]]],
    embeddings: dict = {},
    word_to_index: dict = {},
    target_to_idx: dict = {},
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


def main():

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

    # Load Google's pre-trained Word2Vec embeddings
    embeddings = load_embeddings_from_bin_gz("GoogleNews-vectors-negative300.bin.gz")

    input_size = embeddings.vector_size
    hidden_size = 256
    n_layers = 1
    num_classes = len(processor.target_to_idx)
    learning_rate = 0.0001
    batch_size = 2000
    num_epochs = 6000

    rnn = VanillaRNN(
        preWeights=embeddings.vectors,
        preTrain=True,
        bidirect=False,
        hiddenSize=hidden_size,
        n_layers=n_layers,
        num_classes=num_classes,
    )
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    rnn.to(device)


if __name__ == "__main__":
    main()
