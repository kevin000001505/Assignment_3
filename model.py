import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """RNN-based sequence tagger using pretrained embeddings.

    Supports RNN, LSTM, and GRU recurrent layers, optional bidirectionality,
    and optional fine-tuning of the embedding weights.
    """

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
        """Initialize the VanillaRNN model.

        Args:
            preWeights (ndarray or torch.Tensor): Pretrained embedding weights with shape (vocab_size, embedding_dim).
            layer_mode (str): Type of recurrent layer, one of "RNN", "LSTM", or "GRU".
            fine_tune (bool): If True, allow embedding weights to be updated during training.
            bidirect (bool): If True, use a bidirectional recurrent layer.
            hiddenSize (int): Hidden size of the recurrent layer.
            n_layers (int): Number of recurrent layers.
            num_classes (int): Number of output classes per time step (vocabulary tags).
        """
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
        """Perform a forward pass through the model.

        Args:
            x (torch.LongTensor): Input tensor of token indices with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for each token with shape (batch_size, seq_len, num_classes).
        """
        embeds = self.embedding(x)
        nn_out, _ = self.nn(embeds)
        logits = self.fc(nn_out)
        return logits
