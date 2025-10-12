import torch
import torch.nn as nn


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
