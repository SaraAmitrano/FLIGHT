import torch
from torch import nn


class ASHRAELSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()  # type: ignore

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # If hidden and cell states are not provided, initialize them as zeros
        # x: (batch_size, seq_len, input_dim)
        # with seq_len the temporal window length (e.g., 24 hours)
        # input_dim is the number of features
        hc_0 = None if h0 is None or c0 is None else (h0, c0)

        # Forward pass through LSTM
        self.lstm.flatten_parameters()
        out: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = self.lstm(x, hc_0)
        return self.fc(out[0][:, -1, :])  # Selecting the last output
