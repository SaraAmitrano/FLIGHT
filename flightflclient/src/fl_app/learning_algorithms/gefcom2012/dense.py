import torch
from torch import nn


class GEFCom2012Dense(nn.Module):
    def __init__(
        self, input_dim: int = 10, hidden_dim: int = 256, output_dim: int = 1
    ) -> None:
        super().__init__()  # type: ignore

        input_dims = [input_dim, hidden_dim, hidden_dim // 2, hidden_dim // 4, 32, 16]
        output_dims = [*input_dims[1:], output_dim]

        modules: list[nn.Module] = []
        for i in range(len(input_dims) - 1):
            # modules.append(nn.LayerNorm(input_dims[i]))
            # modules.append(nn.Dropout(0.1))
            modules.append(nn.Linear(input_dims[i], output_dims[i]))
            # modules.append(nn.ReLU())
            modules.append(nn.Tanh())

        self.seq = nn.Sequential(*modules)

        self.output = nn.Linear(input_dims[-1], output_dims[-1], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.seq(x)
        return self.output(x)
