# import table
import torch
from torch import nn


class NoiseScheduler(nn.Module):
    """Class to create noise schedules for diffusion models

    Methods:
        forward: standard torch forward method

    Notes:
        * currently only supports linear noise schedule

    """

    def __init__(self, n_steps: int, start: float = 1e-4, end: float = 0.02):
        """
        Args:
            n_steps (int): number of unique time steps for noise schedule
            start (float): starting beta for noise distribution
            end (float): final beta for noise distribution

        """

        super().__init__()
        # create beta noise schedule
        self.beta = torch.linspace(start, end, n_steps, requires_grad=False)
        # create single alpha transformation
        alpha_t = 1.0 - self.beta
        # compute cumulative alpha schedule
        self.alpha = torch.cumprod(alpha_t, dim=0).requires_grad_(False)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward method to create sampled noise schedule positions

        Args:
            t (torch.Tensor): time steps for selecting schedule positions

        Returns:
            output (torch.Tensor): sampled beta and alpha schedule positions

        """

        return self.beta[t], self.alpha[t]
