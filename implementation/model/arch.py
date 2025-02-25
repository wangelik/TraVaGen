# import table
import math

import torch
import torch.nn.functional as F
from torch import nn


class TimeEmbedding(nn.Module):
    """Class to create timestamp embedding layer

    Methods:
        forward: transform timestamp to unique fixed embedding

    """

    def __init__(self, n_steps: int, embed_dim: int):
        """
        Args:
            n_steps (int): number of unique time steps to embed
            embed_dim (int): target dimension of timestamp embedding

        """

        super().__init__()
        # create target positions
        positions = torch.arange(n_steps).unsqueeze(1).float()
        # create scaling factors
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        # init embeddings w/o gradient information
        embeddings = torch.zeros(n_steps, embed_dim, requires_grad=False)
        # generate scaled sin/cos embeddings
        embeddings[:, 0::2] = torch.sin(positions * div_term)
        embeddings[:, 1::2] = torch.cos(positions * div_term)
        self.embeddings = embeddings

    def forward(self, time_steps: torch.Tensor, device: str) -> torch.Tensor:
        """Forward method to transform raw time steps into embeddings

        Args:
            time_steps (torch.Tensor): tensor of integer timestamps to embed
            device (str): device to store target embeddings on

        Returns:
            time_steps_embedded (torch.Tensor): tensor of target embeddings

        """

        # encode raw time_steps by using embedding matrix
        time_steps_embedded = self.embeddings[time_steps].to(device)

        return time_steps_embedded


class Denoiser(nn.Module):
    """Class to create denoising diffusion neural network

    Methods:
        forward: standard torch forward method

    """

    def __init__(self, input_dim: int, n_steps: int, expander: int = 2):
        """
        Args:
            input_dim (int): dimension of input data (variant corpus)
            n_steps (int): number of unique time steps to embed
            expander (int): upscaling factor for middle data layer size

        """

        super().__init__()
        # define upscaling factor for middle layer
        expander_dim = round(input_dim * expander / 2) * 2
        # define first open layer
        self.open = nn.Linear(input_dim, expander_dim)
        # define middle expanded layer
        self.expand = nn.Linear(expander_dim, expander_dim)
        # define last closing layer
        self.close = nn.Linear(expander_dim, input_dim)
        # define time step embedding layers
        self.time_embedding = TimeEmbedding(n_steps, expander_dim)
        self.embedding_transform = nn.Linear(expander_dim, expander_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward method to combine and transform data and time steps

        Args:
            x (torch.Tensor): data input (encoded variants)
            t (torch.Tensor): time step input

        Returns:
            output (torch.Tensor): combined and processed signal

        """

        # open data layer transmission
        x = self.open(x)
        x = F.relu(x)
        # time embedding transmission
        t = self.time_embedding(t, x.device)
        t = self.embedding_transform(t)
        t = F.relu(t)
        # combine signals
        x = x + t
        # middle expanding layer transmission
        x = self.expand(x)
        x = F.relu(x)
        # closing layer transmission
        output = self.close(x)

        return output
