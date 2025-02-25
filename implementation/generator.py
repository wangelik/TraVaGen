# import table
import torch
from model.scheduler import NoiseScheduler


def noisify(
    x: torch.Tensor, t: torch.Tensor, scheduler: NoiseScheduler
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion function to noisify data at specific intensity

    Args:
        x (torch.Tensor): data to be noisified
        t (torch.Tensor): noisify data at schedule positions t
        scheduler (NoiseScheduler): noise scheduler to define intensity

    Returns:
        x_noisified (torch.Tensor): noisified data at t
        noise (torch.Tensor): generated noise integrated into x

    """

    # generate random Gaussian noise
    noise = torch.randn_like(x, requires_grad=False)
    # create alpha schedule positions
    _, alpha = scheduler(t)
    # transform alpha schedule positions
    alpha = alpha.unsqueeze(1).to(x.device)
    # compute noisified (forward diffused) data
    x_noisified = torch.sqrt(alpha) * x + torch.sqrt(1.0 - alpha) * noise

    return x_noisified, noise
