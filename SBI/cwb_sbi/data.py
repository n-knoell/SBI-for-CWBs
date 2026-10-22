"""
Loading of simulation banks and test sets saved as {"theta": ..., "x": ...}.
"""

# neural networks
import torch


def load_simulations(path, map_location="cpu"):
    """
    Load a simulation file and drop the T = 0 snapshot of every simulation,
    which is just the homogeneous initial condition and carries no
    information about the parameters.

    Args:
        path: Path of the .pt file.
        map_location: Passed to torch.load.

    Returns:
        (theta [N, 7], x [N, T, H, W]).
    """
    simulations = torch.load(path, map_location=map_location)
    theta = simulations.get("theta")
    x = simulations.get("x")[:, 1:, :, :]
    return theta, x
