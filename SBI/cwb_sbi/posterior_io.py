"""
Saving and loading of trained (pickled) NPE posteriors.
"""

# general
import pickle
import sys

# local
from cwb_sbi import embedding_nets


# The posteriors were pickled when the embedding networks lived in a top-level
# module called `helper`; pickle looks the classes up under that name, so it
# is registered as an alias of embedding_nets.
sys.modules.setdefault("helper", embedding_nets)


def load_posterior(path, device=None):
    """
    Load a pickled NPE posterior.

    Args:
        path: Path of the .pkl file.
        device: If given, the posterior network is moved to this device (best
            effort, as `_net` is internal to sbi).

    Returns:
        The posterior.
    """
    with open(path, "rb") as file:
        posterior = pickle.load(file)
    if device is not None:
        try:
            posterior._net.to(device)
        except Exception:
            pass
    return posterior


def save_posterior(posterior, path):
    """Pickle a posterior to path."""
    with open(path, "wb") as file:
        pickle.dump(posterior, file)
