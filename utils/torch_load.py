import torch


def load_torch_file(path, map_location=None):
    """Load trusted local torch files across torch versions."""
    try:
        if map_location is None:
            return torch.load(path, weights_only=False)
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions do not support the weights_only argument.
        if map_location is None:
            return torch.load(path)
        return torch.load(path, map_location=map_location)
