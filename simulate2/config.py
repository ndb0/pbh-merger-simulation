import yaml
import os

class ConfigNamespace:
    """
    A simple class that recursively converts dictionaries into objects,
    allowing you to access nested keys using dot notation (e.g., cfg.compute.use_gpu).
    """
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                # If the value is a dictionary, convert it recursively
                setattr(self, key, ConfigNamespace(value))
            else:
                # Otherwise, just set the attribute
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"ConfigNamespace({self.__dict__})"

def load_config(path: str) -> ConfigNamespace:
    """
    Loads a YAML configuration file from the given path and converts it
    into a nested ConfigNamespace object.

    Args:
        path: The full path to the YAML configuration file.

    Returns:
        A ConfigNamespace object representing the loaded configuration.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)

    return ConfigNamespace(config_data)
