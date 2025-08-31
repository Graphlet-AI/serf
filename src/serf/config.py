"""Configuration management for SERF."""

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class Config:
    """Configuration management for Abzu.

    Loads configuration from config.yml and provides methods to access
    configuration values. Configuration values can use variable interpolation
    with the ${variable_name} syntax.
    """

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the Config object.

        Parameters
        ----------
        config_file : Optional[str], optional
            Path to the configuration file to load. If None, loads from the default
            location, by default None
        """
        if config_file is None:
            # Try several locations for the config file
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            possible_locations = [
                os.path.join(project_root, "config.yml"),  # Root directory
                os.path.join(os.path.dirname(__file__), "config.yml"),  # serf directory
            ]

            for location in possible_locations:
                if os.path.exists(location):
                    config_file = location
                    break

            if config_file is None:
                config_file = possible_locations[0]  # Default to project root

        self.config_file = config_file
        self._config: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        """Reload the configuration file."""
        try:
            with open(self.config_file, "r") as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def _resolve_value(self, value: str) -> str:
        """Resolve a variable reference in a string.

        Parameters
        ----------
        value : str
            The value to resolve, which may contain variable references like ${key.path}

        Returns
        -------
        str
            The resolved value
        """
        # Extract variable references like ${key.path}
        var_pattern = re.compile(r"\${([^}]+)}")
        matches = var_pattern.findall(value)

        # If no variables found, return the original value
        if not matches:
            return value

        # Replace each variable with its resolved value
        result = value
        for match in matches:
            var_value = self.get(match)
            if var_value is not None:
                result = result.replace(f"${{{match}}}", str(var_value))

        # If there are still variables to resolve, recurse
        if var_pattern.search(result):
            return self._resolve_value(result)

        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value.

        Parameters
        ----------
        key_path : str
            Dot-separated path to the configuration value
        default : Any, optional
            Default value to return if the key is not found. If None and the key
            is not found, raises KeyError.

        Returns
        -------
        Any
            The configuration value

        Raises
        ------
        KeyError
            If the key is not found and no default is provided
        """
        # Split the key path into parts
        parts = key_path.split(".")

        # Start at the root of the configuration
        config = self._config
        traversed_path = []

        # Traverse the configuration tree
        for part in parts:
            traversed_path.append(part)
            if not isinstance(config, dict):
                if default is None:
                    raise KeyError(
                        f"Configuration key '{key_path}' not found. "
                        f"'{'.'.join(traversed_path[:-1])}' is not a dictionary."
                    )
                return default
            if part not in config:
                if default is None:
                    available_keys = list(config.keys()) if isinstance(config, dict) else []
                    raise KeyError(
                        f"Configuration key '{key_path}' not found at '{'.'.join(traversed_path)}'. "
                        f"Available keys: {available_keys}"
                    )
                return default
            config = config[part]

        # Resolve variables in string values
        if isinstance(config, str):
            return self._resolve_value(config)
        elif isinstance(config, list) and all(isinstance(item, str) for item in config):
            return [self._resolve_value(item) for item in config]

        return config

    def get_path(self, key_path: str, default: Optional[str] = None) -> Union[Path, list[Path]]:
        """Get a configuration value as a Path object or list of Path objects.

        Parameters
        ----------
        key_path : str
            Dot-separated path to the configuration value
        default : Optional[str], optional
            Default value to return if the key is not found, by default None

        Returns
        -------
        Union[Path, list[Path]]
            The configuration value as a Path object or list of Path objects
        """
        value = self.get(key_path, default)
        if value is None:
            raise ValueError(f"Configuration value not found: {key_path}")

        if isinstance(value, list):
            return [Path(item) for item in value]

        return Path(value)

    def expand_variables(self, value: str) -> str:
        """Expand variables in a string value.

        Parameters
        ----------
        value : str
            String that may contain variable references like ${key.path}

        Returns
        -------
        str
            String with variables expanded
        """
        if isinstance(value, str):
            return self._resolve_value(value)
        return str(value)


# Singleton instance for convenience
config = Config()
