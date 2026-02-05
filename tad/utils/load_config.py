import argparse
from pathlib import Path

import yaml


class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            try:
                key, value = kv.split("=", 1)
            except ValueError as err:
                msg = f"Invalid key-value pair: {kv!r}. Expected format 'key=value'."
                raise argparse.ArgumentError(self, msg) from err
            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError as err:
                # Raise an argparse-specific error with chaining so the original parsing error is visible
                raise argparse.ArgumentError(self, f"Failed to parse value for '{key}': {err}") from err
            options[key] = value
        setattr(namespace, self.dest, options)


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            # Suppress the original KeyError from being shown as the __getattr__ error context
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def fromfile(filename):
        try:
            with open(filename) as f:
                cfg_dict = yaml.safe_load(f)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Config file not found: {filename}") from err
        except yaml.YAMLError as err:
            raise ValueError(f"Failed to parse YAML config file '{filename}': {err}") from err
        return Config._dict_to_config(cfg_dict)

    @staticmethod
    def _dict_to_config(d):
        if isinstance(d, dict):
            cfg = Config()
            for k, v in d.items():
                cfg[k] = Config._dict_to_config(v)
            return cfg
        elif isinstance(d, list):
            return [Config._dict_to_config(v) for v in d]
        else:
            return d

    def merge_from_dict(self, options):
        for key, value in options.items():
            keys = key.split(".")
            current = self
            for k in keys[:-1]:
                try:
                    current = current[k]
                except Exception as err:
                    raise KeyError(f"Invalid config key: {key}") from err
            current[keys[-1]] = value

    @property
    def pretty_text(self):
        def _to_dict(d):
            if isinstance(d, Config):
                return {k: _to_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [_to_dict(v) for v in d]
            elif isinstance(d, Path):
                return str(d)
            else:
                return d

        return yaml.dump(_to_dict(self), sort_keys=False, default_flow_style=None)
