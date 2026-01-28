import yaml
import argparse

class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, value = kv.split('=', 1)
            try:
                value = yaml.safe_load(value)
            except Exception:
                pass
            options[key] = value
        setattr(namespace, self.dest, options)

class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    @staticmethod
    def fromfile(filename):
        with open(filename, 'r') as f:
            cfg_dict = yaml.safe_load(f)
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
            keys = key.split('.')
            current = self
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value

    @property
    def pretty_text(self):
        def _to_dict(d):
            if isinstance(d, Config):
                return {k: _to_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [_to_dict(v) for v in d]
            else:
                return d
        return yaml.dump(_to_dict(self), sort_keys=False, default_flow_style=None)
