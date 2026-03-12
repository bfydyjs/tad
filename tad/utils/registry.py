class Registry:
    """A lightweight registry implementation."""

    def __init__(self, name):
        self.name = name
        self.module_dict = {}

    def register_module(self, name=None):
        def _register(cls):
            module_name = name if name is not None else cls.__name__
            if module_name in self.module_dict:
                raise KeyError(f"'{module_name}' is already registered in {self.name}")
            self.module_dict[module_name] = cls
            return cls

        return _register

    def get(self, key):
        if key not in self.module_dict:
            raise KeyError(f"'{key}' is not registered in {self.name}")
        return self.module_dict[key]

    def build(self, cfg, default_args=None):
        """Build a module from config dict."""
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise TypeError("cfg must be a dict containing the key 'type'")

        args = cfg.copy()
        obj_type = args.pop("type")

        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
        else:
            obj_cls = obj_type

        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)

        return obj_cls(**args)
