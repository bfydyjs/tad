def get_custom_config(cfg, params=None, gflops=None):
    # 1. return full config
    parts = str(cfg.work_dir).strip("/").split("/")
    cfg.group = "_".join(parts[-3:-1])
    if params is not None:
        cfg.Params = f"{params / 1e6:.2f} M"
    if gflops is not None:
        cfg.GFLOPs = gflops
    return cfg


# 2. return optional selected fields
