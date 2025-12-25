import os.path as osp
from typing import Tuple, Dict, Any

def load_config(filename: str) -> Tuple[Dict[Any, Any], str]:
    filename = osp.abspath(osp.expanduser(filename))
    with open(filename, encoding='utf-8') as f:
        cfg_text = f.read()
        
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to load YAML config files.")
    
    cfg_dict = yaml.safe_load(cfg_text)
    if cfg_dict is None:
        cfg_dict = {}

    return cfg_dict, cfg_text