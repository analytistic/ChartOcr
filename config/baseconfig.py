
import toml
import glob
import os

class AttrDict(dict):
    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict):
            return AttrDict(value)
        return value
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class BaseConfig(object):
    def __init__(self, path, args):
        cfg = self._load(path)
        if args:
            cfg.update({k: v for k, v in args.items() if v is not None})

        for k, v in cfg.items():
            if isinstance(v, dict):
                cfg[k] = AttrDict(v)
        self.__dict__.update(cfg)
        
    
    def _load(self, path):
        # read toml list and combine toml
        cfg = {}
        if os.path.isfile(path):
            cfg.update(toml.load(path))
        else:
            for file in glob.glob(os.path.join(path, '*.toml')):
                try:
                    cfg.update(toml.load(file))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        return cfg
