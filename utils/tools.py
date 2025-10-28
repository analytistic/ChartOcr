import numpy as np 
import re
import torch


def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [ndarray_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: ndarray_to_list(v) for k, v in obj.items()}
    else:
        return obj
    

    

def rgb_to_hsv_torch(self, rgb):
    """rgb: [N, 3], range [0,1]"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc, _ = rgb.max(dim=1)
    minc, _ = rgb.min(dim=1)
    v = maxc
    deltac = maxc - minc
    s = deltac / (maxc + 1e-6)

    h = torch.zeros_like(maxc)
    mask = deltac > 1e-6
    r_eq = (maxc == r) & mask
    g_eq = (maxc == g) & mask
    b_eq = (maxc == b) & mask

    h[r_eq] = ((g[r_eq] - b[r_eq]) / deltac[r_eq]) % 6
    h[g_eq] = ((b[g_eq] - r[g_eq]) / deltac[g_eq]) + 2
    h[b_eq] = ((r[b_eq] - g[b_eq]) / deltac[b_eq]) + 4
    h = h / 6.0

    hsv = torch.stack([h, s, v], dim=1)
    return hsv

    
def safe_float(s):
    s = s.replace('O', '0').replace('o', '0').replace('—', '-').replace('–', '-').strip()
    s = s.replace('E', 'e')
    if re.match(r'^e[-+]?\d+$', s):
        s = '1' + s
    match = re.match(r'^10[-‐–](\d+)$', s)
    if match:
        return float(f'1e-{match.group(1)}')
    try:
        return float(s)
    except:
        return float('inf')
    

def list_to_ndarray(obj):
    if isinstance(obj, list):
        if obj and isinstance(obj[0], list):
            return np.array(obj)
        elif obj and not isinstance(obj[0], str):
            return np.array(obj)
        else:
            return obj
    elif isinstance(obj, dict):
        return {k: list_to_ndarray(v) for k, v in obj.items()}
    else:
        return obj

class Abnormal_filter:
    """
    异常过滤
    """

    def __init__(self, cfg=None):

        self.cfg = cfg
        self.method = cfg.method if cfg is not None else 'iqr'
        self.thr = cfg.thr if cfg is not None else 1.5



    @staticmethod
    def mad_mask(x, thr=3.5):
        x = np.asarray(x)
        median = np.median(x)
        abs_dev = np.abs(x - median)
        mad = np.median(abs_dev)
        if mad == 0:
            return np.zeros_like(x, dtype=bool)
        modified_z_score = 0.6745 * abs_dev / mad
        return modified_z_score > thr
    

    @staticmethod
    def iqr_mask(x, k=1.5):
        x = np.asarray(x)
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros_like(x, dtype=bool)
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        return (x < lower) | (x > upper)
    
    @staticmethod
    def zscore_mask(x, thr=3.0):
        x = np.asarray(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return np.zeros_like(x, dtype=bool)
        z_scores = (x - mean) / std
        return np.abs(z_scores) > thr
    

    def detect(self, x):
        if self.method == 'mad':
            return self.mad_mask(x, thr=self.thr)
        elif self.method == 'iqr':
            return self.iqr_mask(x, k=self.thr)
        elif self.method == 'zscore':
            return self.zscore_mask(x, thr=self.thr)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        

    def filter(self, x):
        """
        returns filtered x, indices
        """
        x = np.asarray(x)
        mask = self.detect(x)
        return x[~mask], np.where(~mask)[0]


