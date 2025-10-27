import numpy as np
from scipy.optimize import curve_fit
from utils import safe_float, Abnormal_filter


def linear(x, a, b):
    return a * x + b


class PixelTransform:
    def __init__(self, cfg):
        """
        xp=a⋅log(xdata)+b
        xp=a⋅xdata+b
        """
        self.cfg = cfg
        self.model = linear
        self.params = (1, 0)
        self.filter = Abnormal_filter(cfg.pixel_transform.filter)
        self.log_flag = False


    def _tofloat(self, values):
        return np.array([safe_float(v) for v in values])
    
    def fit(self, bboxes, values, axis='x'):
        if axis == 'x':
            ys = (bboxes[: 0] + bboxes[:, 2]) / 2
        else:
            ys = (bboxes[:, 1] + bboxes[:, 3]) / 2
        xs = self._tofloat(values)
        xs, mask = self.filter.filter(xs)
        ys = ys[mask]
        xs_log = np.log(xs.copy())
        self.params = curve_fit(self.model, xs, ys)