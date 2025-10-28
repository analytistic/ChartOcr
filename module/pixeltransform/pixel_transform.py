import numpy as np
from scipy.optimize import curve_fit
from utils import safe_float, Abnormal_filter


def linear(x, a, b):
    return a * x + b

def re_linear(x, log_flag, a, b):
    if log_flag:
        return np.exp((x - b + 1e-6)/a)
    else:
        return (x - b + 1e-6)/a


class PixelTransform:
    def __init__(self, cfg):
        """
        xp=a⋅log(xdata)+b xdata = exp((xp - b) / a)
        xp=a⋅xdata+b
        """
        self.cfg = cfg
        self.model = linear
        self.remodel = re_linear
        self.params = (1, 0)
        self.filter = Abnormal_filter(cfg.pixel_transform.filter)
        self.log_flag = False

    def _tofloat(self, values):
        return np.array([safe_float(v) for v in values])
    
    def fit(self, bboxes, values, axis='x'):
        if axis == 'x':
            ys = (bboxes[:, 0] + bboxes[:, 2]) / 2
        else:
            ys = (bboxes[:, 1] + bboxes[:, 3]) / 2
        xs = self._tofloat(values)
        mask_inf = np.isinf(xs)
        xs = xs[~mask_inf]
        ys = ys[~mask_inf]
        sort_index = np.argsort(ys)

        if axis == 'y':
            sort_index = sort_index[::-1]
        ys = ys[sort_index]
        xs = xs[sort_index]
        xs, mask = self.filter.filter(xs)
        ys = ys[mask]

        keep = []
        prev = float('inf')
        for i, v in reversed(list(enumerate(xs))):
            keep.append(i)
            if v <= prev:
                prev = v
            else:
                prev = -v
                xs[i] = -v 
        xs = xs[keep]
        ys = ys[keep]
        if len(xs) >= 4:
            xs = xs[1:-1]
            ys = ys[1:-1] 
        xs_log = np.log(xs.copy())
        if np.any(xs <= 0) or np.any(np.isnan(xs_log)) or np.any(np.isinf(xs_log)):
            params = curve_fit(self.model, xs, ys)
            self.params = params[0]
            self.log_flag = False
            re_error = np.abs(self.remodel(ys, self.log_flag, *self.params) - xs)
            return re_error.mean(), re_error.std(), re_error.max(), np.max(np.abs(xs))
        params1 = curve_fit(self.model, xs, ys)
        params2 = curve_fit(self.model, xs_log, ys)
        perr1 = np.sqrt(np.diag(params1[1]))
        perr2 = np.sqrt(np.diag(params2[1]))
        if perr1[0] < perr2[0]:
            self.log_flag = False
            self.params = params1[0]
        else:
            self.log_flag = True
            self.params = params2[0]
        re_error = np.abs(self.remodel(ys, self.log_flag, *self.params) - xs)

        return re_error.mean(), re_error.std(), re_error.max(), np.max(np.abs(xs))

    def pridect(self, pixel):
        return self.remodel(pixel, self.log_flag, *self.params)
        




