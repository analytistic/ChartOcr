from .tools import (
    Abnormal_filter,
)


mad_mask = Abnormal_filter.mad_mask
iqr_mask = Abnormal_filter.iqr_mask
zscore_mask = Abnormal_filter.zscore_mask

__all__ = [
    'Abnormal_filter',
    'mad_mask',
    'iqr_mask',
    'zscore_mask',
]