from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


DEFAULT_CHART_CLASSES = [
    'x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel',
    'chart_title', 'x_tick', 'y_tick', 'legend_patch', 'legend_label',
    'legend_title', 'legend_area', 'mark_label', 'value_label', 'y_axis_area',
    'x_axis_area', 'tick_grouping'
]

DEFAULT_OCR_CLASSES = [
    'xlabel', 'ylabel', 'x_title', 'y_title', 'legend_label',
]

@dataclass
class DetectionResult:
    bboxes_list: List[np.ndarray] = field(default_factory=list)
    class_names: List[str] = field(default_factory=lambda: DEFAULT_CHART_CLASSES.copy())
    _name2index: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not self.bboxes_list:
            self.bboxes_list = [np.empty((0, 5)) for _ in self.class_names]
        if len(self.bboxes_list) != len(self.class_names):
            raise ValueError(
                f"bboxes_list ({len(self.bboxes_list)})"
                f"class_names ({len(self.class_names)})"
            )
        self._name2index = {name: idx for idx, name in enumerate(self.class_names)}
        
    def get_bboxes(self, class_name: str) -> np.ndarray:
        try:
            return self.bboxes_list[self._name2index[class_name]]
        except Exception:
            return np.empty((0, 5))


    def set_bboxes(self, class_name: str, bboxes: np.ndarray):
        try:
            self.bboxes_list[self._name2index[class_name]] = bboxes
        except Exception:
            raise ValueError(f"Invalid class name: {class_name}")
        

    def copy(self) -> 'DetectionResult':
        return DetectionResult(
            bboxes_list=[b.copy() for b in self.bboxes_list],
            class_names=self.class_names.copy()
        )
    
    def to_list(self) -> List[np.ndarray]:
        return self.bboxes_list
    

@dataclass
class OcrResult:
    class_names: List[str] = field(default_factory=lambda: DEFAULT_OCR_CLASSES.copy())
    result_dict: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.result_dict = {name: [] for name in self.class_names}



