from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class RawDetections:
    """直接来自 detector.predict 的原始 list[np.ndarray]"""
    by_list: List[np.ndarray]
    by_name: Dict[str, np.ndarray]

    @staticmethod
    def from_list(result_list: List[np.ndarray], names: List[str]) -> "RawDetections":
        by_name = {names[i]: (result_list[i] if i < len(result_list) else np.empty((0,5), dtype=np.float32))
                   for i in range(len(names))}
        return RawDetections(by_list=[r.copy() for r in result_list], by_name=by_name)

@dataclass
class PostProcessResult:
    """后处理后的 bbox（key -> ndarray (N,5)）和其它中间信息（如颜色）"""
    boxes: Dict[str, np.ndarray]
    legend_colors: Dict[int, np.ndarray]  # legend index -> color (e.g. RGB)

@dataclass
class ChartElements:
    """最终结构化输出，按字段明确命名便于下游使用"""
    xlabel: np.ndarray
    ylabel: np.ndarray
    legend_labels: np.ndarray
    plot_area: np.ndarray
    other: np.ndarray
    # 可扩展字段...
    meta: Dict[str, Any] = None