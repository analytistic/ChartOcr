from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from module.detector.chart_element_detector.utils.types import DetectionResult
import numpy as np


@dataclass
class LegendLabel:
    text: List[str]             
    bbox: List[np.ndarray]      
    color: List[np.ndarray]
    
@dataclass
class LegendInfo:
    area: np.ndarray            
    label: LegendLabel

@dataclass
class AxisLabel:
    value: np.ndarray           
    bbox: List[np.ndarray]

@dataclass
class AxisInfo:
    x_title: str                 
    y_title: str                 
    x_label: AxisLabel          
    y_label: AxisLabel

@dataclass
class PlotInfo:
    bbox: np.ndarray

@dataclass
class ChartElementResult:
    """
    chart detector result
    {
        "legends":{
            "area": np.ndarray,
            "label":{
                "text": [str],
                "bbox": [np.ndarray],
                "color": [np.ndarray],
            },
        },
        "axis":{
            "x_title": (str),
            "y_title": (str),
            "x_label":{
                "value": np.ndarray,
                "bbox": [np.ndarray],
            },
            "y_label":{
                "value": np.ndarray,
                "bbox": [np.ndarray],
            },
        },
        "plot":{
            "bbox": np.ndarray,
        }    
    }
    """
    legends: LegendInfo
    axis: AxisInfo
    plot: PlotInfo

    @classmethod
    def from_detectionresult(
        cls,
        detection_result: DetectionResult,
        legend_colors: np.ndarray,
        text: Any,
    ):
        pass










class LineResult:
    """
    line extractor result
    """