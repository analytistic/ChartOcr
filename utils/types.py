from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from module.detector.chart_element_detector.utils.types import DetectionResult, OcrResult
import numpy as np


@dataclass
class LegendLabel:
    text: List[str]             
    bbox: np.ndarray      
    color: np.ndarray
    
@dataclass
class LegendInfo:
    area: np.ndarray            
    label: LegendLabel

@dataclass
class AxisLabel:
    value: List[str]         
    bbox: np.ndarray

@dataclass
class AxisInfo:
    x_title: List[str]                 
    y_title: List[str]                 
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
            "legendlabel":{
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
        ocr_result: OcrResult,
    ):
        legends = LegendInfo(
            area=detection_result.get_bboxes("legend_area"),
            label=LegendLabel(
                text=ocr_result.result_dict.get("legend_label", []),
                bbox=detection_result.get_bboxes("legend_label"),
                color=legend_colors
            )
        )

        axis = AxisInfo(
            x_title=ocr_result.result_dict.get("x_title", []),
            y_title=ocr_result.result_dict.get("y_title", []),
            x_label=AxisLabel(
                value=ocr_result.result_dict.get("xlabel", []),
                bbox=detection_result.get_bboxes("xlabel"),
            ),
            y_label=AxisLabel(
                value=ocr_result.result_dict.get("ylabel", []),
                bbox=detection_result.get_bboxes("ylabel"),
            ),
        )

        plot = PlotInfo(
            bbox=detection_result.get_bboxes("plot_area")[0],
        )

        return cls(
            legends=legends,
            axis=axis,
            plot=plot
        )







class LineResult:
    """
    line extractor result
    """