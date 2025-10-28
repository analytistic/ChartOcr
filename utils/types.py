from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from module.detector.chart_element_detector.utils.types import DetectionResult, OcrResult
import numpy as np
from .tools import ndarray_to_list, list_to_ndarray
import json
import pandas as pd


@dataclass
class LegendLabel:
    text: List[str] = field(default_factory=lambda: [""])            
    bbox: np.ndarray = field(default_factory=lambda: np.empty((0,5)))
    color: np.ndarray = field(default_factory=lambda: np.empty((0,3)))
    
@dataclass
class LegendInfo:
    area: np.ndarray = field(default_factory=lambda: np.empty((0,5)))     
    label: LegendLabel = field(default_factory=LegendLabel)

@dataclass
class AxisLabel:
    value: List[str] = field(default_factory=lambda: [""])      
    bbox: np.ndarray = field(default_factory=lambda: np.empty((0,5)))

@dataclass
class AxisInfo:
    x_title: List[str] = field(default_factory=lambda: [""])               
    y_title: List[str] = field(default_factory=lambda: [""])             
    x_label: AxisLabel = field(default_factory=AxisLabel) 
    y_label: AxisLabel = field(default_factory=AxisLabel)

@dataclass
class PlotInfo:
    bbox: np.ndarray = field(default_factory=lambda: np.empty((0,5)))

@dataclass
class ChartElementResult:
    """
    chart detector result
    {
        "legends":{
            "area": np.ndarray,
            "legendlabel":{
                "text": [str],
                "bbox": np.ndarray,
                "color": np.ndarray,
            },
        },
        "axis":{
            "x_title": (str),
            "y_title": (str),
            "x_label":{
                "value": np.ndarray,
                "bbox": np.ndarray,
            },
            "y_label":{
                "value": np.ndarray,
                "bbox": np.ndarray,
            },
        },
        "plot":{
            "bbox": np.ndarray,
        }    
    }
    """
    legends: LegendInfo = field(default_factory = LegendInfo)
    axis: AxisInfo = field(default_factory = AxisInfo)
    plot: PlotInfo = field(default_factory = PlotInfo)

    @classmethod
    def from_detectionresult(
        cls,
        detection_result: DetectionResult,
        legend_colors: np.ndarray,
        ocr_result: OcrResult,
    ):
        try:
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
                bbox=detection_result.get_bboxes("plot_area"),
            )

            return cls(
                legends=legends,
                axis=axis,
                plot=plot
            )
        except Exception as e:
            return cls(
                legends = LegendInfo(),
                axis = AxisInfo(),
                plot = PlotInfo(),
            )
    

    def save_json(self, pth):
        try:
            data = asdict(self)
            data = ndarray_to_list(data)
            with open(pth, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")

    @classmethod
    def from_dict(cls, data):
        legends = LegendInfo(
            area=np.array(data['legends']['area']),
            label=LegendLabel(
                text=data['legends']['label']['text'],
                bbox=np.array(data['legends']['label']['bbox']),
                color=np.array(data['legends']['label']['color'])
            )
        )   
        axis = AxisInfo(
            x_title=data['axis']['x_title'],
            y_title=data['axis']['y_title'],
            x_label=AxisLabel(
                value=data['axis']['x_label']['value'],
                bbox=np.array(data['axis']['x_label']['bbox']),
            ),
            y_label=AxisLabel(
                value=data['axis']['y_label']['value'],
                bbox=np.array(data['axis']['y_label']['bbox']),
            ),
        )
        plot = PlotInfo(
            bbox=np.array(data['plot']['bbox']),
        )
        return cls(
            legends=legends,
            axis=axis,
            plot=plot
        )
    
    @classmethod
    def from_json(cls, pth):
        try:
            with open(pth, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            return cls(
                legends = LegendInfo(),
                axis = AxisInfo(),
                plot = PlotInfo(),
            )





@dataclass
class LineResult:
    """
    line extractor result
    LineResult:{
        "legends":{
            "area": np.ndarray,
            "legendlabel":{
                "text": [str],
                "bbox": [np.ndarray], 
                "color": [np.ndarray],
                # "patch_bbox" [np.ndarray], 等待添加
        },
        x_pixel: [np.ndarray]
        y_pixel: [np.ndarray]
    }
    """

    legends: LegendInfo = field(default_factory=LegendInfo)
    x_pixel: np.ndarray = field(default_factory=lambda: np.empty((0,128)))
    y_pixel: np.ndarray = field(default_factory=lambda: np.empty((0,128)))

    @classmethod
    def from_extractresult(
        cls,
        legends: LegendInfo,
        x_pixel: np.ndarray,
        y_pixel: np.ndarray,
    ):
        try:
            return cls(
                legends = legends,
                x_pixel = x_pixel,
                y_pixel = y_pixel,
            )
        except:
            return cls()

    def save_json(self, pth):
        try:
            data = asdict(self)
            data = ndarray_to_list(data)
            with open(pth, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON: {e}")

    @classmethod
    def from_dict(cls, data):
        legends = LegendInfo(
            area=np.array(data['legends']['area']),
            label=LegendLabel(
                text=data['legends']['label']['text'],
                bbox=np.array(data['legends']['label']['bbox']),
                color=np.array(data['legends']['label']['color'])
            )
        )   
        
        x_pixel = np.array(data['x_pixel'])
        y_pixel = np.array(data['y_pixel'])
        return cls(
            legends=legends,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
        )
    
    @classmethod
    def from_json(cls, pth):
        try:
            with open(pth, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            return cls()
        


@dataclass
class PointData:
    sample: str = ""
    point_coordinates: np.ndarray = field(default_factory=lambda: np.empty((128, 2)))

    def coordinates_str(self) -> str:
        return "\n".join([f"{x:.5f} {y:.5f}" for x, y in self.point_coordinates])


@dataclass
class SubChartOcrResult:
    figure_index: str = ""
    x_label: str = ""
    y_label: str = ""
    note: str = ""
    data: List[PointData] = field(default_factory=list)


    @classmethod
    def from_chartocr(cls, figure_index, x_label, y_label, note, samples, points):
        data = []
        for i, v in enumerate(samples):
            arr = points[i, :, :]
            idx = np.argsort(arr[:, 0])
            arr_sortd = arr[idx]
            data.append(
                PointData(
                    sample=v,
                    point_coordinates=arr_sortd,
                )
            )
        return cls(
            figure_index = figure_index,
            x_label = x_label,
            y_label = y_label,
            note = note,
            data = data,
        )
            




@dataclass
class ChartOcrResult:
    figure_name: str=""
    sub_figure: List[SubChartOcrResult] = field(default_factory=list)


@dataclass
class ChartOcrResultList:
    results: List[ChartOcrResult] = field(default_factory=list)


    def save_excel(self, save_file):
        rows = []
        for chart in self.results:
            figure_name = chart.figure_name
            for sub in chart.sub_figure:
                figure_index = sub.figure_index
                x_label = sub.x_label
                y_label = sub.y_label
                note = sub.note
                for point in sub.data:
                    sample = point.sample
                    point_coordinates = point.coordinates_str()
                    rows.append({
                        "figure_name(id)": figure_name,
                        "figure_index": figure_index,
                        "figure_title": "",
                        "x_label": x_label,
                        "y_label": y_label,
                        "sample": sample,
                        "point_coordinates": point_coordinates,
                        "note": note,
                    })
        df = pd.DataFrame(rows)
        df.to_excel(save_file, index=False)   

