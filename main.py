from config import BaseConfig
from module.detector.chart_element_detector.detector import ChartDetector
from module.pixeltransform import PixelTransform
from utils import ChartElementResult
import numpy as np
import mmcv
from model import ChartOcr



if __name__ == "__main__":
    
    cfg = BaseConfig('config/', args=None)

    # detector = ChartDetector(cfg)
    # """
    # ['x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel', 'chart_title', 
    # 'x_tick', 'y_tick', 'legend_patch', 'legend_label', 'legend_title', 'legend_area', 'mark_label', 'value_label', 'y_axis_area', 'x_axis_area', 'tick_grouping']
    # """
    # result0 = detector.predict('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    # img = mmcv.imread('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    # result0 = [r.copy()  for r in result0]
    # result0[15] = np.empty((0, 5))
    # result = detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    # result1 = detector.dete_result.to_list()
    # result1[12] = np.empty((0, 5))
    # result1[13] = np.empty((0, 5))
    # result1[15] = np.empty((0, 5))
    # # detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg')
    # ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/42.png', result0, score_thr=0.3, out_file='result_pre.jpg')
    # ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/42.png', result1, score_thr=0.3, out_file='result_post.jpg')
    # result.save_json(pth="dete_result.json")
    # deteresult = ChartElementResult.from_json(pth="dete_result.json")
    # bboxes = deteresult.axis.y_label.bbox
    # values = deteresult.axis.y_label.value

    # pixel_transform = PixelTransform(cfg=cfg)

    # pixel_transform.fit(bboxes=bboxes, values=values, axis='y')

    imgs = 'data/input/'
    chartocr = ChartOcr(cfg=cfg)
    chartocr.ocr(img=imgs)

