from config import BaseConfig
from module.detector.chart_element_detector.detector import ChartDetector
import numpy as np
import mmcv



if __name__ == "__main__":
    
    cfg = BaseConfig('config/detector.toml', args=None)

    detector = ChartDetector(cfg)
    """
    ['x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel', 'chart_title', 
    'x_tick', 'y_tick', 'legend_patch', 'legend_label', 'legend_title', 'legend_area', 'mark_label', 'value_label', 'y_axis_area', 'x_axis_area', 'tick_grouping']
    """
    result0 = detector.predict('/Users/alex/project/chartocr/ChartOcr/data/input/30.tif')

    img = mmcv.imread('/Users/alex/project/chartocr/ChartOcr/data/input/30.tif')
    result_copy = [r.copy()  for r in result0]
    result0[15] = np.empty((0, 5))
    detector.dete_result.bboxes_list = result_copy
    _ = detector.postprocess(img)
    result1 = detector.dete_result.to_list()

    result1[12] = np.empty((0, 5))
    result1[13] = np.empty((0, 5))
    result1[15] = np.empty((0, 5))
    # detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg')
    ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/30.tif', result0, score_thr=0.3, out_file='result_pre.jpg')
    ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/30.tif', result1, score_thr=0.3, out_file='result_post.jpg')