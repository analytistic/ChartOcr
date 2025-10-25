from config import BaseConfig
from module.detector.chart_element_detector.detector import ChartDetector
import numpy as np



if __name__ == "__main__":
    


    cfg = BaseConfig('config/detector.toml', args=None)

    detector = ChartDetector(cfg)
    """
    
    ['x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel', 'chart_title', 
    'x_tick', 'y_tick', 'legend_patch', 'legend_label', 'legend_title', 'legend_area', 'mark_label', 'value_label', 'y_axis_area', 'x_axis_area', 'tick_grouping']
    
    """

    result0 = detector.predict('/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg')
    result0[9] = np.empty((0, 5))
    result0[12] = np.empty((0, 5))
    result_copy = [r.copy()  for r in result0]
    result1, _ = detector.postprocess(result_copy)
    result1[9] = np.empty((0, 5))
    result1[12] = np.empty((0, 5))
    # detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg')
    ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg', result0, score_thr=0.3, out_file='result_pre.jpg')
    ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg', result1, score_thr=0.3, out_file='result_post.jpg')