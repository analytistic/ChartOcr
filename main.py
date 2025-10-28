from config import BaseConfig
from module import ChartDetector, LineExtractor, PixelTransform
from utils import ChartElementResult
import numpy as np
import mmcv
from model import ChartOcr



if __name__ == "__main__":
    
    cfg = BaseConfig('config/', args=None)

    detector = ChartDetector(cfg)
    extractor = LineExtractor(cfg)
    result0 = detector.predict('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    img = mmcv.imread('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    img = mmcv.bgr2rgb(img)
    result0 = [r.copy()  for r in result0]
    # result0[15] = np.empty((0, 5))
    result = detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    result1 = detector.dete_result.to_list()
    # result1[12] = np.empty((0, 5))
    # result1[13] = np.empty((0, 5))
    # result1[15] = np.empty((0, 5))
    # # detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg')
    ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/42.png', result0, score_thr=0.3, out_file='result_pre.jpg')
    ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/42.png', result1, score_thr=0.3, out_file='result_post.jpg')
    
    lineresult = extractor.getjson(img, detector_result=result)
    LineExtractor.plot('/Users/alex/project/chartocr/ChartOcr/data/input/42.png', lineresult, out_file='extract.jpg')




    # result.save_json(pth="dete_result.json")
    # deteresult = ChartElementResult.from_json(pth="dete_result.json")
    # bboxes = deteresult.axis.y_label.bbox
    # values = deteresult.axis.y_label.value

    # pixel_transform = PixelTransform(cfg=cfg)

    # pixel_transform.fit(bboxes=bboxes, values=values, axis='y')

    # imgs = 'data/input/'
    # chartocr = ChartOcr(cfg=cfg)
    # chartocr.ocr(img=imgs)

