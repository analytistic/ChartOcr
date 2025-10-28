# from config import BaseConfig
# from module import ChartDetector, LineExtractor, PixelTransform
# from utils import ChartElementResult
# import numpy as np
# import mmcv
# from model import ChartOcr
# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Run OCR inference on a sample image 
result = ocr.predict(
    input="截屏2025-10-28 22.27.29.png")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")



# if __name__ == "__main__":
    
    # cfg = BaseConfig('config/', args=None)

    # detector = ChartDetector(cfg)
    # extractor = LineExtractor(cfg)
    # result0 = detector.predict('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    # img = mmcv.imread('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    # img = mmcv.bgr2rgb(img)
    # result0 = [r.copy()  for r in result0]
    # # result0[15] = np.empty((0, 5))
    # result = detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/42.png')
    # result1 = detector.dete_result.to_list()
    # # result1[12] = np.empty((0, 5))
    # # result1[13] = np.empty((0, 5))
    # # result1[15] = np.empty((0, 5))
    # # # detector.getjson('/Users/alex/project/chartocr/ChartOcr/data/input/3.jpg')
    # ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/42.png', result0, score_thr=0.3, out_file='result_pre.jpg')
    # ChartDetector.plot(detector.model, '/Users/alex/project/chartocr/ChartOcr/data/input/42.png', result1, score_thr=0.3, out_file='result_post.jpg')
    
    # lineresult = extractor.getjson(img, detector_result=result)
    # LineExtractor.plot('/Users/alex/project/chartocr/ChartOcr/data/input/42.png', lineresult, out_file='extract.jpg')

    # result.save_json(pth="dete_result.json")
    # deteresult = ChartElementResult.from_json(pth="dete_result.json")mi
    # bboxes = deteresult.axis.y_label.bbox
    # values = deteresult.axis.y_label.value

    # pixel_transform = PixelTransform(cfg=cfg)

    # pixel_transform.fit(bboxes=bboxes, values=values, axis='y')

    # imgs = 'data/input/'
    # chartocr = ChartOcr(cfg=cfg)
    # result_list = chartocr.ocr(img=imgs, visual_out='data/output/')
    # result_list.save_excel(save_file="extractor.xlsx")
    # imgs = 'data/input/30.tif'
    # imgs = mmcv.imread(imgs)
    # chartocr = ChartOcr(cfg=cfg)
    # detector_result = chartocr.detector.predict(imgs)
    # detector_result_raw = detector_result.copy()
    # detector_result = chartocr.detector.getjson(imgs)
    # detector_result2plot = chartocr.detector.dete_result.to_list()
    # ChartDetector.plot(chartocr.detector.model, imgs, detector_result_raw, score_thr=0.3, out_file='detector_result_raw.jpg')
    # ChartDetector.plot(chartocr.detector.model, imgs, detector_result2plot, score_thr=0.3, out_file='detector_result.jpg')
    # extractor_result = chartocr.extractor.getjson(imgs, detector_result=detector_result)
    # LineExtractor.plot(imgs, result=extractor_result, out_file='extractor_result.jpg')
    
    

    

