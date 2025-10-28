from module import PixelTransform, ChartDetector, LineExtractor
from utils import ChartElementResult, LineResult, ChartOcrResult, ChartOcrResultList, PointData, SubChartOcrResult
import mmcv
from glob import glob
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

class ChartOcr:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = ChartDetector(cfg)
        self.x_pixeltransform = PixelTransform(cfg)
        self.y_pixeltransform = PixelTransform(cfg)
        self.extractor = LineExtractor(cfg)

    def ocr(self, img):
        figure_name = []
        if isinstance(img, str):
            if os.path.isfile(img):
                figure_name.append(os.path.basename(img))
                imgs = [img]
            else:
                imgs = []
                for file in Path(img).iterdir():
                    if file.is_file():
                        imgs.append(file)
                        figure_name.append(file.name)
        else:
            imgs = [img]
            figure_name.append('')

        results = []
        for i, img in tqdm(enumerate(imgs), total = len(imgs)):
            if img is not np.ndarray:
                try:
                    img = mmcv.imread(img)
                except:
                    results.append('')
                    continue
            detector_result = self.detector.getjson(img)
     
            x_bboxes = detector_result.axis.x_label.bbox
            y_bboxes = detector_result.axis.y_label.bbox
            x_values = detector_result.axis.x_label.value
            y_values = detector_result.axis.y_label.value
            re_errorx = self.x_pixeltransform.fit(bboxes=x_bboxes, values=x_values, axis='x')
            re_errory = self.y_pixeltransform.fit(bboxes=y_bboxes, values=y_values, axis='y')
            if re_errorx[2] > 0.1*re_errorx[-1] or re_errory[2] > 0.1*re_errory[-1]:
                print(f'error: {figure_name}')
                continue

            extractor_result = self.extractor.getjson(img = img, detector_result=detector_result)

            x_data = self.x_pixeltransform.pridect(extractor_result.x_pixel)
            y_data = self.y_pixeltransform.pridect(extractor_result.y_pixel)
            points = np.stack([x_data, y_data], axis=-1)

            subfigure_result = SubChartOcrResult.from_chartocr(
                figure_index='',
                x_label=detector_result.axis.x_title,
                y_label=detector_result.axis.y_title,
                note='',
                samples=detector_result.legends.label.text,
                points=points,
            )
            results.append(ChartOcrResult(figure_name=figure_name[i], sub_figure=[subfigure_result]))

        return ChartOcrResultList(results)
    



        
                         
                         



