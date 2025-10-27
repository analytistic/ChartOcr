from module import PixelTransform, ChartDetector
from utils import ChartElementResult
import mmcv
from glob import glob
import os
from pathlib import Path
import numpy as np

class ChartOcr:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = ChartDetector(cfg)
        self.x_pixeltransform = PixelTransform(cfg)
        self.y_pixeltransform = PixelTransform(cfg)

    def ocr(self, img):
        if isinstance(img, str):
            if os.path.isfile(img):
                imgs = [img]
            else:
                imgs = []
                for file in Path(img).iterdir():
                    if file.is_file():
                        imgs.append(file)
        else:
            imgs = [img]

        results = []
        for img in imgs:
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
                print('error')
                continue

        return None
    
    def get_json(self, img, save_file):
        results = self.ocr(img)
        return None



        
                         
                         



