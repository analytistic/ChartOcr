from paddleocr import PaddleOCR
from PIL import Image
import pytesseract
import cv2

class BboxOcr:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.ocr = PaddleOCR(
        #     use_angle_cls=True,
        #     lang=self.cfg.detector.ocr.lang,
        #     use_gpu=self.cfg.detector.device=='cuda',
        #     show_log=False,
        #     det=False,
        # )


    def ocr(self, img, bboxes, class_name):
        if len(bboxes) == 0:
            return []
        
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2, _ = bbox
            if class_name == 'x_label':
                x_bar = abs(x2-x1)
                x1 = x1 - x_bar*1
                x2 = x2 + x_bar*0.05
            if class_name == 'y_title':
                y_bar = abs(y2-y1)
                y1 = y1 - y_bar * 0.05
                y2 = y2 + y_bar * 0.05
            h, w = img.shape[:2]
            x1 = max(0, int(x1))
            x2 = min(w, int(x2))
            y1 = max(0, int(y1))
            y2 = min(h, int(y2))
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            crops.append(crop)

            results = []
            for crop in crops:
                if class_name == 'y_title':
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

                pil_img = Image.fromarray(crop)
                try:
                    text = pytesseract.image_to_string(pil_img)
                except Exception as e:
                    text = ""
                results.append(text.strip())
            
            return results

    
