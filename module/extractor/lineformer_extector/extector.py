import infer
import cv2
import line_utils

img_path = "/Users/alex/project/chartocr/ChartOcr/data/input/5.jpg"
img = cv2.imread(img_path) # BGR format

CKPT = "module/extractor/lineformer_extector/iter_3000.pth"
CONFIG = "module/extractor/lineformer_extector/lineformer_swin_t_config.py"
DEVICE = "cpu"

infer.load_model(CONFIG, CKPT, DEVICE)
line_dataseries = infer.get_dataseries(img, to_clean=False)

# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))
    
cv2.imwrite('module/extractor/lineformer_extector/sample_result.png', img)

