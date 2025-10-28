import infer
import cv2
import line_utils

CKPT = "module/extractor/lineformer_extector/iter_3000.pth" #
CONFIG = "module/extractor/lineformer_extector/lineformer_swin_t_config.py"
DEVICE = "cpu"


i = 30

img_path = f"data/input/{i}.tif"
img = cv2.imread(img_path) # BGR format

infer.load_model(CONFIG, CKPT, DEVICE)
line_dataseries = infer.get_dataseries(img, to_clean=False)

# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))
    
cv2.imwrite(f'data/output/{i}.jpg', img)

#source .venv/bin/activate