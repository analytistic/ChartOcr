import infer
import cv2
import line_utils
import os 
import sys
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict

CKPT = "module/extractor/lineformer_extector/iter_3000.pth" #
CONFIG = "module/extractor/lineformer_extector/lineformer_swin_t_config.py"
DEVICE = "cpu"

class LineFormerExtractor:
    def __init__(self, img, detection_result):
        self.img=img
        self.detection_result = detection_result
        self.model=infer.load_model(CONFIG, CKPT, DEVICE)
        self.line_extractor_result = extractor(self.img,self.detection_result)
    # 获取背景色
    def get_background_color(img):
        width, height = img.size
        total_pixels = width * height  # 总像素数，用于计算占比
        
        # 统计每个颜色的出现次数
        color_count = defaultdict(int)
        for x in range(width):
            for y in range(height):
                # 获取像素的RGB值（返回元组：(r, g, b)）
                r, g, b = img_rgb.getpixel((x, y))
                color_count[(r, g, b)] += 1
        
        # 找出出现次数最多的颜色
        most_common_color, count = max(color_count.items(), key=lambda item: item[1])
        # 计算占比
        proportion = (count / total_pixels) * 100
        
        return most_common_color, proportion

    # 割出图像的绘图区
    def segment(img,detection_result):
        # img = img.convert('RGB')
        # img_array = np.array(img)
        height, width, _ = img.shape

        # 1. 获取背景色
        background_color,propotion=get_background_color(img)
        bg_r, bg_g, bg_b = background_color

        # 2. 先将整个图片填充为背景色
        masked_array = np.full_like(img, [bg_r, bg_g, bg_b])

        # 3. 提取plot.bbox区域，保留绘图区域
        plot_bbox = detection_result.plot.bbox
        x1, y1, x2, y2,_ = plot_bbox  # bbox格式：[x1, y1, x2, y2,score]（左上角、右下角）
        # 确保坐标在有效范围内（防止越界）
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(x1, min(x2, width-1))
        y2 = max(y1, min(y2, height-1))
        # 将原图的plot区域复制到结果中
        masked_array[y1:y2+1, x1:x2+1] = img[y1:y2+1, x1:x2+1]
        
        # 4. 覆盖图例区域,得到一张只有绘图区域的图
        if hasattr(detection_result.legends, 'area') and detection_result.legends.area.size > 0:
            legend_bbox = detection_result.legends.area
            lx1, ly1, lx2, ly2,_ = legend_bbox
            lx1 = max(0, min(lx1, width-1))
            ly1 = max(0, min(ly1, height-1))
            lx2 = max(lx1, min(lx2, width-1))
            ly2 = max(ly1, min(ly2, height-1))
            # 用背景色覆盖图例区域
            masked_array[ly1:ly2+1, lx1:lx2+1] = [bg_r, bg_g, bg_b]

        return masked_array

    

    
def extractor(img,detection_result):
    masked_img=segment(img,detection_result)

    layer_imgs=filter(masked_img,detection_result)
    
    line_extractor_result = {}
    # 遍历每个层次图（每个层次图对应一条线）
    for line_num, (layer_img, color) in enumerate(zip(layer_imgs, detectionn_result['legends']['label']['color']), start=1):
        # 1. 对层次图进行推理，获取曲线坐标
        # 假设infer.get_dataseries返回格式为：包含(x,y)坐标的列表或np.ndarray
        # 例如：[(x1,y1), (x2,y2), ...] 或 np.array([[x1,y1], [x2,y2], ...])
        line_dataseries = infer.get_dataseries(layer_img, to_clean=False)
        
        # 2. 确保坐标格式为np.ndarray（形状：[n, 2]，每行是(x,y)）
        if isinstance(line_dataseries, list):
            coordinates = np.array(line_dataseries, dtype=np.float32)
        elif isinstance(line_dataseries, np.ndarray):
            # 确保维度正确（n行2列）
            if line_dataseries.ndim == 1:
                coordinates = line_dataseries.reshape(-1, 2)
            else:
                coordinates = line_dataseries
        else:
            raise TypeError(f"不支持的坐标格式：{type(line_dataseries)}，需为list或np.ndarray")
        
        # 3. 组装当前线的结果（颜色转换为np.ndarray格式）
        line_result = {
            "color": np.array(color, dtype=np.uint8),  # RGB颜色（0-255整数）
            "coordinate": coordinates  # 坐标数组（n,2）
        }
        
        # 4. 存入结果字典（键为线编号，从1开始）
        line_extractor_result[str(line_num)] = line_result  # 用字符串作为键，如"1"、"2"

    return line_extractor_result






        
    



#每次运行前需要
#source .venv/bin/activate
#unset LD_LIBRARY_PATH PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV

# for i in range(1,11):
#     #读入图片
#     img_path = f"data/input/{i}.jpg"  
#     if not os.path.exists(img_path):
#         img_path = f"data/input/{i}.png"
#     if not os.path.exists(img_path):
#         img_path = f"data/input/{i}.tif"
#     img = cv2.imread(img_path) # BGR format# for i in range(1,11):
#     #读入图片
#     img_path = f"data/input/{i}.jpg"  
#     if not os.path.exists(img_path):
#         img_path = f"data/input/{i}.png"
#     if not os.path.exists(img_path):
#         img_path = f"data/input/{i}.tif"
#     img = cv2.imread(img_path) # BGR format

#line_dataseries = infer.get_dataseries(img, to_clean=False)

# Visualize extracted line keypoints
#img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))

#输出提取的像素点
# 指定输出文件路径
# output_file = "data/test/outputTest.txt"
# # 重定向print函数的输出
# with open(output_file, 'w') as f:
#     sys.stdout = f
# # 下面的print内容将被写入"output.txt"
#     print(line_dataseries)
# # 恢复标准输出
# sys.stdout = sys.__stdout__

# # 绘出提取的像素点折线
# height, width, _ = img.shape
# for line in line_dataseries:  # 循环处理每条线
#     x_list = [point['x'] for point in line]  # 提取该线所有x坐标
#     #y_list = [point['y'] for point in line]  # 提取该线所有y坐标
#     y_list = [height - point['y'] for point in line]
#     plt.plot(x_list,y_list,)  # 画折线
# plt.savefig("data/test/outputTest_plot.png")
    
# cv2.imwrite(f'data/output/{i}.jpg', img)