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
    def __init__(self, cfg, img, detection_result):
        self.cfg = cfg
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

    def color_distance(c1, c2):
        """计算两个RGB颜色的欧氏距离（判断颜色是否相近）"""
        # c1和c2为RGB元组或数组，如(255,0,0)或[255,0,0]
        return math.sqrt(sum((x - y) **2 for x, y in zip(c1, c2)))

    def split_by_legend_colors(img, detection_result, threshold=30):
        """
        根据图例颜色分割图片为层次图
        img: 处理好的图片（仅保留绘图区域的PIL.Image对象）
        detection_result: 包含图例颜色的检测结果
        threshold: 颜色相似度阈值（越小越严格，建议20-50）
        return: 层次图列表（每个元素为PIL.Image，对应一个图例颜色）
        """
        # 1. 提取图例颜色（转换为RGB整数格式）
        legend_colors = detection_result.legends.label.color
        # 假设color是numpy数组，形状为(n, 3)，每个元素是[R, G, B]（0-255）
        legend_colors = [tuple(map(int, color)) for color in legend_colors]
        n_legends = len(legend_colors)
        if n_legends == 0:
            raise ValueError("未从detection_result中找到图例颜色")
        
        # 2. 为每个图例颜色生成层次图
        height, width, _ = img.shape
        bg_color = get_background_color(img)  # 复用之前的背景色获取函数
        layer_imgs = []
        for target_color in legend_colors:
            # 创建背景色数组作为基础
            layer_array = np.full_like(img_array, bg_color)
            # 遍历每个像素，保留与目标颜色相近的像素
            for y in range(height):
                for x in range(width):
                    pixel_color = tuple(img_array[y, x])
                    # 跳过背景色像素（避免误判）
                    if color_distance(pixel_color, bg_color) < threshold:
                        continue
                    # 若像素颜色与目标颜色接近，则保留
                    if color_distance(pixel_color, target_color) < threshold:
                        layer_array[y, x] = pixel_color
            # 转换为PIL图片并添加到列表
            layer_imgs.append(Image.fromarray(layer_array))
        
        return layer_imgs

    def extractor(img,detection_result):
        masked_img=segment(img,detection_result)

        layer_imgs=split_by_legend_colors(masked_img,detection_result)
        
        line_extractor_result = {}
        # 遍历每个层次图（每个层次图对应一条线）
        extractor_result= {}
        for layer_img in layer_imgs:
            # 1. 对层次图进行推理，获取曲线坐标
            line_dataseries = infer.get_dataseries(layer_img, to_clean=False)
            full_points = dataseries[0]  # 提取第一条线的所有点
            extractor_result.append(line_dataseries)
        
            # 2. 将字典列表转换为x、y数组
            full_x = np.array([p['x'] for p in full_points], dtype=np.int32)
            full_y = np.array([p['y'] for p in full_points], dtype=np.int32)
            total_points = len(full_x)
        
            # 3. 均匀选取128个点
            if total_points <= 128:
                # 点数不足时，补全至128（边缘填充）
                selected_x = np.pad(full_x, (0, 128 - total_points), mode='edge')
                selected_y = np.pad(full_y, (0, 128 - total_points), mode='edge')
            else:
                # 点数超过时，按步长采样
                sample_step = total_points // 128
                selected_indices = np.arange(0, total_points, sample_step)[:128]
                selected_x = full_x[selected_indices]
                selected_y = full_y[selected_indices]
            
            # 4. 组织为LineResult格式
            line_result = {
                "legends": detector_result.legends,  # 复用detector_result的legends
                "x_pixel": [selected_x],  # 列表中存储当前线的x坐标数组
                "y_pixel": [selected_y]   # 列表中存储当前线的y坐标数组
            }
            
            line_results.append(line_result)

        # 画图
        img = line_utils.draw_lines(img, line_utils.points_to_array(extractor_result))
            
        cv2.imwrite(f'data/test/{i}.jpg', img)
    
        return line_results
            






        
    



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