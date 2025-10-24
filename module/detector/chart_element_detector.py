from mmdet.apis import init_detector, inference_detector
import mmcv
import pickle
import json
import numpy as np
import glob
import os
from pathlib import Path
import cv2
from paddleocr import PaddleOCR
import re

# Specify the path to model config and checkpoint file
config_file = 'mmdetection/configs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = 'module/detector/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 初始化 PaddleOCR (针对 PaddleOCR 2.7.3 优化配置)
# 图表文本识别优化参数
ocr_engine = PaddleOCR(
    use_angle_cls=True,  # 启用文本方向检测
    lang='ch',  # 中英文混合识别模型
    use_gpu=True,  # 使用GPU加速
    show_log=False,  # 关闭日志输出
    det_db_thresh=0.3,  # 降低检测阈值，检测更多小文本
    det_db_box_thresh=0.3,  # 文本框过滤阈值
    det_db_unclip_ratio=2.2,  # 增大文本框扩展比例，获取更完整文本
    rec_batch_num=8,  # 增大批处理提升速度
    drop_score=0.3,  # 降低识别结果过滤阈值，保留更多结果
    use_space_char=True,  # 识别空格字符
    det_limit_side_len=960,  # 检测模型输入图像边长限制
    det_limit_type='max',  # 限制类型
    max_text_length=50,  # 最大文本长度
)


def preprocess_roi_for_ocr(roi, method='enhanced'):
    """
    Args:
        roi: 输入图像ROI区域
        method: 预处理方法 'enhanced'(增强), 'binary'(二值化), 'denoise'(降噪)
    """
    # 确保输入是有效的图像
    if roi is None or roi.size == 0:
        return roi

    # 转换为灰度图
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    if method == 'enhanced':
        # 方法1: 增强对比度和锐化
        # 1. 双边滤波去噪（保留边缘）
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. 自适应直方图均衡化 (CLAHE) - 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. 锐化处理 - 增强文本边缘
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

        # 转回BGR格式
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    elif method == 'binary':
        # 方法2: 自适应二值化（适合低对比度文本）
        # 1. 高斯去噪
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. 自适应二值化
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 3
        )

        # 3. 去除小噪点
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 转回BGR格式
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    elif method == 'denoise':
        # 方法3: 降噪优先（适合有噪声的图像）
        # 1. 非局部均值去噪
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 2. CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 转回BGR格式
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        # 默认返回原图的BGR格式
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return result


def resize_roi_for_better_recognition(roi, target_height=48):
    """
    调整ROI尺寸以提高识别率
    将小尺寸文本放大到更适合识别的尺寸
    """
    h, w = roi.shape[:2]

    # 如果高度太小，放大图像
    if h < target_height and h > 0:
        scale = target_height / h
        new_w = int(w * scale)
        new_h = target_height
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return resized, scale

    return roi, 1.0


def clean_ocr_text(text):
    """
    清理OCR识别结果中的噪声字符
    """
    if not text:
        return ""
    text = re.sub(r'[|_~`]', '', text)  # 移除特殊符号和干扰字符
    text = re.sub(r'\s+', ' ', text)  # 移除多余的空格
    text = text.strip()  # 去除首尾空格
    return text


def extract_text_with_paddleocr(image, bbox, rotate=False, rotation_angle=90):
    """
    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        rotate: 是否旋转图像（用于竖直文字如y_title），默认False
        rotation_angle: 旋转角度，90=顺时针，-90=逆时针，默认90
    """
    x1, y1, x2, y2 = map(int, bbox)

    # 确保坐标在图像范围内
    height, width = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # 提取ROI区域
    roi = image[y1:y2, x1:x2]

    # 检查ROI是否有效
    if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return ""

    # 计算自适应padding（根据ROI大小）
    roi_h, roi_w = roi.shape[:2]
    # 如果需要旋转，使用更大的padding避免旋转后丢失信息
    if rotate:
        padding = max(10, int(min(roi_h, roi_w) * 0.2))
    else:
        padding = max(8, int(min(roi_h, roi_w) * 0.15))

    y1_pad = max(0, y1 - padding)
    y2_pad = min(height, y2 + padding)
    x1_pad = max(0, x1 - padding)
    x2_pad = min(width, x2 + padding)
    roi_padded = image[y1_pad:y2_pad, x1_pad:x2_pad]

    # 如果需要旋转（用于竖直文字识别）
    if rotate:
        if rotation_angle == 90:
            # 顺时针旋转90度（从下到上的竖直文字 → 水平）
            roi_padded = cv2.rotate(roi_padded, cv2.ROTATE_90_CLOCKWISE)

    def parse_ocr_result(ocr_result):
        """
        解析OCR结果并返回文本和平均置信度
        PaddleOCR 2.7.3 返回格式: [[[bbox], (text, confidence)], ...]
        """
        texts = []
        confidences = []

        if ocr_result is not None and len(ocr_result) > 0:
            for line_result in ocr_result:
                if line_result is not None and isinstance(line_result, list):
                    for detection in line_result:
                        if detection is not None and len(detection) >= 2:
                            # detection[0] 是bbox坐标列表
                            # detection[1] 是(text, confidence)元组
                            text_info = detection[1]
                            if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                                text = str(text_info[0]).strip()
                                confidence = float(text_info[1])

                                # 清理文本
                                text = clean_ocr_text(text)

                                # 降低阈值以获取更多文本
                                if confidence > 0.25 and len(text) > 0:
                                    texts.append(text)
                                    confidences.append(confidence)

        combined_text = ' '.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return combined_text, avg_confidence

    # 存储所有尝试的结果
    results = []

    try:
        # 策略1: 原图识别
        ocr_result = ocr_engine.ocr(roi_padded, cls=True)
        text, conf = parse_ocr_result(ocr_result)
        if text and conf > 0:
            results.append((text, conf, 'original'))

        # 策略2: 如果文本高度较小，放大后识别
        if roi_h < 30:
            roi_resized, scale = resize_roi_for_better_recognition(roi_padded, target_height=48)
            ocr_result = ocr_engine.ocr(roi_resized, cls=True)
            text, conf = parse_ocr_result(ocr_result)
            if text and conf > 0:
                # 放大图像的识别结果给予轻微加成
                results.append((text, conf * 1.05, 'resized'))

        # 策略3: 如果之前结果不佳，使用增强预处理
        if not results or max(r[1] for r in results) < 0.65:
            roi_enhanced = preprocess_roi_for_ocr(roi_padded, method='enhanced')
            ocr_result = ocr_engine.ocr(roi_enhanced, cls=True)
            text, conf = parse_ocr_result(ocr_result)
            if text and conf > 0:
                results.append((text, conf, 'enhanced'))

        # 策略4: 如果仍然不佳，尝试二值化
        if not results or max(r[1] for r in results) < 0.5:
            roi_binary = preprocess_roi_for_ocr(roi_padded, method='binary')
            ocr_result = ocr_engine.ocr(roi_binary, cls=True)
            text, conf = parse_ocr_result(ocr_result)
            if text and conf > 0:
                results.append((text, conf, 'binary'))

        # 选择最佳结果
        if results:
            # 按置信度排序
            results.sort(key=lambda x: x[1], reverse=True)
            best_text, best_conf, best_method = results[0]

            # 如果有多个相似结果，选择最常见的
            if len(results) > 1:
                text_counts = {}
                for text, conf, method in results:
                    text_lower = text.lower()
                    if text_lower not in text_counts:
                        text_counts[text_lower] = []
                    text_counts[text_lower].append((text, conf, method))

                # 如果有文本出现多次，优先选择它
                for text_lower, occurrences in text_counts.items():
                    if len(occurrences) >= 2:
                        # 选择置信度最高的版本
                        occurrences.sort(key=lambda x: x[1], reverse=True)
                        return occurrences[0][0]

            return best_text

        return ""

    except Exception as e:
        print(f"OCR识别错误 (bbox: {bbox}): {str(e)}")

        # 最后尝试：使用降噪预处理
        try:
            roi_denoised = preprocess_roi_for_ocr(roi_padded, method='denoise')
            ocr_result = ocr_engine.ocr(roi_denoised, cls=True)
            text, _ = parse_ocr_result(ocr_result)
            return text if text else ""
        except Exception as e2:
            print(f"OCR失败: {str(e2)}")
            return ""


def get_dominant_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)

    # 确保坐标在图像范围内
    height, width = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # 策略1: 取整个bbox区域（不只是中心，避免遗漏重要颜色）
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return (128, 128, 128)  # 默认灰色

    # 将图像转换为RGB（OpenCV默认是BGR）
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # 重塑为像素列表
    pixels = roi_rgb.reshape(-1, 3)

    # 过滤掉背景色（只过滤纯白色，保留灰色、黑色等所有其他颜色）
    def is_background_color(pixel, white_threshold=240):
        """判断是否为背景色（仅判断白色）"""
        # 白色或接近白色
        if np.all(pixel > white_threshold):
            return True
        return False

    # 过滤背景色
    foreground_pixels = np.array([p for p in pixels if not is_background_color(p)])

    # 如果过滤后没有像素，使用原始像素
    if len(foreground_pixels) == 0:
        foreground_pixels = pixels

    # 策略2: 使用K-means聚类找到主要颜色（改进版：对灰色和黑色更友好）
    try:
        pixels_float = foreground_pixels.astype(np.float32)

        # 确定聚类数量（根据像素数量自适应调整）
        pixel_count = len(foreground_pixels)
        if pixel_count < 50:
            k = 1
        elif pixel_count < 200:
            k = 2
        else:
            k = min(3, max(1, pixel_count // 100))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels_float, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # 统计每个簇的像素数量
        label_counts = np.bincount(labels.flatten())

        # 选择最大簇的中心作为主导色
        dominant_cluster = np.argmax(label_counts)
        dominant_color = centers[dominant_cluster].astype(int)

    except Exception as e:
        # 如果K-means失败，使用中位数作为备选
        print(f"K-means失败，使用中位数: {e}")
        dominant_color = np.median(foreground_pixels, axis=0).astype(int)

    # 策略3: 使用中位数进行微调（改进版：更适合均匀颜色如灰色、黑色）
    # 找到与dominant_color接近的像素
    distances = np.sqrt(np.sum((foreground_pixels - dominant_color) ** 2, axis=1))

    # 使用更宽松的阈值，特别是对于灰色和黑色这种变化小的颜色
    # 如果颜色标准差很小（如灰色、黑色），使用更大百分比
    color_variance = np.std(foreground_pixels, axis=0).mean()
    if color_variance < 20:  # 低方差，说明是均匀颜色
        threshold_percentile = 70
    else:
        threshold_percentile = 50

    close_pixels = foreground_pixels[distances < np.percentile(distances, threshold_percentile)]

    if len(close_pixels) > 10:
        # 使用这些接近的像素的中位数作为最终颜色
        final_color = np.median(close_pixels, axis=0).astype(int)
    else:
        final_color = dominant_color

    # 确保颜色值在有效范围内，并转换为Python原生int类型（支持JSON序列化）
    r = np.clip(final_color[0], 0, 255)
    g = np.clip(final_color[1], 0, 255)
    b = np.clip(final_color[2], 0, 255)

    # 使用 Python int() 强制转换，确保不是 numpy 类型
    rgb_color = (
        int(r.item()) if hasattr(r, 'item') else int(r),
        int(g.item()) if hasattr(g, 'item') else int(g),
        int(b.item()) if hasattr(b, 'item') else int(b)
    )

    return rgb_color


def ensure_json_serializable(obj):
    """
    确保对象可以被 JSON 序列化
    将 numpy 类型转换为 Python 原生类型
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def calculate_bbox_center(bbox):
    """计算边界框的中心点"""
    x1, y1, x2, y2 = bbox[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def check_vertical_alignment(bbox1, bbox2, tolerance=0.3):
    """
    检查两个bbox是否在垂直方向上对齐（y坐标有重叠）
    tolerance:表示允许的偏移比例
    """
    y1_min, y1_max = bbox1[1], bbox1[3]
    y2_min, y2_max = bbox2[1], bbox2[3]

    # 计算垂直重叠区域
    overlap = min(y1_max, y2_max) - max(y1_min, y2_min)
    min_height = min(y1_max - y1_min, y2_max - y2_min)

    # 如果有重叠，或者偏移在容忍范围内，认为是对齐的
    return overlap > -tolerance * min_height


def infer_patch_color_from_label(image, label_bbox, search_direction='left', search_distance=200):
    """
    当未检测到legend_patch时，基于legend_label位置精准推断并提取颜色

    Args:
        image: 输入图像
        label_bbox: legend_label的边界框 [x1, y1, x2, y2]
        search_direction: 搜索方向 'left'(左侧) 或 'right'(右侧) 或 'both'(两侧都尝试)
        search_distance: 最大搜索距离（像素）

    Returns:
        color: RGB颜色元组，如果未找到返回None
    """
    x1, y1, x2, y2 = map(int, label_bbox)
    height, width = image.shape[:2]

    # 如果是both方向，先尝试left，再尝试right
    if search_direction == 'both':
        color = infer_patch_color_from_label(image, label_bbox, 'left', search_distance)
        if color is not None:
            return color
        return infer_patch_color_from_label(image, label_bbox, 'right', search_distance)

    # 计算label的中心y坐标和高度
    label_center_y = (y1 + y2) // 2
    label_height = y2 - y1

    # 定义搜索区域（在label的左侧或右侧）
    if search_direction == 'left':
        search_x_start = max(0, x1 - search_distance)
        search_x_end = x1 - 2  # 留一点间隙避免与label重叠
    else:
        search_x_start = x2 + 2
        search_x_end = min(width, x2 + search_distance)

    # 垂直方向对齐label的位置，适当扩展以容纳不完美对齐的情况
    vertical_expansion = max(5, int(label_height * 0.3))
    search_y_start = max(0, y1 - vertical_expansion)
    search_y_end = min(height, y2 + vertical_expansion)

    # 提取搜索区域
    search_roi = image[search_y_start:search_y_end, search_x_start:search_x_end]

    # 转换到RGB空间
    search_roi_rgb = cv2.cvtColor(search_roi, cv2.COLOR_BGR2RGB)
    roi_height, roi_width = search_roi_rgb.shape[:2]

    # 逐列扫描，寻找颜色区域的起始和结束位置
    def analyze_column(col_idx):
        """分析某一列的颜色密度和颜色值（改进版：更好地处理灰色和黑色）"""
        if col_idx < 0 or col_idx >= roi_width:
            return 0, None, 0

        column = search_roi_rgb[:, col_idx, :]

        # 过滤白色和接近白色的像素（降低阈值以更好地检测浅灰色边界）
        is_colored = ~np.all(column > 235, axis=1)
        colored_count = np.sum(is_colored)

        if colored_count > 0:
            colored_pixels = column[is_colored]
            # 计算平均颜色
            avg_color = np.mean(colored_pixels, axis=0)
            # 计算颜色的饱和度（用于区分真正的颜色和噪声）
            color_std = np.std(colored_pixels, axis=0).mean()
            return colored_count, avg_color, color_std

        return 0, None, 0

    # 从靠近label的一侧开始扫描
    if search_direction == 'left':
        scan_range = range(roi_width - 1, -1, -1)
    else:
        scan_range = range(0, roi_width)

    # 寻找patch的边界（改进版：更灵活的检测策略）
    patch_start = None
    patch_end = None
    max_density = 0
    accumulated_colors = []
    color_stds = []

    # 动态阈值：基于ROI高度自适应调整
    start_threshold = max(3, int(roi_height * 0.15))  # 至少3个像素或15%高度
    continue_threshold = max(2, int(roi_height * 0.10))  # 至少2个像素或10%高度

    for idx, col in enumerate(scan_range):
        density, col_color, color_std = analyze_column(col)

        # 找到颜色区域的开始
        if patch_start is None and density >= start_threshold:
            patch_start = col
            if col_color is not None:
                accumulated_colors.append(col_color)
                color_stds.append(color_std)

        # 在patch内部，继续累积颜色
        elif patch_start is not None and density >= continue_threshold:
            if col_color is not None:
                accumulated_colors.append(col_color)
                color_stds.append(color_std)
            patch_end = col
            max_density = max(max_density, density)

        # 颜色区域结束（允许有小的间隙）
        elif patch_start is not None:
            # 检查接下来的几列，看是否还有颜色
            gap_tolerance = 3
            has_color_ahead = False
            for lookahead in range(1, gap_tolerance + 1):
                next_col = col + (1 if search_direction == 'right' else -1) * lookahead
                if 0 <= next_col < roi_width:
                    next_density, _, _ = analyze_column(next_col)
                    if next_density >= continue_threshold:
                        has_color_ahead = True
                        break

            if not has_color_ahead:
                break

        # 限制搜索宽度（最多250像素）
        if patch_start is not None and abs(col - patch_start) > 250:
            break

    # 如果找到了patch区域
    if patch_start is not None and len(accumulated_colors) > 0:

        # 确定patch的精确边界
        if patch_end is None:
            patch_end = patch_start

        if search_direction == 'left':
            patch_x1 = search_x_start + min(patch_start, patch_end)
            patch_x2 = search_x_start + max(patch_start, patch_end) + 1
        else:
            patch_x1 = search_x_start + min(patch_start, patch_end)
            patch_x2 = search_x_start + max(patch_start, patch_end) + 1

        # 使用垂直方向的精确裁剪
        # 分析每一行，找到实际有颜色的范围
        patch_roi = search_roi_rgb[:, min(patch_start, patch_end):max(patch_start, patch_end) + 1, :]

        if patch_roi.size > 0:
            # 找到垂直方向的实际边界（改进版：更好地处理灰色和黑色）
            row_densities = []
            for row_idx in range(patch_roi.shape[0]):
                row = patch_roi[row_idx, :, :]
                # 过滤白色背景（使用更合理的阈值）
                is_colored = ~np.all(row > 235, axis=1)
                row_densities.append(np.sum(is_colored))

            # 找到有效行的范围（使用动态阈值）
            if len(row_densities) > 0:
                max_row_density = max(row_densities)
                # 至少有1个非白色像素，或者超过最大密度的10%
                density_threshold = max(1, int(max_row_density * 0.1))
                valid_rows = [i for i, d in enumerate(row_densities) if d >= density_threshold]
            else:
                valid_rows = []

            if valid_rows:
                actual_y1 = search_y_start + min(valid_rows)
                actual_y2 = search_y_start + max(valid_rows) + 1
            else:
                actual_y1 = search_y_start
                actual_y2 = search_y_end

            # 构建最终的patch bbox
            final_patch_bbox = [
                float(patch_x1),
                float(actual_y1),
                float(patch_x2),
                float(actual_y2)
            ]

            # 提取颜色 - 多重策略验证
            try:
                # 策略1: 使用改进的get_dominant_color
                color = get_dominant_color(image, final_patch_bbox)

                # 验证颜色有效性
                if color:
                    color_array = np.array(color)
                    # 只排除纯白色（接近255的背景色）
                    is_white = np.all(color_array > 240)

                    if not is_white:
                        # 额外验证：计算累积颜色的一致性
                        if len(accumulated_colors) > 0:
                            # 计算所有扫描到的颜色的平均值
                            avg_accumulated = np.mean(accumulated_colors, axis=0)
                            # 如果提取的颜色与扫描平均颜色接近，说明提取准确
                            color_diff = np.sqrt(np.sum((color_array - avg_accumulated) ** 2))

                            # 如果差异太大，使用扫描的平均颜色
                            if color_diff > 50:
                                # 转换为Python原生int类型以支持JSON序列化
                                final_color = tuple(int(x) for x in avg_accumulated.astype(int))
                                # 验证最终颜色不是白色
                                if not np.all(np.array(final_color) > 240):
                                    return final_color
            except Exception as e:
                print(f"颜色提取失败: {e}")
                # 如果get_dominant_color失败，尝试使用扫描的平均颜色
                if len(accumulated_colors) > 0:
                    try:
                        avg_color = np.mean(accumulated_colors, axis=0).astype(int)
                        # 转换为Python原生int类型以支持JSON序列化
                        fallback_color = tuple(int(x) for x in avg_color)
                        if not np.all(avg_color > 240):
                            return fallback_color
                    except:
                        pass

    # 方法2: 如果上述方法失败，使用固定距离估算（作为后备）
    # 尝试多种不同的距离和宽度组合
    if search_direction == 'left':
        # 尝试不同的距离和patch宽度
        attempts = [
            (10, 30),  # 距离10px, 宽度30px
            (20, 35),  # 距离20px, 宽度35px
            (30, 40),  # 距离30px, 宽度40px
            (40, 45),  # 距离40px, 宽度45px
            (50, 50),  # 距离50px, 宽度50px
            (60, 40),  # 距离60px, 宽度40px
            (80, 35),  # 距离80px, 宽度35px
            (100, 30),  # 距离100px, 宽度30px
        ]

        for distance, patch_width in attempts:
            estimated_x1 = max(0, x1 - distance - patch_width)
            estimated_x2 = max(0, x1 - distance)

            if estimated_x2 > estimated_x1 + 5:  # 确保有足够的宽度
                estimated_bbox = [
                    float(estimated_x1),
                    float(y1),
                    float(estimated_x2),
                    float(y2)
                ]
                try:
                    color = get_dominant_color(image, estimated_bbox)
                    if color:
                        color_array = np.array(color)
                        # 只排除纯白色背景
                        is_white = np.all(color_array > 240)

                        if not is_white:
                            return color
                except:
                    continue
    elif search_direction == 'right':
        # 如果是右侧搜索，也尝试多种距离
        attempts = [
            (10, 30), (20, 35), (30, 40), (40, 45),
            (50, 50), (60, 40), (80, 35), (100, 30),
        ]

        for distance, patch_width in attempts:
            estimated_x1 = min(width, x2 + distance)
            estimated_x2 = min(width, x2 + distance + patch_width)

            if estimated_x2 > estimated_x1 + 5:
                estimated_bbox = [
                    float(estimated_x1),
                    float(y1),
                    float(estimated_x2),
                    float(y2)
                ]
                try:
                    color = get_dominant_color(image, estimated_bbox)
                    if color:
                        color_array = np.array(color)
                        is_white = np.all(color_array > 240)

                        if not is_white:
                            return color
                except:
                    continue

    return None


def match_legend_patches_to_labels(patches, labels, image=None):
    """
    将 legend_patch 和 legend_label 进行空间配对
    如果patches为空或数量不足，会尝试从label位置推断颜色

    Args:
        patches: list of dicts, 每个包含 'bbox' 和 'color'
        labels: list of dicts, 每个包含 'bbox' 和 'text'
        image: 原始图像，用于在缺少patch时推断颜色

    Returns:
        matched_legends: list of dicts, 包含配对后的图例信息
    """
    matched_legends = []
    used_labels = set()

    # 情况1: 有patches，进行正常配对
    if patches:
        # 按照patch的y坐标排序（从上到下）
        patches_sorted = sorted(patches, key=lambda p: (p['bbox'][1] + p['bbox'][3]) / 2)

        for patch in patches_sorted:
            patch_center = calculate_bbox_center(patch['bbox'])
            patch_x_center, patch_y_center = patch_center

            best_match = None
            best_distance = float('inf')
            best_label_idx = -1

            # 遍历所有未使用的label，找到最佳匹配
            for idx, label in enumerate(labels):
                if idx in used_labels:
                    continue

                label_center = calculate_bbox_center(label['bbox'])
                label_x_center, label_y_center = label_center

                # 检查垂直对齐（图例通常是水平排列：patch在左，label在右）
                if check_vertical_alignment(patch['bbox'], label['bbox']):
                    # 计算水平距离（优先考虑）
                    horizontal_distance = abs(label_x_center - patch_x_center)
                    vertical_distance = abs(label_y_center - patch_y_center)

                    # 综合距离（水平距离权重更大）
                    distance = horizontal_distance + vertical_distance * 0.5

                    # label应该在patch的右侧（对于标准图例布局）
                    # 但也允许一定的灵活性
                    if label_x_center >= patch['bbox'][0] - 20:  # 允许label稍微在左边
                        if distance < best_distance:
                            best_distance = distance
                            best_match = label
                            best_label_idx = idx

            # 如果找到匹配，添加到结果中
            if best_match is not None:
                matched_legends.append({
                    'bbox': best_match['bbox'],
                    'label_text': best_match['text'],
                    'color': patch.get('color', None),
                })
                used_labels.add(best_label_idx)
            else:
                # 如果没有找到匹配的label，仍然记录patch信息
                matched_legends.append({
                    'bbox': None,
                    'label_text': 'Unknown',
                    'color': patch.get('color', None),
                })

    # 情况2: 处理未匹配的labels（没有对应的patch）
    # 或者完全没有检测到patches
    unmatched_labels = [label for idx, label in enumerate(labels) if idx not in used_labels]

    if unmatched_labels and image is not None:
        for label in unmatched_labels:
            # 尝试从label位置推断patch颜色（使用改进的精准算法）
            # 先尝试both方向（左右都尝试），提高识别成功率
            color = infer_patch_color_from_label(
                image,
                label['bbox'],
                search_direction='both',  # 改为both，自动尝试左右两侧
                search_distance=200  # 增加搜索距离
            )
            if color:
                matched_legends.append({
                    'bbox': label['bbox'],
                    'label_text': label['text'],
                    'color': color,
                })
            else:
                # 如果仍然失败，记录一下以便调试
                print(f"无法推断图例颜色: {label['text']}")
                matched_legends.append({
                    'bbox': label['bbox'],
                    'label_text': label['text'],
                    'color': None,
                })
    elif unmatched_labels:
        # 没有图像，无法推断颜色
        for label in unmatched_labels:
            matched_legends.append({
                'bbox': label['bbox'],
                'label_text': label['text'],
                'color': None,
            })

    return matched_legends


img_folder = 'module/detector/figs' #待修改
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']

# 获取所有图片路径
img_list = []
for ext in image_extensions:
    img_list.extend(glob.glob(os.path.join(img_folder, ext)))

# 遍历处理
for img_path in img_list:
    img_name = os.path.basename(img_path)  # 获取: '40.tif'
    img_stem = Path(img_path).stem  # 获取: '40' (无扩展名)

    # test a single image and show the results
    img = f"{img_path}"  # or img = mmcv.imread(img), which will only load it once
    image_cv = cv2.imread(f"{img_path}")
    result = inference_detector(model, img)

    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file=f'module/detector/result/result_img/{img_name}')

    # 设置置信度阈值
    score_threshold = 0.5
    score_threshold_xlabel = 0.6
    score_threshold_ytitle = 0.8
    score_threshold_legndlabel = 0.6
    score_threshold_marklabel = 0.3

    detection_results = {
        'coordinates': {
            'x': {},  # 5
            'y': {},  # 6
        },
        'legend': {},  # 9
        'axis_label': {},  # y1 x2
        'plot_area': {},  # 2
        'legend_area': {}  # 12
    }
    """
    classes=[
                0'x_title', 1'y_title', 2'plot_area', 3'other', 4'xlabel', 5'ylabel',
                6'chart_title', 7'x_tick', 8'y_tick', 9'legend_patch', 10'legend_label',
                11'legend_title', 12'legend_area', 13'mark_label', 14'value_label',
                15y_axis_area', 16'x_axis_area', 17'tick_grouping'
            ] 
    """
    # 临时存储 legend_patch 和 legend_label 的信息
    legend_patches = []
    legend_labels = []

    # 遍历每个类别
    for class_id, class_result in enumerate(result):
        # class_result 是一个 (N, 5) 的数组，每行是 [x1, y1, x2, y2, score]
        if class_id == 4:  # xlabel
            for bbox in class_result:
                if bbox[4] >= score_threshold_xlabel:  # bbox[4] 是置信度分数
                    bbox_coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    detected_text = extract_text_with_paddleocr(image_cv, bbox_coords)
                    detection_results['coordinates']['x'].update({
                        f'{detected_text}': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    })

        elif class_id == 5:  # ylabel
            for bbox in class_result:
                if bbox[4] >= score_threshold:
                    bbox_coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    detected_text = extract_text_with_paddleocr(image_cv, bbox_coords)
                    detection_results['coordinates']['y'].update({
                        f'{detected_text}': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    })

        elif class_id == 9:  # legend_patch  13mark_label
            for bbox in class_result:
                if bbox[4] >= score_threshold or bbox[4] >= score_threshold_marklabel:
                    bbox_coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    rgb_color = get_dominant_color(image_cv, bbox_coords)
                    legend_patches.append({
                        'bbox': bbox_coords,
                        'color': [int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])]
                    })

        elif class_id == 10:  # legend_label  13mark_label
            for bbox in class_result:
                if bbox[4] >= score_threshold_legndlabel or bbox[4] >= score_threshold_marklabel:
                    bbox_coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    detected_text = extract_text_with_paddleocr(image_cv, bbox_coords)
                    legend_labels.append({
                        'bbox': bbox_coords,
                        'text': detected_text,
                    })

        elif class_id == 13:  # 13mark_label
            for bbox in class_result:
                if bbox[4] >= score_threshold_marklabel:
                    bbox_coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    detected_text = extract_text_with_paddleocr(image_cv, bbox_coords)
                    rgb_color = get_dominant_color(image_cv, bbox_coords)
                    legend_labels.append({
                        'bbox': bbox_coords,
                        'text': detected_text,
                    })
                    legend_patches.append({
                        'bbox': bbox_coords,
                        'color': [int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])]
                    })

        elif class_id == 0:  # x_title
            for bbox in class_result:
                if bbox[4] >= score_threshold:
                    bbox_coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    detected_text = extract_text_with_paddleocr(image_cv, bbox_coords)
                    if detected_text:
                        detection_results['axis_label']['x'] = detected_text

        elif class_id == 1:  # y_title (竖直文字，需要旋转)
            for bbox in class_result:
                if bbox[4] >= score_threshold_ytitle:
                    bbox_coords = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
                    # 旋转识别竖直文字：rotate=True, 顺时针旋转90度将竖直文字转为水平
                    detected_text = extract_text_with_paddleocr(image_cv, bbox_coords, rotate=True, rotation_angle=90)
                    if detected_text:
                        detection_results['axis_label']['y'] = detected_text

        elif class_id == 2:  # plot_area
            for bbox in class_result:
                if bbox[4] >= score_threshold:
                    detection_results['plot_area'].update({
                        'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    })

        elif class_id == 12:  # legend_area
            for bbox in class_result:
                if bbox[4] >= score_threshold:
                    detection_results['legend_area'].update({
                        'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    })
    # 进行图例配对（传入图像以支持颜色推断）
    if legend_patches or legend_labels:
        matched_legends = match_legend_patches_to_labels(legend_patches, legend_labels, image=image_cv)

        # 将配对结果存储到 detection_results
        for idx, legend_item in enumerate(matched_legends):
            legend_key = legend_item['label_text']
            legend_data = {
                'bbox': legend_item['bbox'],
                'color': legend_item['color'],
            }
            detection_results['legend'][legend_key] = legend_data
    else:
        print("未检测到图例信息")

    # 保存为JSON文件（确保所有数据都是JSON可序列化的）
    # 将所有 numpy 类型转换为 Python 原生类型
    detection_results_serializable = ensure_json_serializable(detection_results)

    with open(f'module/detector/result/result_json/{img_stem}.json', 'w', encoding='utf-8') as f: #待修改
        json.dump(detection_results_serializable, f, indent=4, ensure_ascii=False)
