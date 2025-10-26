from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
from  utils import Abnormal_filter
from .utils.types import DetectionResult
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import colorgram
from PIL import Image



class ChartDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = init_detector(self.cfg.detector.config_path, self.cfg.detector.checkpoint_path, device=self.cfg.detector.device)
        self.classes = self.model.CLASSES
        self.filter = Abnormal_filter(cfg.detector.filter)
        self.dete_result = DetectionResult(class_names=list(self.classes))

    @staticmethod
    def extractRGB(img, rgb_bbox, back_color_thr=253, color_num=5):
        colors = []
        for bbox in rgb_bbox:
            crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
            crop = crop[..., ::-1]
            crop_pil = Image.fromarray(crop)
            extracted = colorgram.extract(crop_pil, color_num)
            non_white = [c for c in extracted if not (c.rgb.r > back_color_thr and c.rgb.g > back_color_thr and c.rgb.b > back_color_thr)]
            if non_white:
                main_color = non_white[0]
                color = np.array([main_color.rgb.r, main_color.rgb.g, main_color.rgb.b], dtype=np.uint8)
            else:
                color = np.array([0, 0, 0], dtype=np.uint8)
            colors.append(color)
        return np.array(colors)

    def _tojson(self, input):
        pass

    def _combinebbox(self, input, thr1, thr2, area=(True, True, None), axis='y'):
        """
            input: list[array]
            thr1: bbox_set 的阈值
            thr2: 按y轴差合并框时, y轴差的阈值
            area: x, y, area_bbox
            axis: 按x, y轴合并框
        return: list[array]
        """
        if area[-1] is None:
            return np.empty((0, 5))
        
        px1, py1, px2, py2, _ = area[-1]
        bbox_set = input
        axis1, axis2 = (3, 4) if axis == 'y' else (1, 2)
        bbox_set = np.concatenate(bbox_set, axis=0)

        mask = (
            (((bbox_set[:, 0] <= px2) & (bbox_set[:, 2] >= px1)) | (not area[0])) &
            (((bbox_set[:, 1] <= py2) & (bbox_set[:, 3] >= py1)) | (not area[1])) &
            (bbox_set[:, 4] >= thr1)
        )
        if not mask.any():
            return bbox_set

        bbox_set = bbox_set[mask][:, [4, 0, 2, 1, 3]]
        diff_axis_1 = np.abs(bbox_set[:, axis1, None] - bbox_set[None, :, axis1])
        diff_axis_2 = np.abs(bbox_set[:, axis2, None] - bbox_set[None, :, axis2])
        diff_axis_0 = np.abs(bbox_set[:, axis1] - bbox_set[:, axis2])

        diff_axis_0 = np.minimum(diff_axis_0[:, None], diff_axis_0[None, :]) * thr2
        merge_pair = (diff_axis_1 <= diff_axis_0) | (diff_axis_2 <= diff_axis_0)
        if axis == 'x':
            diff_axis_12 = np.abs(bbox_set[:, axis1, None] - bbox_set[None, :, axis2])
            cross_mean = np.mean(np.min(diff_axis_12, axis=1))
            merge_pair = merge_pair | (diff_axis_12 <= cross_mean * thr2)
        np.fill_diagonal(merge_pair, False)

        if not merge_pair.any():
            return bbox_set[:, [1, 3, 2, 4, 0]]
        
        n_comp, labels = connected_components(csr_matrix(merge_pair), directed=False)
        merged = []
        for comp_id in range(n_comp):
            members = np.where(labels == comp_id)[0]
            mx1 = float(bbox_set[members, 1].min())
            my1 = float(bbox_set[members, 3].min())
            mx2 = float(bbox_set[members, 2].max())
            my2 = float(bbox_set[members, 4].max())
            mscore = float(bbox_set[members, 0].max())
            merged.append([mx1, my1, mx2, my2, mscore])

        merged = np.array(merged)
        merged = merged[np.argsort(-merged[:, 4])] 
        return merged
    
    def _getRGB(self, input, img, thr, back_color_thr=253, color_num=5):
        """
            input: list[legend_label, rgb_bbox, legend_area]
            img: 图像
            thr: rgbbboxs筛选阈值
            back_color_thr: 背景颜色阈值
            color_num: 获取颜色数量
        return: list[array], list[array]
        """
        legend_label, rgb_bbox, legend_area = input
        legend_area = legend_area[0] if legend_area is not None and legend_area[0][-1] >= thr else None
        if legend_area is None:
            rgb_colors = self.extractRGB(img, legend_label, back_color_thr=back_color_thr, color_num=color_num)
            return legend_label, rgb_colors
        
        rgb_bbox = rgb_bbox[rgb_bbox[:, 4] >= thr]
        label_mid_y = legend_label[:, [1, 3]].mean(axis=1)
        rgb_mid_y = rgb_bbox[:, [1, 3]].mean(axis=1)
        label_mid_h = np.mean(legend_label[:, 3] - legend_label[:, 1])
          
        diff_y = np.abs(label_mid_y[:, None] - rgb_mid_y[None, :])
        diff_y_sign = label_mid_y[:, None] - rgb_mid_y[None, :]
        min_index = np.argmin(diff_y, axis=1)
        diff_y = diff_y[np.arange(len(label_mid_y)), min_index]
        diff_y_sign = diff_y_sign[np.arange(len(label_mid_y)), min_index]
        mask = (diff_y <= label_mid_h)
        rgb_bbox = rgb_bbox[min_index]

        if (~mask).any():
            rgb_mid_h = np.mean(rgb_bbox[mask][:, 3] - rgb_bbox[mask][:, 1]) if mask.any() else label_mid_h
            mid_diff_y = diff_y_sign[mask].mean() if mask.any() else 0
            redete_index = ~mask
            x1 = rgb_bbox[mask, 0].mean() if mask.any() else legend_area[0]
            x2 = rgb_bbox[mask, 2].mean() if mask.any() else legend_area[2]
            y1 = label_mid_y[redete_index] + mid_diff_y - rgb_mid_h / 2
            y2 = label_mid_y[redete_index] + mid_diff_y + rgb_mid_h / 2
            rgb_bbox[redete_index, 0] = x1
            rgb_bbox[redete_index, 2] = x2
            rgb_bbox[redete_index, 1] = y1
            rgb_bbox[redete_index, 3] = y2

        rgb_colors = self.extractRGB(img, rgb_bbox, back_color_thr=back_color_thr, color_num=color_num)
        return rgb_bbox, rgb_colors

    def _toocr(self, input):
        pass

    def predict(self, img_pth):
        result = inference_detector(self.model, img_pth)
        return result
    
    def postprocess(self, img):
        """
        后处理
        1. 合并图例检测框, 去除x,ylabel异常框
        2. 获取图例颜色
            result['x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel', 'chart_title', 
                    'x_tick', 'y_tick', 'legend_patch', 'legend_label', 'legend_title', 'legend_area',
                    'mark_label', 'value_label', 'y_axis_area', 'x_axis_area', 'tick_grouping']
            img
        return: list[array], list[array]
        """
        xlabel_bbox = self.dete_result.get_bboxes('xlabel')
        ylabel_bbox = self.dete_result.get_bboxes('ylabel')
        legendlabel_bbox = self.dete_result.get_bboxes('legend_label')
        other_bbox = self.dete_result.get_bboxes('other')
        plotarea_bbox = self.dete_result.get_bboxes('plot_area')
        mark_label = self.dete_result.get_bboxes('mark_label')
        x_axis_area = self.dete_result.get_bboxes('x_axis_area')
        y_axis_area = self.dete_result.get_bboxes('y_axis_area')
        rgb_bbox = self.dete_result.get_bboxes('legend_patch')
        legend_area = self.dete_result.get_bboxes('legend_area')


        legend_area_temp = legend_area[0] if legend_area is not None and legend_area[0][-1] >= self.cfg.detector.combine_legend.thr1 else None
        plot_area_temp = plotarea_bbox[0] if plotarea_bbox is not None and plotarea_bbox[0][-1] >= self.cfg.detector.combine_legend.thr1 else None
        area_temp = legend_area_temp if legend_area_temp is not None else plot_area_temp
        legendlabel_bbox = self._combinebbox(
            input=[legendlabel_bbox, other_bbox, xlabel_bbox, ylabel_bbox, mark_label],
            thr1=self.cfg.detector.combine_legend.thr1,
            thr2=self.cfg.detector.combine_legend.thr2,
            area=(True, True, area_temp),
            axis='y',
        )

        xlabel_bbox = self._combinebbox(
            input=[xlabel_bbox],
            thr1=self.cfg.detector.combine_xlabel.thr1,
            thr2=self.cfg.detector.combine_xlabel.thr2,
            area=(False, True, x_axis_area[0]), 
            axis='x',
        )

        ylabel_bbox = self._combinebbox(
            input=[ylabel_bbox],
            thr1=self.cfg.detector.combine_ylabel.thr1,
            thr2=self.cfg.detector.combine_ylabel.thr2,
            area=(True, False, y_axis_area[0]),
            axis='y',
        )

        rgb_bbox, rgb_colors = self._getRGB(
            input=[legendlabel_bbox, rgb_bbox, legend_area],
            img=img,
            thr = self.cfg.detector.getRGB.thr,
            back_color_thr=self.cfg.detector.getRGB.back_color_thr,
            color_num=self.cfg.detector.getRGB.color_num,
        )

        self.dete_result.set_bboxes('legend_label', legendlabel_bbox)
        self.dete_result.set_bboxes('xlabel', xlabel_bbox)
        self.dete_result.set_bboxes('ylabel', ylabel_bbox)
        self.dete_result.set_bboxes('legend_patch', rgb_bbox)
        self.dete_result.set_bboxes('legend_area', np.empty((0, 5)))
        return rgb_colors

    def getjson(self, img_pth):
        """
        return dict
        """
        img = mmcv.imread(img_pth)
        self.dete_result.bboxes_list = self.predict(img_pth)
        rgb_colors = self.postprocess(img)


        return None
    
    @staticmethod
    def plot(model, img_pth, result, score_thr, out_file='result.jpg'):
        model.show_result(img_pth, result, score_thr=score_thr, out_file=out_file)





