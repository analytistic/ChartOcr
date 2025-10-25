from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
from  utils import Abnormal_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components



class ChartDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = init_detector(self.cfg.detector.config_path, self.cfg.detector.checkpoint_path, device=self.cfg.detector.device)
        self.classes = self.model.CLASSES
        self.filter = Abnormal_filter(cfg.detector.filter)






    def _tojson(self, input):
        pass

    def _combinebbox(self, input, thr1, thr2, area=(True, True), axis='y'):
        """
        合并图例框, xlabel, ylabel, other, legend_label 合并y_mid 命中的框， 去除扁框，去除绘图区外的框
        input: list[array, plot]
        thr1: bbox_set 的阈值
        thr2: 按y轴差合并框时, y轴差的阈值
        area: 按x, y区域筛选
        axis: 按x, y轴合并框
        """
        if input[-1] is None:
            return None
        


        
        x1, y1, x2, y2, _ = input[-1][0]
        bbox_set = input[:-1]
        axis1, axis2 = (3, 4) if axis == 'y' else (1, 2)

        bbox_set = np.concatenate(bbox_set, axis=0)

        mask = (
            ((bbox_set[:, 0] >= x1) | (not area[0])) &
            ((bbox_set[:, 2] <= x2) | (not area[0])) &
            ((bbox_set[:, 1] >= y1) | (not area[1])) &
            ((bbox_set[:, 3] <= y2) | (not area[1])) &
            (bbox_set[:, 4] >= thr1)
        )
        if not mask.any():
            return bbox_set

        bbox_set = bbox_set[mask][:, [4, 0, 2, 1, 3]] # M, 5


        diff_axis_1 = np.abs(bbox_set[:, axis1, None] - bbox_set[None, :, axis1]) # M, M
        diff_axis_2 = np.abs(bbox_set[:, axis2, None] - bbox_set[None, :, axis2]) # M, M

        diff_axis_0 = np.abs(bbox_set[:, axis1] - bbox_set[:, axis2]) # M


        diff_axis_0 = np.minimum(diff_axis_0[:, None], diff_axis_0[None, :]) * thr2
        merge_pair = (diff_axis_1 <= diff_axis_0) | (diff_axis_2 <= diff_axis_0)

        if axis == 'x':
            diff_axis_12 = np.abs(bbox_set[:, axis1, None] - bbox_set[None, :, axis2]) # M, M
            cross_mean = np.mean(np.min(diff_axis_12, axis=1))
            merge_pair = merge_pair | (diff_axis_12 <= cross_mean)

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

        merged = np.array(merged)  # K x 5
        merged = merged[np.argsort(-merged[:, 4])] 
        return merged



    def _getRGB(self, input):
        """
        思路是legend_label, 寻找颜色
        """
        pass

    def _ocr(self, input):
        pass


    def predict(self, img_pth):
        result = inference_detector(self.model, img_pth)
        return result
    
    def postprocess(self, result):
        """
        后处理

        1. 合并图例检测框, 去除x,ylabel异常框
        2. 获取图例颜色
            
        ['x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel', 'chart_title', 
        'x_tick', 'y_tick', 'legend_patch', 'legend_label', 'legend_title', 'legend_area', 'mark_label', 'value_label', 'y_axis_area', 'x_axis_area', 'tick_grouping']

        - result, 图例颜色
        return:
            list[array], list[array]
        """
        xlabel_bbox = result[4]
        ylabel_bbox = result[5]
        legendlabel_bbox = result[10]
        rgb_bbox = result[9]
        other_bbox = result[3]
        plotarea_bbox = result[2]
        mark_label = result[13]
        x_axis_area = result[16]
        y_axis_area = result[15]


        legendlabel_bbox = self._combinebbox(
            input=[legendlabel_bbox, other_bbox, xlabel_bbox, ylabel_bbox, mark_label, plotarea_bbox],
            thr1=self.cfg.detector.combine_legend.thr1,
            thr2=self.cfg.detector.combine_legend.thr2,
            area=(True, True),
            axis='y',
        )

        xlabel_bbox = self._combinebbox(
            input=[xlabel_bbox, x_axis_area],
            thr1=self.cfg.detector.combine_xlabel.thr1,
            thr2=self.cfg.detector.combine_xlabel.thr2,
            area=(False, True), 
            axis='x',
        )

        ylabel_bbox = self._combinebbox(
            input=[ylabel_bbox, y_axis_area],
            thr1=self.cfg.detector.combine_ylabel.thr1,
            thr2=self.cfg.detector.combine_ylabel.thr2,
            area=(True, False), 
            axis='y',
        )

        result[10] = legendlabel_bbox
        result[4] = xlabel_bbox
        result[5] = ylabel_bbox
        result[9] = np.empty((0, 5))
        result[12] = np.empty((0, 5))


        return result, None

    
    def getjson(self, img_pth):
        """
        1. 获得result
        2. 后处理result
        3. list转dict
        4. 获取ocr结果
        
        return dict
        
        """
        result = self.predict(img_pth)
        result, _ = self.postprocess(result)


        return None
    
    @staticmethod
    def plot(model, img_pth, result, score_thr, out_file='result.jpg'):
        """
        接受后处理后的result, 绘制结果
        """
        model.show_result(img_pth, result, score_thr=score_thr, out_file=out_file)





