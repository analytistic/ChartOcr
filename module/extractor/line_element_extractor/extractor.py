import torch
from utils import ChartElementResult, LineResult
import colorgram
import numpy as np
import mmcv
import cv2
from PIL import Image 
from utils.tools import rgb_to_hsv_torch


class Sampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.strategy = {
            'multinomial': self._multinomial,
            'singlenomial': self._singlenomial,
        }


    def _multinomial(self, mask, num):
        weights = mask.float()
        weights_sum = weights.sum(dim=0, keepdim=True).clamp_min(1)
        prob = weights / weights_sum
        valid_index = prob.sum(dim=0) != 0
        prob = prob[:, valid_index]
        indices = torch.multinomial(prob.T, num, replacement=False)

        return indices, valid_index

    
    def _singlenomial(self, mask, num):
        samples_per_color = []
        for i in range(mask.shape[1]):
            valid_idx = torch.nonzero(mask[:, i], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                samples = torch.empty((0,), device=mask.device, dtype=torch.long)
            else:
                sel = torch.randint(0, valid_idx.numel(), (num,), device=mask.device)
                samples = valid_idx[sel]
            samples_per_color.append(samples)
        return torch.stack(samples_per_color, dim=0), None


    def sample(self, mask, W, num):
        index, valid_label = self.strategy[self.cfg.extractor.sampler](mask, num)
        y = index // W
        x = index % W
        return torch.stack([x, y], dim=-1), valid_label # [N, num, 2]
    

class LineExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.extractor.device
        self.sampler = Sampler(cfg)


    def _img_preprocess(self, img, plot_area, legend_area, legend_label):
        bt = self.cfg.extractor.color.bth_sacle
        pixel_bar = int(img.shape[1] * bt)
        x1, y1, x2, y2 = map(int, plot_area[0, :4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), pixel_bar)
        if legend_area.any():
            lx1, ly1, lx2, ly2 = map(int, legend_area[0, :4])
            cv2.rectangle(img, (lx1-pixel_bar, ly1-pixel_bar), (lx2+pixel_bar, ly2+pixel_bar), (255, 255, 255), -1)
        for i in range(legend_label.bbox.shape[0]):
            try:
                sx1, sy1, sx2, sy2 = map(int, legend_label.bbox[i, :4])
                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 255, 255), -1)
            except:
                pass
        return img
    
    


    def getjson(self, img, detector_result: ChartElementResult):
        if isinstance(img, str):
            img = mmcv.imread(img)
        img = mmcv.bgr2rgb(img)

        plot_area = detector_result.plot.bbox
        legend_area = detector_result.legends.area
        legend_label = detector_result.legends.label
        img = self._img_preprocess(img, plot_area, legend_area, legend_label)
        img = img[int(plot_area[0, 1]):int(plot_area[0, 3]), int(plot_area[0, 0]):int(plot_area[0, 2])]
        W = img.shape[1]
        color = torch.from_numpy(detector_result.legends.label.color).float().to(self.device)
        img = torch.from_numpy(img).to(self.device) 
        pixels = img.reshape(-1, 3).float()

        if self.cfg.extractor.color.dist == 'infinity':
            diff = pixels.unsqueeze(1) - color.unsqueeze(0)
            dists = diff.abs().max(dim=-1)[0]
        elif self.cfg.extractor.color.dist == 'hsv':
            pixels_hsv = rgb_to_hsv_torch(self, pixels/255)
            color_hsv = rgb_to_hsv_torch(self, color/255)
            dh = torch.min(torch.abs(pixels_hsv[:, 0].unsqueeze(1) - color_hsv[:, 0].unsqueeze(0)), 1 - torch.abs(pixels_hsv[:, 0].unsqueeze(1) - color_hsv[:, 0].unsqueeze(0))) * 2
            ds = pixels_hsv[:, 1].unsqueeze(1) - color_hsv[:, 1].unsqueeze(0)
            dv = pixels_hsv[:, 2].unsqueeze(1) - color_hsv[:, 2].unsqueeze(0)

            dists = torch.sqrt(dh ** 2 + ds ** 2 + dv ** 2)
        else:
            dists = torch.cdist(pixels, color, p=self.cfg.extractor.color.dist) # [H*W,N]

        masks = (dists <= self.cfg.extractor.color.thr_high) & (dists >= self.cfg.extractor.color.thr_low)

        pixels, valid_index = self.sampler.sample(masks, W, self.cfg.extractor.num_points)
        pixels = pixels + torch.tensor([int(plot_area[0, 0]), int(plot_area[0, 1])]).to(self.device)

        pixels = pixels.cpu().numpy()


        valid_index = torch.where(valid_index)
        valid_index = valid_index[0].cpu().numpy()
        detector_result.legends.label.bbox = detector_result.legends.label.bbox[valid_index, :]
        detector_result.legends.label.color = detector_result.legends.label.color[valid_index, :]
        text = [detector_result.legends.label.text[i] for i in range(len(valid_index))]

        detector_result.legends.label.text = text

        return LineResult(
            legends=detector_result.legends,
            x_pixel=pixels[:, :, 0],
            y_pixel=pixels[:, :, 1]
        )
    

    @staticmethod
    def plot(img_pth, result: LineResult, out_file='result.jpg', highlight_size = 10):
        x_pixel = result.x_pixel.reshape(-1)
        y_pixel = result.y_pixel.reshape(-1)
        img = mmcv.imread(img_pth)
        dark = (img * 0.3).astype(np.uint8) # H, W, 3
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        mask[y_pixel, x_pixel] = True
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((highlight_size, highlight_size), np.uint8))
        dark[mask > 0] = img[mask > 0]
        mmcv.imwrite(dark, out_file)


       




        






        