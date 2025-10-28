import torch
from utils import ChartElementResult, LineResult
import numpy as np
import mmcv
import cv2

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
        indices = torch.multinomial(prob.T, num, replacement=True)
        return indices

    
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
        return torch.stack(samples_per_color, dim=0)


    def sample(self, mask, W, num):
        index = self.strategy[self.cfg.extractor.sampler](mask, num)
        y = index // W
        x = index % W
        return torch.stack([x, y], dim=-1) # [N, num, 2]
    

class LineExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.extractor.device
        self.sampler = Sampler(cfg)



    


    def getjson(self, img, detector_result: ChartElementResult):
        if isinstance(img, str):
            img = mmcv.imread(img)
            img = mmcv.bgr2rgb(img)

        plot_area = detector_result.plot.bbox
        legend_area = detector_result.legends.area
        legend_label = detector_result.legends.label
        img[int(legend_area[0, 1]):int(legend_area[0, 3]), int(legend_area[0, 0]):int(legend_area[0, 2])] = 255
        for i in range(legend_label.bbox.shape[0]):
            x1, y1, x2, y2 = map(int, legend_label.bbox[i, :4])
            img[y1:y2, x1:x2] = 0
        img = img[int(plot_area[0, 1]):int(plot_area[0, 3]), int(plot_area[0, 0]):int(plot_area[0, 2])]
        W = img.shape[1]
        color = torch.from_numpy(detector_result.legends.label.color).float().to(self.device)
        img = torch.from_numpy(img).to(self.device) 
        pixels = img.reshape(-1, 3).float()

        if self.cfg.extractor.color.dist == 'infinity':
            diff = pixels.unsqueeze(1) - color.unsqueeze(0)
            dists = diff.abs().max(dim=-1)[0]
        else:
            dists = torch.cdist(pixels, color, p=self.cfg.extractor.color.dist) # [H*W,N]

        masks = dists < self.cfg.extractor.color.thr

        pixels = self.sampler.sample(masks, W, self.cfg.extractor.num_points)
        pixels = pixels + torch.tensor([int(plot_area[0, 0]), int(plot_area[0, 1])]).to(self.device)

        pixels = pixels.cpu().numpy()

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


       




        






        