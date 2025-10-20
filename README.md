## Install Environment

项目依赖[MMdetection Framework](https://github.com/open-mmlab/mmdetection).


## workflow

<img src="chartocr workflow.jpg" width="500">

## 项目结构
```text
ChartOcr/
├── data/
│   ├── input/
│   └── output/
├── module/
│   ├── extractor/
│   │   ├── config.py
│   │   ├── lineEX_extector.py
│   │   └── lineformer_extector.py
│   ├── detector/
│   │   ├── config.py
│   │   └── chart_element_detector.py
│   ├── transform/
│   │   ├── config.py
│   │   └── pixel_transform.py
│   └── ocr/
│       └── ocr_model.py
├── model/
│   ├── config.py
│   └── chartocr.py
├── main.py
├── test.sh
└── ...

```

## 接口说明

### 数据流程概述



### 1. Detector 模块

**输入：**
- 图片文件（支持格式：PNG, JPG, JPEG）

**输出：**
- 图表元素检测 JSON 文件





### 2. Extractor 模块

**输入：**
- 原始图片文件
- 图表元素检测 JSON 文件（特别是 plot_area 的 bbox 信息）

**输出：**
- 曲线像素坐标提取 JSON 文件





### 3. Transform 模块

**输入：**
- 图表元素检测 JSON 文件
- 曲线像素坐标提取 JSON 文件

**输出：**
- 曲线真实坐标 JSON 文件





            