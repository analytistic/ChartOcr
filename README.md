## Install Environment

项目依赖[MMdetection Framework](https://github.com/open-mmlab/mmdetection).
环境管理依赖uv

```bash
git clone https://github.com/analytistic/ChartOcr.git
# 如果服务器太慢 用ssh绕过
git clone git@github.com:analytistic/ChartOcr.git
cd ChartOcr
git submodule update --init --recursive
bash install.sh
uv sync
```

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

# 流程 2025-10-21

Chart Ocr开发环境

```bash
git clone https://github.com/analytistic/ChartOcr.git
# 如果服务器太慢 用ssh绕过
git clone git@github.com:analytistic/ChartOcr.git
cd ChartOcr
git submodule update --init --recursive # 初始化子模块mmdet,这里子模块的修改有自己的git仓库,是从官方仓库fork的
uv sync # 用来1.依照pyproject安装(更新)环境,2.构建整个项目(会把整个项目构建成一个包,过程中忽略mmdet文件夹)可能会遇到没安装torch的问题
source .venv/bin/activate

# 先安装setuptools和torch
uv pip install setuptools>=75.3.2
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
uv sync

# 可编译安装mmdet
cd mmdetection
git checkout v2.28.2
git checkout -b feature/lineformerv2.28.2  # 或其他分支名
cd ..
uv pip install -e mmdetection --no-build-isolation
uv sync
uv run mim install "mmcv-full==1.7.1"
```

fork/main         →  跟踪官方 main 保持同步,不用做开发
fork/for_chartocr → 用作子模块的合并分支

配置提交身份

```bash
git config user.name "你的名字"
git config user.email "你的邮箱"
```

开发新功能分支

```bash
git checkout -b feature/xxx
```
