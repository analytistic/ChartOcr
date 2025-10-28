## ChartOcr——科研图表自动识别与结构化数据提取工具


ChartOcr 是一款面向科研图表的自动识别与结构化数据提取工具，专为学术论文、科技报告等场景设计。其核心亮点在于：无需人工干预，即可从各类折线图片中自动识别图例、坐标轴、解析曲线数据，输出结构化结果，极大提升科研数据复用效率。

本项目基于 MMDetection (mmdet) 框架，创新性地融合了目标检测模型、文本识别技术与数据提取算法，实现了对图表元素（如坐标轴标签、图例、采样点等）的高精度定位与曲线数据解析。系统支持中英文混合文本识别，能够自动处理图片旋转、异常值过滤等复杂情况，确保数据提取结果的准确性与鲁棒性。

通过模块化架构设计，项目支持自定义异常值过滤方法（如 MAD、IQR、Z-Score）及多种子模型灵活切换，可适配科研分析、数据挖掘、知识管理等多种应用场景。此外，系统基于 uv 构建，可通过 MCP 协议无缝集成至各类智能体框架，为智能化信息处理提供强大的科研曲线图解析能力。

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
├── config/
├── data/
│   ├── input/
│   └── output/
├── module/
│   ├── extractor/
│   ├── detector/
│   ├── transform/
│   └── ocr/
├── model/
│   └── chartocr.py
├── main.py

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
