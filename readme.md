# Fracture Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green) ![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)

## Overview
**Fracture Detection System** is a full-cycle Deep Learning application for automated X-ray analysis. Unlike standard object detection wrappers, this project implements a **custom two-stage pipeline** from scratch:
1.  **Classification:** Determines if the bone is fractured (Custom ResNet-like CNN).
2.  **Detection:** Localizes the fracture area using Segmentation (Custom U-Net with heatmap post-processing).

The system is wrapped in a high-performance **FastAPI** backend with a user-friendly Web UI.

**Key Features:**
*   **Custom Architectures:** Implemented CNN (with Residual Blocks) and U-Net from scratch in PyTorch.
*   **Segmentation-based Detection:** Uses pixel-level heatmaps to generate bounding boxes.
*   **Full Web Stack:** FastAPI backend + SQLite database + HTML/JS Frontend.
*   **Real-time Inference:** Optimized for CPU/GPU inference.

## Tech Stack
*   **Core:** Python 3.10+
*   **DL Framework:** PyTorch, Torchvision
*   **Computer Vision:** OpenCV, PIL, NumPy
*   **Backend:** FastAPI, Uvicorn
*   **Frontend:** HTML5, JavaScript (Vanilla)

## Installation & Usage

1.  **Clone the repository**
    ```
    git clone https://github.com/your-username/fracture-detection.git
    cd fracture-detection
    ```

2.  **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

3.  **Run the server**
    ```
    python main.py
    ```
    The server will start at `http://0.0.0.0:8000`.

4.  **Open Web UI**
    Open your browser and visit `http://localhost:8000`.

## Project Structure

## Dataset
This project uses a cleaned and unified dataset hosted on Kaggle:
👉 **[Clean Bone Fracture Detection Dataset](https://www.kaggle.com/datasets/dmtrrmnv/clean-bone-fracture)**


# 骨折检测系统 (Fracture Detection System)

## 简介
这是一个用于X光骨折检测的深度学习系统。与普通的检测工具不同，本项目完全使用 **PyTorch** 从零搭建了一个**两阶段模型**：
1.  **分类 (Classification):** 判断骨头是否骨折（使用自定义的 ResNet 结构）。
2.  **检测 (Detection):** 定位骨折的位置（使用 U-Net 分割网络）。

系统包含了完整的 **FastAPI** 后端和网页界面，可以直接上传图片进行测试。

## 主要功能
*   **自定义网络结构:** 手写实现了 CNN (残差模块) 和 U-Net 模型。
*   **基于分割的检测:** 通过生成热力图 (Heatmap) 来计算骨折区域。
*   **完整的 Web 系统:** 包含 FastAPI 后端、数据库和前端页面。
*   **实时推理:** 优化了代码，支持 CPU 和 GPU 运行。

## 技术栈
*   **核心:** Python 3.10+
*   **深度学习:** PyTorch, Torchvision
*   **计算机视觉:** OpenCV, PIL
*   **后端:** FastAPI, Uvicorn

## 如何使用

1.  **安装依赖**
    ```
    pip install -r requirements.txt
    ```

2.  **运行服务器**
    ```
    python main.py
    ```

3.  **打开网页**
    在浏览器中访问 `http://localhost:8000`，上传X光图片即可看到结果。

## 数据集
本项目使用了我在 Kaggle 上发布的清洗后的数据集：
👉 **[Clean Bone Fracture Detection Dataset](https://www.kaggle.com/datasets/dmtrrmnv/clean-bone-fracture)**
