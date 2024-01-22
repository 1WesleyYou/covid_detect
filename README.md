# 新冠检测

## 简介

这是一个深度学习 unet 网络的学习训练代码仓库,基于 pytorch 框架

## 数据集

来自 [巴西圣保罗医院的covid病毒(非)患者的CT扫描图像](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) 

## 训练

### 先前准备

本代码基于 `pytorch` 框架, 所以请提前下载 pytorch, 建议使用 conda 环境

### 使用

运行 `detect.py` 就可以, 命令行指令

```bash
python detect.py
```

### 代码分布

detect.py 是主文件, 包含了训练的逻辑

model.py 定义了神经网络模型, 包含了 unet 网络 

preload.py 用于读取数据集,也就是将图片存入指针,同时根据路径存入 label 的值. 项目的目录结构如下
```txt
covid_detect
├─dataset ----数据集
│  ├─COVID ---------结果是covid的图片
│  ├─non-COVID --------结果不是covid的图片
├─.gitignore ----配置文件
├─detect.py --------检测训练逻辑
├─model.py -------神经网络模型定义
├preload.py --------读取数据集
```
