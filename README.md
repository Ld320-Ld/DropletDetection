# 微流控液滴检测项目

本项目基于YOLOv8实现微流控芯片中液滴的自动检测，适用于数字微流控（DMF）平台的液滴定位、分裂等操作的智能化升级。

---

## 目录结构说明

```
├── 001-500/               # 原始数据集存放位置
├── 501-1100/              # 原始数据集存放位置
├── main.py                # 主程序入口
├── requirements.txt       # Python依赖包列表
├── dataset.yaml           # YOLO数据集配置文件
├── .gitignore             # Git忽略文件配置
├── dataset/               # 自动生成的数据集目录
│   ├── images/            # 图片目录
│   │   ├── train/         # 训练集图片 
│   │   └── val/           # 验证集图片
│   ├── labels/            # 标签目录
│   │   ├── train/         # 训练集标签
│   │   └── val/           # 验证集标签
├── README.md              # 项目说明文档


```

---

## 数据集结构说明

- `001-500` 和 `501-1100` 是原始数据集存放位置。
- 下载链接在：[下载链接]()
- `dataset` 和 `runs` 是自动生成的文件夹。

---

## 环境依赖

python == 3.10

建议使用conda或venv创建虚拟环境。

```bash
conda create -n YOLO python=3.10
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 使用方法

### 1. 数据准备

- 按照上述结构准备好图片和标签文件。
- 修改 `dataset.yaml`中的路径和类别信息以适配你的数据集。

### 2. 训练模型

```bash
python main.py
```

- 程序会自动检查数据集和配置文件，准备好后开始训练。

### 3. 测试与推理

- 训练完成后，模型权重会保存在 `runs/train/exp*/weights/`目录下。
- 可根据YOLOv8官方文档进行推理或评估。

---

## 参考

- [YOLOv8官方文档](https://docs.ultralytics.com/)
- 微流控与数字微流控相关文献

---

如有问题欢迎提issue或联系作者。
