import os
import json
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import traceback  
from ultralytics import YOLO
from tqdm import tqdm
import gc
import psutil  # 用于系统资源监控



def tile_image(image, tile_size=1024, overlap=128):
    """将大图切割成重叠的小块

    Args:
        image: 输入图像
        tile_size: 切片大小
        overlap: 重叠区域大小

    Returns:
        tiles: 切片列表
        positions: 每个切片在原图中的位置 (x, y)
    """
    height, width = image.shape[:2]
    tiles = []
    positions = []

    stride = tile_size - overlap

    for y in range(0, height - overlap, stride):
        for x in range(0, width - overlap, stride):
            # 确保最后一块能覆盖到边缘
            end_x = min(x + tile_size, width)
            end_y = min(y + tile_size, height)
            start_x = max(0, end_x - tile_size)
            start_y = max(0, end_y - tile_size)

            tile = image[start_y:end_y, start_x:end_x]
            tiles.append(tile)
            positions.append((start_x, start_y))

    return tiles, positions


def convert_json_to_yolo_tiles(
    json_path, img_width, img_height, tile_size=1024, overlap=128
):
    """将JSON格式的标注转换为YOLO格式，并根据图像切片调整标注"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stride = tile_size - overlap
    tiles_annotations = {}

    # 计算需要的切片数量
    n_tiles_x = (img_width + stride - 1) // stride
    n_tiles_y = (img_height + stride - 1) // stride

    # 初始化每个切片的标注列表
    for y in range(n_tiles_y):
        for x in range(n_tiles_x):
            tiles_annotations[f"{x}_{y}"] = []

    for shape in data["shapes"]:
        if shape["shape_type"] == "rectangle":
            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 对于每个边界框，找到它属于哪些切片
            min_tile_x = int(min(x1, x2)) // stride
            max_tile_x = int(max(x1, x2)) // stride
            min_tile_y = int(min(y1, y2)) // stride
            max_tile_y = int(max(y1, y2)) // stride

            # 为每个相关的切片创建标注
            for tile_y in range(min_tile_y, max_tile_y + 1):
                for tile_x in range(min_tile_x, max_tile_x + 1):
                    if f"{tile_x}_{tile_y}" not in tiles_annotations:
                        continue

                    # 计算切片的边界
                    tile_start_x = tile_x * stride
                    tile_start_y = tile_y * stride
                    tile_end_x = min(tile_start_x + tile_size, img_width)
                    tile_end_y = min(tile_start_y + tile_size, img_height)
                    tile_start_x = max(0, tile_end_x - tile_size)
                    tile_start_y = max(0, tile_end_y - tile_size)

                    # 调整边界框坐标到切片坐标系
                    box_x1 = max(0, min(x1 - tile_start_x, tile_size))
                    box_y1 = max(0, min(y1 - tile_start_y, tile_size))
                    box_x2 = max(0, min(x2 - tile_start_x, tile_size))
                    box_y2 = max(0, min(y2 - tile_start_y, tile_size))

                    # 如果边界框在切片内
                    if (
                        box_x2 > 0
                        and box_y2 > 0
                        and box_x1 < tile_size
                        and box_y1 < tile_size
                    ):
                        # 计算YOLO格式的标注
                        x_center = (box_x1 + box_x2) / (2 * tile_size)
                        y_center = (box_y1 + box_y2) / (2 * tile_size)
                        width = abs(box_x2 - box_x1) / tile_size
                        height = abs(box_y2 - box_y1) / tile_size

                        # 确保值在0-1范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        yolo_line = f"0 {x_center} {y_center} {width} {height}"
                        tiles_annotations[f"{tile_x}_{tile_y}"].append(yolo_line)

    return tiles_annotations


def get_image_number(filename):
    """从文件名中提取图片编号"""
    return int(filename.stem[1:])  # 去掉'A'前缀，转换为数字


def clean_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def monitor_resources():
    """监控系统资源使用情况"""
    try:
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU内存 - 已使用: {gpu_memory_used:.1f}GB / 总计: {gpu_memory_total:.1f}GB / 缓存: {gpu_memory_cached:.1f}GB")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"CPU使用率: {cpu_percent:.1f}% | 系统内存使用: {memory.percent:.1f}%")
    except Exception as e:
        print(f"资源监控出错: {e}")


def process_single_image(args):
    """处理单张图片的切片和标注"""
    img_path, is_train, tile_size, overlap = args
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return []

        height, width = img.shape[:2]
        tiles, positions = tile_image(img, tile_size, overlap)

        json_path = str(img_path.with_suffix(".json"))
        if not os.path.exists(json_path):
            print(f"警告：找不到对应的标注文件 {json_path}")
            return []

        tiles_annotations = convert_json_to_yolo_tiles(
            json_path, width, height, tile_size, overlap
        )
        results = []

        for i, (tile, pos) in enumerate(zip(tiles, positions)):
            tile_x, tile_y = pos[0] // (tile_size - overlap), pos[1] // (
                tile_size - overlap
            )
            tile_name = f"{img_path.stem}_tile_{tile_x}_{tile_y}"

            split_type = "train" if is_train else "val"
            img_save_path = f"dataset/images/{split_type}/{tile_name}.png"
            label_save_path = f"dataset/labels/{split_type}/{tile_name}.txt"

            annotations = tiles_annotations.get(f"{tile_x}_{tile_y}", [])
            if annotations:
                results.append((tile, img_save_path, annotations, label_save_path))

        # 清理内存
        del img, tiles, positions, tiles_annotations
        gc.collect()

        return results

    except Exception as e:
        print(f"处理文件 {img_path} 时出错: {e}")
        return []


def prepare_dataset(tile_size=640, overlap=128):
    """准备数据集，将数据分割为训练集和验证集，并对图像进行切片处理"""
    random.seed(42)
    np.random.seed(42)

    image_files = []
    folders = ["001-500", "501-1100"]

    for folder in folders:
        folder_path = Path(folder)
        if folder_path.exists():
            for file in folder_path.glob("*.png"):
                image_files.append(file)

    if not image_files:
        raise Exception("没有找到任何PNG图片文件！请检查数据集路径。")

    image_files = sorted(image_files, key=get_image_number)
    total_images = len(image_files)

    val_size = 220
    step = total_images // val_size

    val_indices = set(range(0, total_images, step))
    if len(val_indices) > val_size:
        val_indices = set(random.sample(list(val_indices), val_size))
    elif len(val_indices) < val_size:
        remaining_indices = set(range(total_images)) - val_indices
        additional_indices = random.sample(
            list(remaining_indices), val_size - len(val_indices)
        )
        val_indices.update(additional_indices)

    train_indices = set(range(total_images)) - val_indices
    train_files = [image_files[i] for i in sorted(train_indices)]
    val_files = [image_files[i] for i in sorted(val_indices)]

    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)

    def process_files(files, is_train):
        """单线程处理文件"""
        for img_path in tqdm(files, desc="处理图片"):
            results = process_single_image((img_path, is_train, tile_size, overlap))
            if results:
                for tile, img_save_path, annotations, label_save_path in results:
                    cv2.imwrite(img_save_path, tile)
                    with open(label_save_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(annotations))

            # 清理内存
            gc.collect()

    print("\n处理训练集...")
    process_files(train_files, True)

    print("\n处理验证集...")
    process_files(val_files, False)

    print("\n数据集准备完成！")


def create_dataset_yaml():
    """创建数据集配置文件"""
    yaml_content = """
path: dataset  # 数据集根目录
train: images/train  # 训练图片相对路径
val: images/val  # 验证图片相对路径

# 类别数和名称
nc: 1  # 类别数
names: ['droplet']  # 类别名称
    """

    with open("dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())
    print("数据集配置文件创建完成！")


def check_dataset_ready():
    """检查数据集是否已经准备好"""
    required_dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
    ]

    # 检查目录是否存在
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            return False

    # 检查文件数量
    train_images = len(list(Path("dataset/images/train").glob("*.png")))
    val_images = len(list(Path("dataset/images/val").glob("*.png")))
    train_labels = len(list(Path("dataset/labels/train").glob("*.txt")))
    val_labels = len(list(Path("dataset/labels/val").glob("*.txt")))

    if (
        train_images == 880
        and val_images == 220
        and train_labels == train_images
        and val_labels == val_images
    ):
        print(f"\n检测到已存在的数据集:")
        print(f"训练集: {train_images}张图片和标签")
        print(f"验证集: {val_images}张图片和标签")
        return True

    return False


def train_model():
    """训练模型"""
    model_path = "yolov8m.pt"

    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n使用GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.1f}GB")

        # 清理GPU内存
        clean_gpu_memory()
    else:
        print("\n警告：未检测到GPU，将使用CPU训练（训练速度会很慢）")

    try:
        if os.path.exists(model_path):
            print(f"使用本地模型: {model_path}")
            model = YOLO(model_path)
        else:
            print("本地未找到模型文件，将从官方下载...")
            model = YOLO("yolov8m.pt")

        # 安全地添加CBAM模块
        try:
            from models.edge_attention import CBAM
            
            # 获取检测头
            detect_layer = model.model.model[-1]
            
            print("\n正在添加CBAM模块到检测头...")
            
            # 检查检测头结构是否符合预期
            if hasattr(detect_layer, 'cv2') and isinstance(detect_layer.cv2, (list, nn.ModuleList)):
                print(f"检测到 {len(detect_layer.cv2)} 个检测头分支")
                
                # 安全地遍历所有检测头分支并添加CBAM
                for i in range(len(detect_layer.cv2)):
                    try:
                        # 检查当前分支的结构
                        current_branch = detect_layer.cv2[i]
                        if isinstance(current_branch, nn.Sequential) and len(current_branch) >= 3:
                            # 获取第一个卷积层的输入通道数
                            first_conv = current_branch[0]
                            if hasattr(first_conv, 'conv'):
                                in_channels = first_conv.conv.in_channels
                            elif hasattr(first_conv, 'in_channels'):
                                in_channels = first_conv.in_channels
                            else:
                                print(f"警告：无法确定第 {i} 个分支的输入通道数，跳过")
                                continue
                            
                            # 创建CBAM模块
                            cbam_module = CBAM(in_channels=in_channels)
                            
                            # 构建新的序列，将CBAM插入到开头
                            new_layers = [cbam_module] + list(current_branch)
                            new_seq = nn.Sequential(*new_layers)
                            
                            # 替换原有分支
                            detect_layer.cv2[i] = new_seq
                            print(f"成功为第 {i} 个分支添加CBAM (输入通道: {in_channels})")
                            
                        else:
                            print(f"警告：第 {i} 个分支结构不符合预期，跳过CBAM添加")
                            
                    except Exception as branch_e:
                        print(f"为第 {i} 个分支添加CBAM时出错: {str(branch_e)}")
                        continue
                
                print("CBAM模块添加完成!")
                
                # 确保模型在正确的设备上
                if device != "cpu":
                    model.model = model.model.to(f"cuda:{device}")
                    
            else:
                print("警告：检测头结构不符合预期，无法添加CBAM模块")
                
        except ImportError as ie:
            print(f"警告：无法导入CBAM模块: {str(ie)}")
            print("将使用原始YOLOv8模型进行训练")
        except Exception as cbam_e:
            print(f"添加CBAM模块时出现错误: {str(cbam_e)}")
            print("将使用原始YOLOv8模型进行训练")
            # 重新加载原始模型以确保干净状态
            model = YOLO(model_path)
            
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        print(traceback.format_exc())
        return

    print("\n开始训练...")
    try:
        # 更保守的batch size计算
        if device == "0":
            # 为CBAM模块预留更多内存
            available_memory = gpu_memory * 0.7  # 使用70%的GPU内存
            suggested_batch_size = min(8, max(1, int(available_memory / 2)))  # 更保守的估算
            batch_size = suggested_batch_size
        else:
            batch_size = 2  # CPU训练使用更小的batch size

        print(f"使用batch_size: {batch_size}")

        # 添加检查点恢复机制
        resume_path = "runs/train/exp6/weights/last.pt"
        resume = resume_path if os.path.exists(resume_path) else False
        if resume:
            print(f"检测到检查点文件，将从 {resume_path} 恢复训练")

        results = model.train(
            data="dataset.yaml",
            epochs=100,
            imgsz=640,
            batch=batch_size,
            patience=20,
            augment=True,
            device=device,
            project="runs/train",
            name="exp6",  # 使用新的实验名称避免冲突
            save=True,
            save_period=5,  # 更频繁地保存检查点
            cache=False,
            amp=True,
            workers=0,
            exist_ok=True,
            pretrained=True,
            optimizer="auto",
            close_mosaic=10,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.2,
            label_smoothing=0.1,
            cos_lr=True,
            warmup_epochs=3,
            weight_decay=0.0005,
            # 确保生成评估曲线
            plots=True,
            save_json=True,
            val=True,
            resume=resume,  # 支持断点续训
        )

        print("\n训练完成！")
        print(f"最佳模型保存在: {os.path.join('runs/train/exp6/weights/best.pt')}")

        # 清理GPU内存
        clean_gpu_memory()

    except KeyboardInterrupt:
        print("\n训练被用户中断！")
        print("可以通过设置 resume=True 从最后的检查点继续训练")
        clean_gpu_memory()
    except RuntimeError as re:
        if "out of memory" in str(re).lower():
            print(f"\nGPU内存不足错误: {str(re)}")
            print("建议降低batch_size或使用更小的模型")
        else:
            print(f"运行时错误: {str(re)}")
        clean_gpu_memory()
        raise re
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        print(traceback.format_exc())
        clean_gpu_memory()
        raise e


def merge_predictions(predictions, original_shape, tile_size=640, overlap=128):
    """合并切片的预测结果"""
    height, width = original_shape[:2]
    merged_boxes = []
    merged_scores = []
    merged_classes = []

    # 使用非极大值抑制合并重叠框
    for pred in predictions:
        if len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy()

            merged_boxes.extend(boxes)
            merged_scores.extend(scores)
            merged_classes.extend(classes)

    if merged_boxes:
        merged_boxes = np.array(merged_boxes)
        merged_scores = np.array(merged_scores)
        merged_classes = np.array(merged_classes)

        # 执行NMS
        indices = cv2.dnn.NMSBoxes(
            merged_boxes.tolist(),
            merged_scores.tolist(),
            score_threshold=0.25,
            nms_threshold=0.45,
        )

        if len(indices) > 0:
            if isinstance(indices, tuple):  # OpenCV 3.x returns tuple
                indices = indices[0]

            return (
                merged_boxes[indices],
                merged_scores[indices],
                merged_classes[indices],
            )

    return [], [], []


def process_single_prediction(args):
    """处理单个切片的预测

    Args:
        args: (model, tile, pos) 元组
    """
    model, tile, pos = args
    pred = model.predict(tile, conf=0.25)
    if len(pred) > 0 and len(pred[0].boxes) > 0:
        # 调整预测框的坐标到原图坐标系
        for box in pred[0].boxes:
            box.xyxy[:, [0, 2]] += pos[0]  # 调整x坐标
            box.xyxy[:, [1, 3]] += pos[1]  # 调整y坐标
        return pred[0]
    return None


def test_model():
    """测试模型并可视化结果"""
    model = YOLO("runs/train/exp5/weights/best.pt")

    os.makedirs("runs/detect/results", exist_ok=True)

    val_images = list(Path("dataset/images/val").glob("*.png"))
    print(f"开始在验证集图片上进行测试...")

    # 获取原始验证集图片
    original_val_images = set()
    for img_path in val_images:
        original_name = img_path.stem.split("_tile_")[0]
        original_val_images.add(original_name)

    for original_name in tqdm(original_val_images):
        # 清理GPU内存
        clean_gpu_memory()

        original_img_path = None
        for folder in ["001-500", "501-1100"]:
            test_path = Path(folder) / f"{original_name}.png"
            if test_path.exists():
                original_img_path = test_path
                break

        if original_img_path is None:
            continue

        img = cv2.imread(str(original_img_path))
        if img is None:
            continue

        height, width = img.shape[:2]
        tiles, positions = tile_image(img, tile_size=640, overlap=128)

        # 单线程处理预测
        all_predictions = []
        for tile, pos in tqdm(
            zip(tiles, positions), desc=f"预测 {original_name}", leave=False
        ):
            pred = model.predict(tile, conf=0.25)
            if len(pred) > 0 and len(pred[0].boxes) > 0:
                # 调整预测框的坐标到原图坐标系
                for box in pred[0].boxes:
                    box.xyxy[:, [0, 2]] += pos[0]
                    box.xyxy[:, [1, 3]] += pos[1]
                all_predictions.append(pred[0])

            # 每次预测后清理GPU内存
            clean_gpu_memory()

        # 合并预测结果
        merged_boxes, merged_scores, merged_classes = merge_predictions(
            all_predictions, img.shape, tile_size=640, overlap=128
        )

        # 在原图上绘制结果
        result_img = img.copy()
        for box, score, cls in zip(merged_boxes, merged_scores, merged_classes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                result_img,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # 保存结果
        cv2.imwrite(f"runs/detect/results/{original_name}_result.png", result_img)

        # 清理内存
        del img, result_img, tiles, positions, all_predictions
        gc.collect()


def main():
    """主函数"""
    if not check_dataset_ready():
        print("\n数据集未准备，开始处理...")
        prepare_dataset()
    else:
        print("\n检测到已存在的数据集，跳过处理步骤...")

    if not os.path.exists("dataset.yaml"):
        print("\n创建数据集配置文件...")
        create_dataset_yaml()
    else:
        print("\n检测到已存在的配置文件...")

    print("\n准备开始训练...")
    train_model()


if __name__ == "__main__":
    main()