import os
import cv2
import numpy as np
from sklearn.metrics import f1_score

def load_mask(image_path):
    """加载并二值化掩码图像"""
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
    mask = (mask > 127).astype(np.uint8)  # 设定阈值，将大于127的归为1，其他为0
    return mask

def compute_metrics(pred_mask, gt_mask):
    """计算 Dice, IoU, Sensitivity 和 F1-score"""
    TP = np.sum(np.logical_and(pred_mask, gt_mask))  # 真阳性
    FP = np.sum(pred_mask) - TP  # 假阳性
    FN = np.sum(gt_mask) - TP  # 假阴性

    # 计算 Dice
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    # 计算 IoU
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # 计算 Sensitivity（召回率）
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算 F1-score
    f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=1)

    return dice, iou, sensitivity, f1

def evaluate_masks(pred_folder, gt_folder):
    """遍历文件夹计算所有 mask 的评估指标"""
    pred_files = sorted(os.listdir(pred_folder))  # 获取预测文件列表
    gt_files = sorted(os.listdir(gt_folder))  # 获取 GT 文件列表

    dice_scores, iou_scores, sensitivity_scores, f1_scores = [], [], [], []

    for filename in pred_files:
        pred_path = os.path.join(pred_folder, filename)
        gt_path = os.path.join(gt_folder, filename)

        if not os.path.exists(gt_path):  # 确保 GT 存在
            print(f"Warning: {filename} 在 GT 文件夹中找不到，跳过。")
            continue

        # 读取 mask
        pred_mask = load_mask(pred_path)

        gt_mask = load_mask(gt_path)

        # 计算指标
        dice, iou, sensitivity, f1 = compute_metrics(pred_mask, gt_mask)

        # 记录指标
        dice_scores.append(dice)
        iou_scores.append(iou)
        sensitivity_scores.append(sensitivity)
        f1_scores.append(f1)

        print(f"{filename}: Dice={dice:.4f}, IoU={iou:.4f}, Sensitivity={sensitivity:.4f}, F1={f1:.4f}")

    # 计算所有图片的平均指标
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_sensitivity = np.mean(sensitivity_scores)
    avg_f1 = np.mean(f1_scores)

    print("\n=== 平均评估指标 ===")
    print(f"平均 Dice: {avg_dice:.4f}")
    print(f"平均 IoU: {avg_iou:.4f}")
    print(f"平均 Sensitivity: {avg_sensitivity:.4f}")
    print(f"平均 F1-score: {avg_f1:.4f}")

# 文件夹路径
pred_folder = r"S:\ResUKAN\ooooutput\ResUKAN\1"  # 预测 mask 文件夹
gt_folder = r"S:\ResUKAN\busi_dataset\Testdataset\1\masks"  # Ground Truth mask 文件夹

evaluate_masks(pred_folder, gt_folder)
