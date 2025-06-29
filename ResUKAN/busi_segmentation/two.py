import cv2
import os
import numpy as np


def binarize_images(input_folder, output_folder, threshold=127):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建完整的文件路径
        file_path = os.path.join(input_folder, filename)

        # 检查是否为图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            # 读取图片
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # 对图片进行二值化处理
            _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

            # 构建输出文件路径
            output_file_path = os.path.join(output_folder, filename)

            # 保存二值化后的图片
            cv2.imwrite(output_file_path, binary_image)
            print(f"Processed and saved: {output_file_path}")


# 使用示例
input_folder = r"S:\ResUKAN\ooooutput\ResUKAN\1"  # 替换为你的输入文件夹路径
output_folder = r"S:\ResUKAN\ooooutput\ResUKAN\2"  # 替换为你的输出文件夹路径

binarize_images(input_folder, output_folder)