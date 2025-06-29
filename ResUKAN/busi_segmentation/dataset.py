import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch  # 导入 torch
import torchvision.transforms as transforms
from PIL import Image
class BUSIDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # 获取所有图像文件（假设 mask 具有相同文件名）
        self.img_ids = sorted(f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg')))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        mask_path = os.path.join(self.mask_dir, img_id)

        # 读取图片和 mask
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 彩色图片
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 单通道 mask

        if image is None or mask is None:
            raise RuntimeError(f"无法加载: {img_path} 或 {mask_path}")

        mask = (mask > 0).astype(np.float32)  # 归一化到 0 和 1

        # 进行数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # 确保 mask 是 PyTorch 张量
        if isinstance(mask, np.ndarray):  # 如果 mask 是 NumPy 数组
            mask = torch.from_numpy(mask)  # 转换为 PyTorch 张量

        # 确保 mask 的形状是 [1, H, W]
        if mask.dim() == 2:  # 如果 mask 是 [H, W]
            mask = mask.unsqueeze(0)  # 增加通道维度 -> [1, H, W]

        return image, mask

import os
from PIL import Image
import torchvision.transforms as transforms
import torch

class test_result_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)
                       if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG')]
        self.images = sorted(self.images)

        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),  # 调整为指定大小
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        # 加载原始图像
        image = self.rgb_loader(self.images[self.index])
        original_size = image.size  # 记录原始图像大小 (width, height)

        # 对图像进行预处理
        image = self.transform(image).unsqueeze(0)

        # 获取图像名称
        name = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1

        # 返回预处理后的图像、名称和原始大小
        return image, name, original_size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
# 示例：数据增强
def get_transforms(input_w, input_h):
    return A.Compose([
        A.Resize(input_w, input_h),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()  # 自动将 image 和 mask 转换为 PyTorch 张量
    ])


# 训练和验证数据集加载
if __name__ == "__main__":
    config = {
        'data_dir': '../busi_dataset',
        'input_w': 224,
        'input_h': 224
    }

    train_dataset = BUSIDataset(
        os.path.join(config['data_dir'], 'Traindataset/images'),
        os.path.join(config['data_dir'], 'Traindataset/masks'),
        transform=get_transforms(config['input_w'], config['input_h'])
    )

    val_dataset = BUSIDataset(
        os.path.join(config['data_dir'], 'Testdataset/images'),
        os.path.join(config['data_dir'], 'Testdataset/masks'),
        transform=get_transforms(config['input_w'], config['input_h'])
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    for images, masks in val_loader:
        print("Image shape:", images.shape)  # [B, 3, H, W]
        print("Mask shape:", masks.shape)  # [B, 1, H, W]
        break