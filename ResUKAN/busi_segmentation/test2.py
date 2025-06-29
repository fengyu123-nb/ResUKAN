import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from model import ResUKAN
from dataset import test_result_dataset  # 假设修改后的类保存在 dataset.py 中

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=224, help='testing size')
    parser.add_argument('--pth_path', type=str, default=r'S:\ResUKAN\BUSI_segmentation\outputs_1\None\best_model.pth')   # 将训练好的权重PTH文件加载进来。
    opt = parser.parse_args()

    # 初始化模型
    model = ResUKAN()   # model换成你的UNet
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()   # 你的是cpu，这里就不用cuda
    model.eval()

    for _data_name in ['1']:  # 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB'       这个是要测试的数据集，可以只写一个，也可以写多个。

        ##### put data_path here #####
        data_path = '../busi_dataset/TestDataset/{}'.format(_data_name)   #  测试数据集路径。
        ##### save_path #####
        save_path = '../ooooutput/ResUKAN/{}/'.format(_data_name)   #  保存路径

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)

        # 初始化数据集
        test_loader = test_result_dataset(image_root, opt.testsize)

        for i in range(test_loader.size):
            # 加载图像、名称和原始大小
            image, name, original_size = test_loader.load_data()

            # 将图像移动到 GPU
            image = image.cuda()

            # 模型推理
            output = model(image)

            # 将输出调整回原始图像大小
            res = F.interpolate(output, size=original_size[::-1], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            # 保存结果
            cv2.imwrite(os.path.join(save_path, name), res * 255)

        print(_data_name, 'Finish!')