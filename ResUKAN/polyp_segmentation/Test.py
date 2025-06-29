import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from model import ResUKAN
from utils.dataloader import test_result_dataset
import cv2

# 此代码是将所有测试集进行预测可视化，并进行保存。

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=224, help='testing size')
    parser.add_argument('--pth_path', type=str, default='../model_Kvasir_Linear_2.pth/RESUKAN/88RESUKAN.pth')   # 将训练好的权重PTH文件加载进来。
    opt = parser.parse_args()
    model = ResUKAN()   # model换成你的UNet
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()   # 你的是cpu，这里就不用cuda
    model.eval()
    for _data_name in ['CVC-ColonDB']:  # 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB'       这个是要测试的数据集，可以只写一个，也可以写多个。

        ##### put data_path here #####
        data_path = '../polyp_dataset/TestDataset/{}'.format(_data_name)   #  测试数据集路径。
        ##### save_path #####
        save_path = '../last_linear_result_map/KANNet/{}/'.format(_data_name)   #  保存路径

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)

        num1 = len(os.listdir(image_root))

        test_loader = test_result_dataset(image_root, 224)

        for i in range(num1):
            image, name = test_loader.load_data()

            image = image.cuda()
            myshape = (image.shape[2], image.shape[3])
            output = model(image)
            res = F.upsample(output, size=myshape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        print(_data_name, 'Finish!')
