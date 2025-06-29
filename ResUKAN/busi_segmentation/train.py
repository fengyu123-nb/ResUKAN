import argparse
import os
os.environ['ALBUMENTATIONS_DISABLE_UPDATE_CHECK'] = '1'
import random
import numpy as np
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from collections import OrderedDict
from torchvision.transforms import Compose, Resize
from tqdm import tqdm
import losses  # 导入损失函数模块
from dataset import BUSIDataset  # 导入自定义数据集类
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool
from torch.optim import AdamW
from tensorboardX import SummaryWriter
import shutil
from torch.optim import lr_scheduler
from albumentations import Resize
import albumentations as A
from albumentations.pytorch import ToTensorV2
# 定义可选的损失函数
LOSS_NAMES = ['DiceBCELoss', 'LovaszHingeLoss']


# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--dataseed', default=2981, type=int, help='random seed for dataset')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResUKAN')  # 使用 ResUKAN 模型
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=224, type=int, help='image width')
    parser.add_argument('--input_h', default=224, type=int, help='image height')

    # loss
    parser.add_argument('--loss', default='DiceBCELoss', choices=LOSS_NAMES,
                        help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: DiceBCELoss)')

    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')
    parser.add_argument('--data_dir', default='../busi_dataset', help='dataset directory')
    parser.add_argument('--output_dir', default='outputs_6', help='output directory')

    # optimizer
    parser.add_argument('--optimizer', default='AdamW', choices=['Adam', 'SGD', 'AdamW'],
                        help='optimizer: Adam | SGD | AdamW (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()
    return config


# 设置随机种子
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 训练函数
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:

        input = input.cuda()
        target = target.cuda()

        if config['deep_supervision']:
            outputs = model(input)
            loss = sum(criterion(output, target) for output in outputs) / len(outputs)
            iou, dice, _ = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg)])


# 验证函数
def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)
                loss = sum(criterion(output, target) for output in outputs) / len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict(
                [('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg), ('dice', avg_meters['dice'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict(
        [('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg), ('dice', avg_meters['dice'].avg)])


# 主函数
def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        config['name'] = f"{config['dataset']}_{config['arch']}_withDS" if config[
            'deep_supervision'] else f"{config['dataset']}_{config['arch']}_withoutDS"

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    # with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
    #     yaml.dump(config, f)

    # 定义损失函数
    loss_fn = getattr(losses, config['loss'])()  # 动态加载所选损失函数

    cudnn.benchmark = True

    # 根据配置选择模型
    from model import ResUKAN  # 导入 ResUKAN 模型
    model = ResUKAN(config['num_classes'], [256, 320, 512])
    model = model.cuda()

    # 设置优化器
    param_groups = [{'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']} for name, param in model.named_parameters()]

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':  # 新增 AdamW 选项
        optimizer = optim.AdamW(param_groups, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # 设置学习率调度器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr']) if config[
                                                                                                                   'scheduler'] == 'CosineAnnealingLR' else None

    # shutil.copy2('train.py', f'{output_dir}/{exp_name}/train.py')

    train_transform = A.Compose([
        Resize(config['input_w'], config['input_h']),  # 调整图像和掩码的尺寸
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.3),
        # A.RandomScale(scale_limit=(0.8, 1.2), p=0.5),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()  # 自动将 image 和 mask 转换为 PyTorch 张量
    ])

    val_transform = A.Compose([
        Resize(config['input_w'], config['input_h']),  # 调整图像和掩码的尺寸
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()  # 自动将 image 和 mask 转换为 PyTorch 张量
    ])

    # 加载数据集


    # 创建数据集
    train_dataset = BUSIDataset(
        img_dir=os.path.join(config['data_dir'], 'Traindataset/images'),
        mask_dir=os.path.join(config['data_dir'], 'Traindataset/masks'),
        transform=train_transform
    )

    val_dataset = BUSIDataset(
        img_dir=os.path.join(config['data_dir'], 'Testdataset/images'),
        mask_dir=os.path.join(config['data_dir'], 'Testdataset/masks'),
        transform=val_transform
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True
    )

    best_iou = 0
    # trigger = 0  # 记录早停的触发次数

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}/{config['epochs']}")

        # 训练阶段
        train_log = train(config, train_loader, model, loss_fn, optimizer)
        val_log = validate(config, val_loader, model, loss_fn)

        # 记录日志
        my_writer.add_scalar('train/loss', train_log['loss'], epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], epoch)

        # 保存最佳模型
        is_best = val_log['iou'] > best_iou
        best_iou = max(best_iou, val_log['iou'])

        if is_best:
            torch.save(model.state_dict(), f"{output_dir}/{exp_name}/best_model.pth")
            print(f"Saved Best Model at epoch {epoch + 1}")


        # 更新学习率
        if scheduler is not None:
            scheduler.step()

    my_writer.close()
    print(f"Training completed! Best IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()