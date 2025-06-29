
"""
import torch
from torchsummary import summary
from model import ResUKAN  # 假设模型文件为 polyp_segmentation.py
from thop import profile

# 初始化模型
model = ResUKAN()

# 确保模型和输入都在 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = torch.randn(1, 3, 224, 224).to(device)
# 定义输入尺寸


flops, params = profile(model, inputs=(input_tensor, ))

# 输出为 Gflops 和参数数量（单位为百万）
print(f"Gflops: {flops / 1e9:.3f}")
print(f"Params: {params / 1e6:.3f} M")

# 打印模型的参数数量和理论大小
total_params = sum(p.numel() for p in model.parameters())
print(f"参数数量: {total_params:,} ({total_params / 1e6:.2f} M)")
print(f"模型理论大小: {total_params * 4 / (1024 ** 2):.2f} MB")  # 每个参数占 4 字节

# 调用 summary
# summary(model, input_size=input_size)
"""
# 安装库：pip install thop
import torch
from thop import profile
from model import ResUKAN
model = ResUKAN()  # 实例化模型
input = torch.randn(1, 3, 224, 224)  # 输入尺寸需与模型匹配
flops, params = profile(model, inputs=(input,))
print(f"Gflops: {flops / 1e9:.4f}, Params: {params / 1e6:.4f}M")
