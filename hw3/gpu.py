import torch
import time

# 配置
device = torch.device("cuda:0")
# 张量尺寸，调大占用更高，调小更低
size = 16384
# 循环间隔，控制负载密度
sleep_sec = 0.002

# 分配常驻显存大矩阵
a = torch.randn(size, size, device=device, dtype=torch.float32)
b = torch.randn(size, size, device=device, dtype=torch.float32)
c = torch.zeros_like(a)

print(f"Start GPU load test, size={size}, device={device}")
print("Check nvidia-smi for GPU utilization ~80%")

while True:
    # 持续矩阵乘，占用GPU算力
    c = torch.matmul(a, b)
    # 同步等待计算完成
    torch.cuda.synchronize(device)
    # 短暂休眠降低平均占用到80%左右
    time.sleep(sleep_sec)