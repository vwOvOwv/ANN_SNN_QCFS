from cv2 import mean
from sympy import print_rcode
import torch
import torch.nn as nn

# 定义一个类用于合并时间维度
class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        # 将输入张量展平成连续的形状，合并时间维度
        return x_seq.flatten(0, 1).contiguous()

# 定义一个类用于扩展时间维度
class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        # 计算新形状
        y_shape = [self.T, int(x_seq.shape[0] / self.T)]
        y_shape.extend(x_seq.shape[1:])
        # 重新调整输入张量的形状
        return x_seq.view(y_shape)

# 定义自定义的激活函数 ZIF
class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        # 正向传播逻辑：如果输入大于等于0，则输出1，否则输出0
        out = (input >= 0).float()
        # 保存上下文
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播逻辑
        input, out, others = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        # 计算梯度
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

# 定义自定义的 floor 函数
class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

# 使用自定义的 floor 函数
myfloor = GradFloor.apply

# 定义 IF 类
class IF(nn.Module):
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0): # 默认T为0，等同于ANN的neuron！
        super(IF, self).__init__()
        self.act = ZIF.apply  # 使用自定义的激活函数 ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)  # 设置膜电位阈值为可训练参数
        self.tau = tau  # 时间常数（此处未使用）
        self.gama = gama  # 用于激活函数的参数 gama

        # 扩展和合并时间维度的操作
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)

        # 量化级别 L 和时间步长 T
        self.L = L
        self.T = T

        self.loss = 0  # 损失变量（未使用）

    def forward(self, x):
        if self.T > 0:  # 如果 T > 0 （测试SNN），处理时间序列数据
            thre = self.thresh.data
            x = self.expand(x)  # 扩展输入的时间维度
            mem = 0.5 * thre  # 初始化膜电位为阈值的一半
            spike_pot = []  # 创建一个列表保存每个时间步的脉冲电位
            for t in range(self.T):  # 循环处理每个时间步
                mem = mem + x[t, ...]  # 累积当前时间步的输入到膜电位
                spike = self.act(mem - thre, self.gama) * thre  # 计算脉冲，如果膜电位超过阈值，则发放脉冲
                mem = mem - spike  # 更新膜电位，减去发放的脉冲
                spike_pot.append(spike)  # 将当前时间步的脉冲添加到列表
            x = torch.stack(spike_pot, dim=0)  # 将所有时间步的脉冲堆叠成一个张量
            x = self.merge(x)  # 合并时间维度
        else:  # 如果 T <= 0 （训练ANN），处理静态输入数据，下面4行是本篇文章的核心
            x = x / self.thresh  # 归一化输入
            x = torch.clamp(x, 0, 1)  # 限制输入值在0到1之间
            x = myfloor(x * self.L + 0.5) / self.L  # 对输入进行量化
            x = x * self.thresh  # 反归一化输入
        return x

# 定义一个函数用于增加维度
def add_dimention(x, T):
    x.unsqueeze_(1)  # 在第1维增加一个维度
    x = x.repeat(T, 1, 1, 1, 1)  # 重复 T 次
    return x
