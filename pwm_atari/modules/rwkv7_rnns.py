# 文件路径: pwm_atari/modules/rwkv7_rnns.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.rwkv7_layer import RWKV7Layer # 引用你提供的封装

class RWKV7Cell(nn.Module):
    def __init__(self, inp_size, hidden, act, head_size=64, verbose=False):
        super().__init__()
        self.verbose = verbose
        # 适配 PaMoRL 的维度需求，RWKV v7 作为核心算子
        self.rwkv = RWKV7Layer(dim=hidden, head_size=head_size, verbose=verbose)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),
            act(),
            nn.LayerNorm(hidden)
        )
        self.norm = nn.LayerNorm(hidden)
        
    @torch.no_grad()
    def initial(self, batch_size, layer_id):
        # 初始化状态：对于 RWKV v7，状态在并行模式下由 Kernel 自动管理
        # 推理模式下需要缓存上一个时刻的特征
        # return torch.zeros(batch_size, self.rwkv.dim, device=device)
        # 修正：必须返回字典，键名格式需与 PSSM 逻辑一致
        return {f"rnn_{layer_id}": torch.zeros(batch_size, self.rwkv.dim)}
    
    def forward(self, inp, is_first, state, parallel, layer_id):
        # inp: [B, T, D] if parallel else [B, D]
        if parallel:
            # 并行模式：直接利用 RWKV v7 的 Fused Kernel 加速
            output, _ = self.rwkv(inp) 
            output = self.norm(inp + output)
            output = self.ffn(output)
            # return output, inp[:, -1], None # 返回最后时刻特征作为状态
            # 返回输出和该层更新后的状态字典
            return output, {f"rnn_{layer_id}": output[-1]} # 取序列最后一个 step
        else:
            # 循环推理模式 (用于 Imagination)
            # 此时 T=1，inp 为 [B, D] -> [B, 1, D]
            output, _ = self.rwkv(inp.unsqueeze(1))
            output = output.squeeze(1)
            output = self.norm(inp + output)
            output = self.ffn(output)
            # return output, output
            return output, {f"rnn_{layer_id}": output}