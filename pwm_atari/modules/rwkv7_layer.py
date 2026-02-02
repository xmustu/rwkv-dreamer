import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

# =======================================================
# 1. 自动加载 CUDA Kernel
# =======================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_dir = os.path.join(current_dir, 'read_rwkv_v7', 'cuda')

cpp_src = os.path.join(cuda_dir, "wkv7_op.cpp")
cu_src = os.path.join(cuda_dir, "wkv7_cuda.cu")

HEAD_SIZE = 64
CHUNK_LEN = 16

if not os.path.exists(cpp_src) or not os.path.exists(cu_src):
    print(f"[Error] 找不到 CUDA 源码文件，路径: {cuda_dir}")
    wkv7_cuda = None
else:
    try:
        # 使用 wkv7_cuda_v7 确保重新编译
        wkv7_cuda = load(
            name="wkv7_cuda_v7", 
            sources=[cpp_src, cu_src],
            verbose=False, 
            extra_cuda_cflags=[
                f"-D_C_={HEAD_SIZE}",          
                f"-D_CHUNK_LEN_={CHUNK_LEN}",  
                "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"
            ]
        )
    except Exception as e:
        print(f"[Error] Failed to load RWKV v7 CUDA kernel: {e}")
        wkv7_cuda = None

# =======================================================
# 2. RWKV v7 TimeMix Layer
# =======================================================
class RWKV7Layer(nn.Module):
    def __init__(self, dim, head_size=64, verbose=False):
        super().__init__()
        assert head_size == HEAD_SIZE
        assert dim % head_size == 0
        self.verbose = verbose
        self.dim = dim
        self.head_size = head_size
        self.n_head = dim // head_size
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.decay = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.a_proj = nn.Linear(dim, dim, bias=False)
        
        self.output = nn.Linear(dim, dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, dim, eps=1e-5)

    def forward(self, x, state=None):
        # x: [Batch, Time, Dim]
        B, T, C = x.size()
        H = self.n_head
        
        # 1. Time Shift 
        xx = self.time_shift(x) - x
        x_in = x + xx * 0.5 
        
        # 2. Project
        r = self.receptance(x_in)
        k = self.key(x_in)
        v = self.value(x_in)
        g = torch.sigmoid(self.gate(x_in))
        
        w_raw = self.decay(x_in)
        w = -torch.exp(w_raw) 
        
        a = torch.sigmoid(self.a_proj(x_in))
        
        # 3. CUDA Kernel
        r = r.view(B, T, H, -1).float().contiguous()
        k = k.view(B, T, H, -1).float().contiguous()
        v = v.view(B, T, H, -1).float().contiguous()
        w = w.view(B, T, H, -1).float().contiguous()
        a = a.view(B, T, H, -1).float().contiguous()
        b = torch.ones_like(a).float().contiguous()
        
        # 必须确保输入到 CUDA 的是连续的 bfloat16 张量
        def to_kernel(t):
            return t.view(B, T, H, -1).to(dtype=torch.bfloat16).contiguous()
        
        rk, kk, vk, wk, ak = map(to_kernel, [r, k, v, w, a])
        bk = torch.ones_like(ak)
        yk = torch.empty_like(vk)
        
        y = torch.empty_like(v)
        
        # 状态张量保持 float32 保证精度累积
        s = torch.zeros(B, H, T // CHUNK_LEN + 1, self.head_size, self.head_size, 
                        device=x.device, dtype=torch.float32)
        sa = torch.zeros(B, T, H, self.head_size, 
                         device=x.device, dtype=torch.float32)
        
        if wkv7_cuda is not None:
            # wkv7_cuda.forward(w, r, k, v, a, b, y, s, sa)
            wkv7_cuda.forward(wk, rk, kk, vk, ak, bk, yk, s, sa)
        else:
            # y = v * r # Dummy fallback
            yk = vk * rk
            
        # y = y.view(B, T, C)
        # 3. 后处理：转回原始精度并执行“数值熔断”
        y = yk.to(dtype=x.dtype).view(B, T, C)
        
        # --- 条件 Log: 检查算子输出 ---
        if self.verbose:
            if not torch.isfinite(y).all():
                print(f"[RWKV Kernel Debug] Output 'y' exploded! Max: {y.max().item()}, Min: {y.min().item()}")
                print(f"  - Weight 'w' Max: {w.max().item()}, 'r' Max: {r.max().item()}")
        
        # --- 新增：数值截断保护 --- 2026.2.2
        # 限制在 [-24, 24] 范围内，防止 softmax 溢出产生 NaN
        y = torch.clamp(y, min=-24, max=24)
        
        # 4. GroupNorm Fix: Permute to [B, C, T]
        y = y.transpose(1, 2) # -> [B, C, T]
        y = self.ln_x(y)      # GroupNorm expects channels at dim 1
        y = y.transpose(1, 2) # -> [B, T, C]
        
        y = y * g
        y = self.output(y)
        
        return y, None