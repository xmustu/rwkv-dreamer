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
            verbose=True, 
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
        
        # 1. 增加 LayerNorm 稳定输入
        self.ln_in = nn.LayerNorm(dim)
        
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
        # 1. 输入归一化
        # print(x)
        # print(x.max(), x.min(), x.mean())
        x = self.ln_in(x)
        # print(x)
        print(state)
        # print(x.shape)
        # print(x.max(), x.min(), x.mean())
        B, T, C = x.size()
        H = self.n_head
        N = self.head_size
        # --- [1. 输入检查] ---
        if self.verbose and not torch.isfinite(x).all():
            print(f"\033[91m[DEBUG RWKV] Input 'x' is already NaN before any calc!\033[0m")
            
        # 1. Time Shift 
        xx = self.time_shift(x) - x
        x_in = x + xx * 0.5 
        
     
        # 2. Project
        r = self.receptance(x_in).view(B, T, H, N)
        k = self.key(x_in).view(B, T, H, N)
        v = self.value(x_in).view(B, T, H, N)
        g = torch.sigmoid(self.gate(x_in))
        
        w_raw = self.decay(x_in).view(B, T, H, N)
        w = -torch.exp(w_raw.float()) 
        
        a = torch.sigmoid(self.a_proj(x_in)).view(B, T, H, N)
        # --- [2. 投影层检查] ---
        if self.verbose:
            names = ['r', 'k', 'v', 'w', 'a', 'g']
            tensors = [r, k, v, w, a, g]
            for name, t in zip(names, tensors):
                if not torch.isfinite(t).all():
                    print(f"\033[91m[DEBUG RWKV] Projection '{name}' is NaN!\033[0m Possible weight corruption.")
               
        # 3. CUDA Kernel
        # r = r.view(B, T, H, -1).float().contiguous()
        # k = k.view(B, T, H, -1).float().contiguous()
        # v = v.view(B, T, H, -1).float().contiguous()
        # w = w.view(B, T, H, -1).float().contiguous()
        # a = a.view(B, T, H, -1).float().contiguous()
        # b = torch.ones_like(a).float().contiguous()
        b = torch.ones_like(a)
        # 2. 状态初始化 (矩阵状态 S: [B, H, N, N])
        # 训练时 state 通常为 None，想象时 state 为 [B, H, N, N]
        if state is None:
            state = torch.zeros(B, H, N, N, device=x.device, dtype=torch.float32)
        else:
            state = state.float()
            if self.verbose and not torch.isfinite(state).all():
                print(f"\033[91m[DEBUG RWKV] Input 'state' is NaN! Source: Previous img_step.\033[0m")
                
        if T > 1 and wkv7_cuda is not None:
            # --- 并行模式 (训练): 使用 CUDA Kernel ---
            # 统一使用 float32 连续张量，防止 illegal memory access
            r_c, k_c, v_c, w_c, a_c, b_c = [i.float().contiguous() for i in [r, k, v, w, a, b]]
            y = torch.empty_like(v_c)
            # 注意：这里的 s_chunk 是为了内核内部加速，不是最终递归状态
            s_chunk = torch.zeros(B, H, T // CHUNK_LEN + 1, N, N, device=x.device, dtype=torch.float32)
            sa_chunk = torch.zeros(B, T, H, N, device=x.device, dtype=torch.float32)
            
            
            wkv7_cuda.forward(w_c, r_c, k_c, v_c, a_c, b_c, y, s_chunk, sa_chunk)
            y = y.to(x.dtype).view(B, T, C)
            print("y: ", y)
            # 更新状态为序列最后一步（此处需通过数学模拟或内核输出获取最终状态）
            # 简化起见，MBRL 训练时 state 主要在 img_step 递归中使用
            next_state = state # 训练时通常不回传状态，仅想象时回传
        else:
            # --- 递归模式 (想象): T=1 或单步推理 ---
            # 实现与内核完全一致的数学逻辑
            y_out = []
            for t in range(T):
                # 获取当前时刻向量 [B, H, N]
                rt, kt, vt, at, bt, wt = r[:, t], k[:, t], v[:, t], a[:, t], b[:, t], w[:, t]
                # 计算衰减因子: exp(-exp(w_raw))
                # dt = torch.exp(-torch.exp(w_raw[:, t].float())) 
                
                # 核心递归公式:
                # 1. sa = S * a
                sa = torch.einsum('bhij,bhj->bhi', state, at) 
                
                # 检查 sa 
                if self.verbose and not torch.isfinite(sa).all():
                    print(f"\033[91m[DEBUG RWKV] sa = S*a became NaN at step {t}!\033[0m")
                    
                # 2. S = S * w + sa @ b.T + k @ v.T
                state = state * torch.exp(wt).unsqueeze(-1) + \
                        torch.einsum('bhi,bhj->bhij', sa, bt) + \
                        torch.einsum('bhi,bhj->bhij', kt, vt)
                # 3. y = S * r
                yt = torch.einsum('bhij,bhj->bhi', state, rt)
                print("yt: ", yt)
                y_out.append(yt)
            
            y = torch.stack(y_out, dim=1).view(B, T, C)
            next_state = state
        # # 必须确保输入到 CUDA 的是连续的 bfloat16 张量
        # def to_kernel(t):
        #     return t.view(B, T, H, -1).to(dtype=torch.bfloat16).contiguous()
        
        # rk, kk, vk, wk, ak = map(to_kernel, [r, k, v, w, a])
        # bk = torch.ones_like(ak)
        # yk = torch.empty_like(vk)
        
        # y = torch.empty_like(v)
        
        # # 状态张量保持 float32 保证精度累积
        # s = torch.zeros(B, H, T // CHUNK_LEN + 1, self.head_size, self.head_size, 
        #                 device=x.device, dtype=torch.float32)
        # sa = torch.zeros(B, T, H, self.head_size, 
        #                  device=x.device, dtype=torch.float32)
        
        # if wkv7_cuda is not None:
        #     # wkv7_cuda.forward(w, r, k, v, a, b, y, s, sa)
        #     wkv7_cuda.forward(wk, rk, kk, vk, ak, bk, yk, s, sa)
        # else:
        #     # y = v * r # Dummy fallback
        #     yk = vk * rk
            
        # y = y.view(B, T, C)
        # 3. 后处理：转回原始精度并执行“数值熔断”
        y = y.to(dtype=x.dtype).view(B, T, C)
        
        # --- [4. 数值熔断与输出检查] ---
        if not torch.isfinite(y).all():
            if self.verbose:
                print(f"\033[91m[DEBUG RWKV] Final y_out is NaN! T={T}\033[0m")
            # 强力恢复，防止梯度污染
            y = torch.nan_to_num(y, nan=0.0)
            next_state = torch.nan_to_num(next_state, nan=0.0)
        # --- 新增：数值截断保护 --- 2026.2.2
        # 限制在 [-24, 24] 范围内，防止 softmax 溢出产生 NaN
        # y = torch.clamp(y, min=-24, max=24)
        
        # 4. GroupNorm Fix: Permute to [B, C, T]
        y = y.transpose(1, 2) # -> [B, C, T]
        y = self.ln_x(y)      # GroupNorm expects channels at dim 1
        y = y.transpose(1, 2) # -> [B, T, C]
        
        y = y * g
        y = self.output(y)
        
        return y, next_state