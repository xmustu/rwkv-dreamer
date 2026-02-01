import sys
import os
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

# 强制刷新
sys.stdout.reconfigure(line_buffering=True)

print("[Init] 初始化环境...")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'pwm_atari'))

# 导入模块
try:
    from modules.parallel_rnns import ParaRNNLayer
    from rwkv7_layer import RWKV7Layer
    print("[Init] 模块导入成功")
except ImportError as e:
    print(f"\n[Error] 模块导入失败: {e}")
    sys.exit(1)

def benchmark_one_step(model, x, name="Model"):
    # 预热
    try:
        for _ in range(3):
            y, _ = model(x)
            y.sum().backward()
            model.zero_grad()
    except Exception as e:
        print(f"  [Warmup Failed] {e}")
        return 0, 0, 0
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Forward
    start_t = time.time()
    y, _ = model(x)
    loss = y.sum()
    torch.cuda.synchronize()
    fwd_time = (time.time() - start_t) * 1000
    
    # Backward
    start_t = time.time()
    loss.backward()
    torch.cuda.synchronize()
    bwd_time = (time.time() - start_t) * 1000
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    return fwd_time, bwd_time, peak_mem

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on {device}")
    
    if device.type == 'cpu':
        print("[Error] 必须使用 GPU 运行此实验！")
        return

    B = 4
    D = 512
    L_list = [128, 512, 1024, 2048, 4096]
    
    results = []
    
    print(f"\n[Config] Batch={B}, Dim={D}")
    print(f"{'Method':<15} | {'Length':<6} | {'Time(ms)':<10} | {'Mem(MB)':<10}")
    print("-" * 50)
    
    for L in L_list:
        # 输入数据: [Batch, Time, Dim]
        x = torch.randn(B, L, D, device=device, requires_grad=True)
        
        # --- RWKV v7 测试 ---
        # RWKV 期望 [Batch, Time, Dim]，直接使用 x
        try:
            rwkv = RWKV7Layer(dim=D, head_size=64).to(device)
            fwd, bwd, mem = benchmark_one_step(rwkv, x, "RWKV")
            total_t = fwd + bwd
            
            results.append({"Length": L, "Method": "RWKV v7", "Time": total_t, "Memory": mem})
            print(f"{'RWKV v7':<15} | {L:<6} | {total_t:<10.2f} | {mem:<10.2f}")
            del rwkv
        except Exception as e:
            print(f"RWKV Error at L={L}: {e}")
            
        # --- PSSM 测试 ---
        # PSSM 期望 [Time, Batch, Dim]，需要转置
        try:
            pssm = ParaRNNLayer(inp_size=D, hidden=D, divisor=4).to(device)
            
            # PSSM 数据准备 (需要转置为 T-major)
            # drops: [T, B, 1]
            drops = torch.zeros(L, B, 1, device=device)
            # is_first: [T, B, 1]
            is_first = torch.zeros(L, B, 1, device=device)
            # inits: Batch-major 即可
            inits = (torch.zeros(B, D, device=device), torch.zeros(B, D*D//16, device=device))
            
            # 输入 x 也需要转置: [B, T, D] -> [T, B, D]
            x_pssm = x.transpose(0, 1).contiguous()
            
            class PSSMWrapper(torch.nn.Module):
                def __init__(self, l): super().__init__(); self.l = l
                def forward(self, inp):
                    # inp is already [T, B, D]
                    out, _, _ = self.l._parallel_forward(inp, drops, is_first, inits)
                    # output is [T, B, D], transpose back to [B, T, D] for consistent loss calculation
                    return out.transpose(0, 1), None
            
            model = PSSMWrapper(pssm)
            fwd, bwd, mem = benchmark_one_step(model, x_pssm, "PSSM")
            total_t = fwd + bwd
            
            results.append({"Length": L, "Method": "PSSM", "Time": total_t, "Memory": mem})
            print(f"{'PSSM':<15} | {L:<6} | {total_t:<10.2f} | {mem:<10.2f}")
            del pssm, model, x_pssm
        except Exception as e:
            print(f"PSSM Error at L={L}: {e}")
            import traceback
            traceback.print_exc()
            
        torch.cuda.empty_cache()

    # --- 绘图 ---
    if results:
        df = pd.DataFrame(results)
        df.to_csv("final_benchmark.csv", index=False)
        try:
            plt.figure(figsize=(10,5))
            for m in df['Method'].unique():
                d = df[df['Method'] == m]
                plt.plot(d['Length'], d['Time'], marker='o', label=m)
            plt.legend(); plt.title("Speed Comparison"); plt.ylabel("Time (ms)")
            plt.savefig("benchmark_speed.png")
            
            plt.figure(figsize=(10,5))
            for m in df['Method'].unique():
                d = df[df['Method'] == m]
                plt.plot(d['Length'], d['Memory'], marker='o', label=m)
            plt.legend(); plt.title("Memory Comparison"); plt.ylabel("Memory (MB)")
            plt.savefig("benchmark_memory.png")
            
            print("\n[Success] 结果已保存至 benchmark_speed.png 和 benchmark_memory.png")
        except:
            pass

if __name__ == "__main__":
    run_experiment()