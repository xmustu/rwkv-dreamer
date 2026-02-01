import torch
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 算子定义：从 PaMoRL 提取 ---
def binary_return_fn(cur_i, cur_j):
    coef_i, in_i = cur_i
    coef_j, in_j = cur_j
    return coef_i * coef_j, coef_j * in_i + in_j

def interleave(odd, even):
    padded_odd = torch.cat((odd, torch.zeros_like(odd[-1:])), dim=0)
    outputs = torch.stack((even, padded_odd[:even.shape[0]]), dim=1)
    outputs = outputs.flatten(0, 1)[:(odd.shape[0] + even.shape[0])]
    return outputs

def odd_even_parallel_scan(inputs, operator):
    Length = inputs[0].shape[0]
    if Length < 2: return inputs
    reduced_inputs = operator(
        (input[:-1][0::2] for input in inputs), 
        (input[1::2] for input in inputs)
    )
    odd_inputs = odd_even_parallel_scan(reduced_inputs, operator)
    if Length % 2 == 0:
        even_inputs = operator((input[:-1] for input in odd_inputs), (input[2::2] for input in inputs))
    else:
        even_inputs = operator((input for input in odd_inputs), (input[2::2] for input in inputs))
    even_inputs = [torch.cat((input[0:1], even_input), dim=0) for (input, even_input) in zip(inputs, even_inputs)]
    return [interleave(odd_i, even_i) for (even_i, odd_i) in zip(even_inputs, odd_inputs)]

# --- 算子定义：从 DreamerV3 提取 ---
def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[index] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = {k: v.clone().unsqueeze(0) for k, v in last.items()} if isinstance(last, dict) else [x.clone().unsqueeze(0) for x in last]
            flag = False
        else:
            if isinstance(last, dict):
                for k in last: outputs[k] = torch.cat([outputs[k], last[k].unsqueeze(0)], dim=0)
            else:
                for j in range(len(outputs)): outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
    return outputs

# --- 实验主体 ---
def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []

    for L in L_list:
        print(f"Testing L={L}...")
        a = torch.randn(L, 16, 512, device=device)
        b = torch.randn(L, 16, 512, device=device)
        h0 = torch.zeros(16, 512, device=device)

        # 1. 测试 Static Scan (DreamerV3)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        def scan_fn(h, ai, bi): return ai * h + bi
        
        start_event.record()
        _ = static_scan(scan_fn, (a, b), h0)
        end_event.record()
        torch.cuda.synchronize()
        time_static = start_event.elapsed_time(end_event)
        mem_static = torch.cuda.max_memory_allocated() / 1024 / 1024

        # 2. 测试 Parallel Scan (PaMoRL)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_event.record()
        _ = odd_even_parallel_scan([a, b], binary_return_fn)
        end_event.record()
        torch.cuda.synchronize()
        time_parallel = start_event.elapsed_time(end_event)
        mem_parallel = torch.cuda.max_memory_allocated() / 1024 / 1024

        results.append({
            "Length": L,
            "Static_Time_ms": time_static,
            "Parallel_Time_ms": time_parallel,
            "Static_Mem_MB": mem_static,
            "Parallel_Mem_MB": mem_parallel
        })

    df = pd.DataFrame(results)
    df.to_csv("scan_benchmark_results.csv", index=False)
    print("Benchmark complete. Results saved to scan_benchmark_results.csv")
    
    # A. 绘制时间对比图
    plt.figure(figsize=(10, 6))
    plt.plot(df["Length"], df["Static_Time_ms"], 'o-', color='tab:red', label='Static Scan (DreamerV3)')
    plt.plot(df["Length"], df["Parallel_Time_ms"], 's-', color='tab:blue', label='Parallel Scan (PaMoRL)')
    plt.xscale('log', base=2)
    plt.yscale('log') # 使用对数坐标轴观察 $O(L)$ 与 $O(\log L)$ 的差异
    plt.xlabel('Sequence Length (L)', fontsize=12)
    plt.ylabel('Computation Time (ms)', fontsize=12)
    plt.title('Time Efficiency: Static vs. Parallel Scan', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig('time_efficiency.png') # 保存时间图
    plt.close()

    # B. 绘制显存对比图
    plt.figure(figsize=(10, 6))
    plt.plot(df["Length"], df["Static_Mem_MB"], 'o-', color='tab:red', label='Static Scan (DreamerV3)')
    plt.plot(df["Length"], df["Parallel_Mem_MB"], 's-', color='tab:blue', label='Parallel Scan (PaMoRL)')
    plt.xscale('log', base=2)
    plt.xlabel('Sequence Length (L)', fontsize=12)
    plt.ylabel('Peak Memory (MB)', fontsize=12)
    plt.title('Memory Usage: Static vs. Parallel Scan', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig('memory_efficiency.png') # 保存显存图
    plt.close()

    print("实验图表已成功保存为 'time_efficiency.png' 和 'memory_efficiency.png'。")

if __name__ == "__main__":
    run_benchmark()