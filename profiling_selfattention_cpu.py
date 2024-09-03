import torch
import torch.nn.functional as F
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tracemalloc

def self_attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

def check_performance(input_length, num_trials):
    device = torch.device('cpu')
    n = input_length
    d = 4
    dq = 2
    dk = dq
    dv = 4
    flops = []
    times = []
    memory = []

    for _ in range(num_trials):
        X = torch.randn(n, d, device=device)
        Wq = torch.rand(d, dq, device=device)
        Wk = torch.rand(d, dk, device=device)
        Wv = torch.randn(d, dv, device=device)

        start_time = time.time()
        Q = torch.matmul(X, Wq)
        K = torch.matmul(X, Wk)
        V = torch.matmul(X, Wv)
        tracemalloc.start()
        S = self_attention(Q, K, V)
        current, peak = tracemalloc.get_traced_memory()
        end_time = time.time()

        memory_usage = peak - current
        tracemalloc.stop()

        times.append(end_time - start_time)
        memory.append(memory_usage)
        FLOPs = (2 * (n * dq * d + n * dq * (d - 1))  # XWq + XWk
                + n * dv * d + n * dv * (d - 1)  # XWv
                + n * n * dq + n * n * (dq - 1)  # Q.K
                + 1  # Sqrt(dq)
                + n * n + n * (n - 1) + n * n  # softmax exp + add + division
                + n * dv * d + n * dv + (d - 1))  # weights x V
        # FLOPs = torchprofile.profile_macs(model, input_tensor)
        flops.append(FLOPs)

    flops_mean = np.mean(flops)
    memory_mean = np.mean(memory)
    time_mean = np.mean(times)

    flops_std_err = np.std(flops) / np.sqrt(num_trials)
    memory_std_err = np.std(memory) / np.sqrt(num_trials)
    time_std_err = np.std(times) / np.sqrt(num_trials)

    return flops_mean, memory_mean, time_mean, flops_std_err, memory_std_err, time_std_err

input_lengths = [10, 20, 30, 50, 100, 500, 750, 1000,3000, 5000, 6000, 8000, 10000, 50000]
num_trials = 10

results = {
    'Input Length': [],
    'FLOPS': [],
    'Memory Usage (bytes)': [],
    'Time': [],
    'FLOPS Std Err': [],
    'Memory Std Err': [],
    'Time Std Err': []
}
print(results)

for seq_len in input_lengths:
    flops_mean, memory_mean, time_mean, flops_std_err, memory_std_err, time_std_err = check_performance(seq_len, num_trials)
    results['Input Length'].append(seq_len)
    results['FLOPS'].append(flops_mean)
    results['Memory Usage (bytes)'].append(memory_mean)
    results['Time'].append(time_mean)
    results['FLOPS Std Err'].append(flops_std_err)
    results['Memory Std Err'].append(memory_std_err)
    results['Time Std Err'].append(time_std_err)

df = pd.DataFrame(results)
print(results)
sns.set_theme(style="darkgrid")

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.lineplot(data=df, x='Input Length', y='FLOPS', color='b', label='Mean')

plt.fill_between(df['Input Length'],
                 df['FLOPS'] - df['FLOPS Std Err'],
                 df['FLOPS'] + df['FLOPS Std Err'],
                 color='b',
                 alpha=0.2,
                 label='±1 Std Dev')

plt.xlabel('Input Length')
plt.ylabel('FLOPS')
plt.title('Computational Complexity')

plt.subplot(1, 3, 2)
sns.lineplot(data=df, x='Input Length', y='Memory Usage (bytes)', color='b', label='Mean')
plt.fill_between(df['Input Length'],
                 df['Memory Usage (bytes)'] - df['Memory Std Err'],
                 df['Memory Usage (bytes)'] + df['Memory Std Err'],
                 color='b',
                 alpha=0.2,
                 label='±1 Std Dev')
plt.xlabel('Input Length')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory Usage')

plt.subplot(1, 3, 3)
sns.lineplot(data=df, x='Input Length', y='Time', color='b', label='Mean')
plt.fill_between(df['Input Length'],
                 df['Time'] - df['Time Std Err'],
                 df['Time'] + df['Time Std Err'],
                 color='b',
                 alpha=0.2,
                 label='±1 Std Dev')
plt.xlabel('Input Length')
plt.ylabel('Time')
plt.title('Time')

plt.tight_layout()
plt.savefig('my_plot_cpu_with_error_bars.png')
plt.show()
