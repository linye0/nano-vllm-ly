import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_benchmark():
    RESULT_DIR = Path("result")
    # 尝试加载两种模式的数据进行对比
    paths = {
        "Chunked Prefill": RESULT_DIR / "latency_chunked.csv",
        "Legacy Prefill": RESULT_DIR / "latency_legacy.csv"
    }

    plt.figure(figsize=(12, 7))
    colors = {"Chunked Prefill": "#1f77b4", "Legacy Prefill": "#ff7f0e"}
    
    found_data = False
    for label, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            plt.plot(df['step'], df['latency_ms'], marker='o', markersize=4, 
                     label=label, color=colors[label], linewidth=2)
            found_data = True
            
            # 找到峰值并标注
            max_val = df['latency_ms'].max()
            max_step = df.loc[df['latency_ms'].idxmax(), 'step']
            plt.annotate(f'{label} Max: {max_val:.1f}ms', 
                         xy=(max_step, max_val), xytext=(max_step+2, max_val+50),
                         arrowprops=dict(facecolor=colors[label], shrink=0.05, width=1))

    if not found_data:
        print("Error: 在 result 文件夹下未找到 CSV 数据文件。")
        return

    # 样式美化
    plt.axvline(x=3, color='gray', linestyle='--', alpha=0.7)
    plt.text(3.2, 10, 'Burst Event (14k Tokens)', rotation=90, color='gray')
    
    plt.title('Performance Impact: Chunked vs Legacy Prefill (nano-vllm)', fontsize=14)
    plt.xlabel('Engine Step', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.yscale('log')  # 建议使用对数坐标，因为 Legacy 的延迟可能高出几个数量级
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend()

    # 保存图表至 result 文件夹
    save_path = RESULT_DIR / "latency_comparison_report.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Success] 对比报告已生成: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_benchmark()