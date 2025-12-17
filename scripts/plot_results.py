#!/usr/bin/env python3
"""
PDEBench 实验结果可视化脚本

读取 experiment_history.jsonl 并生成用于 NeurIPS 论文的高质量图表：
1. Time-Accuracy Trade-off (帕累托前沿)
2. Optimization Trajectory (优化轨迹)

用法：
    python scripts/plot_results.py
    python scripts/plot_results.py --output figures/ --format pdf
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime

# 设置论文级别的绘图风格
mpl.rcParams['font.size'] = 11
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14


def load_experiment_history(log_file):
    """加载实验历史记录"""
    experiments = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    return experiments


def plot_pareto_front(experiments, output_dir, fmt='png'):
    """
    绘制 Time-Accuracy Trade-off 帕累托前沿图
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    times = []
    errors = []
    labels = []
    colors_list = []
    
    # 使用颜色映射表示时间顺序
    cmap = plt.cm.viridis
    n_exp = len(experiments)
    
    for idx, exp in enumerate(experiments):
        summary = exp['summary']
        
        # 只绘制通过的实验
        if summary['pass_rate'] == 1.0:
            times.append(summary['total_wall_time'])
            errors.append(summary['avg_rel_error'])
            labels.append(exp['experiment_id'])
            colors_list.append(cmap(idx / max(n_exp - 1, 1)))
    
    if not times:
        print("⚠️  没有成功的实验记录，无法绘制帕累托图")
        return
    
    # 绘制散点，颜色表示时间顺序
    scatter = ax.scatter(times, errors, c=range(len(times)), cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                         zorder=3)
    
    # 标注第一个和最后一个点
    if len(times) > 0:
        ax.annotate('Baseline', (times[0], errors[0]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    if len(times) > 1:
        ax.annotate('Latest', (times[-1], errors[-1]), 
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=9, color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # 找到帕累托前沿
    pareto_indices = []
    for i in range(len(times)):
        is_pareto = True
        for j in range(len(times)):
            if i != j:
                # 如果存在另一个点既更快又更准，当前点就不在帕累托前沿上
                if times[j] <= times[i] and errors[j] <= errors[i]:
                    if times[j] < times[i] or errors[j] < errors[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)
    
    # 绘制帕累托前沿连线
    if len(pareto_indices) > 1:
        pareto_times = [times[i] for i in sorted(pareto_indices, key=lambda x: times[x])]
        pareto_errors = [errors[i] for i in sorted(pareto_indices, key=lambda x: times[x])]
        ax.plot(pareto_times, pareto_errors, 'r--', alpha=0.5, linewidth=2, 
               label='Pareto Front', zorder=2)
    
    ax.set_xlabel('Total Wall Time (seconds)', fontweight='bold')
    ax.set_ylabel('Average Relative Error (L2)', fontweight='bold')
    ax.set_title('Time-Accuracy Trade-off (Pareto Front)', fontweight='bold', pad=15)
    
    # 使用 log scale 以便更好地显示
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加颜色条表示实验顺序
    cbar = plt.colorbar(scatter, ax=ax, label='Experiment Index (Chronological)')
    
    # 添加图例
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = output_dir / f'pareto_front.{fmt}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 帕累托前沿图已保存: {output_path}")
    plt.close()


def plot_optimization_trajectory(experiments, output_dir, fmt='png'):
    """
    绘制优化轨迹图，展示性能随实验步数的变化
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    steps = list(range(1, len(experiments) + 1))
    times = [exp['summary']['total_wall_time'] for exp in experiments]
    errors = [exp['summary']['avg_rel_error'] for exp in experiments]
    pass_rates = [exp['summary']['pass_rate'] * 100 for exp in experiments]
    iters = [exp['summary']['avg_iters'] for exp in experiments]
    
    # 子图 1: 总耗时
    ax1.plot(steps, times, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Wall Time')
    ax1.fill_between(steps, times, alpha=0.3, color='#1f77b4')
    ax1.set_ylabel('Total Wall Time (s)', fontweight='bold')
    ax1.set_title('Optimization Trajectory Over Experiments', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best')
    
    # 标注最优点
    min_time_idx = np.argmin(times)
    ax1.scatter([steps[min_time_idx]], [times[min_time_idx]], 
               color='red', s=150, zorder=5, marker='*', label='Best')
    
    # 子图 2: 平均误差
    ax2.plot(steps, errors, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='Avg Rel Error')
    ax2.fill_between(steps, errors, alpha=0.3, color='#ff7f0e')
    ax2.set_ylabel('Avg Relative Error', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best')
    
    # 子图 3: 通过率 & 平均迭代次数
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(steps, pass_rates, '^-', color='#2ca02c', linewidth=2, 
                     markersize=6, label='Pass Rate (%)')
    line2 = ax3_twin.plot(steps, iters, 'd-', color='#d62728', linewidth=2, 
                          markersize=6, label='Avg Iterations')
    
    ax3.set_xlabel('Experiment Step', fontweight='bold')
    ax3.set_ylabel('Pass Rate (%)', fontweight='bold', color='#2ca02c')
    ax3_twin.set_ylabel('Avg Iterations', fontweight='bold', color='#d62728')
    
    ax3.tick_params(axis='y', labelcolor='#2ca02c')
    ax3_twin.tick_params(axis='y', labelcolor='#d62728')
    
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = output_dir / f'optimization_trajectory.{fmt}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 优化轨迹图已保存: {output_path}")
    plt.close()


def plot_per_case_comparison(experiments, output_dir, fmt='png'):
    """
    绘制每个 case 在不同实验中的性能对比（仅对比首次和最新）
    """
    if len(experiments) < 2:
        print("⚠️  实验记录少于 2 个，跳过 per-case 对比图")
        return
    
    baseline = experiments[0]
    latest = experiments[-1]
    
    case_ids = list(baseline['per_case'].keys())
    
    baseline_times = [baseline['per_case'][c]['wall_time'] for c in case_ids]
    latest_times = [latest['per_case'][c]['wall_time'] for c in case_ids]
    
    speedup = [b / l if l > 0 else 0 for b, l in zip(baseline_times, latest_times)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(case_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_times, width, label='Baseline', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, latest_times, width, label='Latest', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Test Case', fontweight='bold')
    ax.set_ylabel('Wall Time (seconds)', fontweight='bold')
    ax.set_title('Per-Case Performance: Baseline vs Latest', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上标注加速比
    for i, (b, l, s) in enumerate(zip(baseline_times, latest_times, speedup)):
        if s > 1:
            ax.text(i, max(b, l) * 1.05, f'{s:.2f}x', ha='center', fontsize=8, 
                   color='green', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / f'per_case_comparison.{fmt}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Per-case 对比图已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="PDEBench 实验结果可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log-file",
        default="experiment_history.jsonl",
        help="实验历史日志文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default="figures",
        help="输出目录"
    )
    parser.add_argument(
        "--format",
        choices=['png', 'pdf', 'svg'],
        default='png',
        help="图片格式"
    )
    
    args = parser.parse_args()
    
    # 设置路径
    repo_root = Path(__file__).parent.parent
    log_file = repo_root / args.log_file
    output_dir = repo_root / args.output
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查日志文件是否存在
    if not log_file.exists():
        print(f"❌ 未找到实验历史文件: {log_file}")
        print(f"   请先运行: python scripts/benchmark_score.py --log-history")
        return 1
    
    # 加载实验历史
    print(f"📖 读取实验历史: {log_file}")
    experiments = load_experiment_history(log_file)
    print(f"   共找到 {len(experiments)} 条实验记录\n")
    
    if len(experiments) == 0:
        print("❌ 实验历史为空，无法绘图")
        return 1
    
    # 生成图表
    print("🎨 开始生成图表...\n")
    
    plot_pareto_front(experiments, output_dir, args.format)
    plot_optimization_trajectory(experiments, output_dir, args.format)
    plot_per_case_comparison(experiments, output_dir, args.format)
    
    print(f"\n✅ 所有图表已生成完毕！")
    print(f"📁 输出目录: {output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

