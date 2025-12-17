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
    不同模型用不同颜色标记
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 按模型家族分组并收集数据
    model_data = {
        'Baseline': {'times': [], 'errors': [], 'labels': [], 'color': '#1f77b4', 'marker': 'o'},
        'GPT-5.2': {'times': [], 'errors': [], 'labels': [], 'color': '#2ca02c', 'marker': 's'},
        'Gemini-2.5-Pro': {'times': [], 'errors': [], 'labels': [], 'color': '#ff7f0e', 'marker': '^'}
    }
    
    for exp in experiments:
        summary = exp['summary']
        exp_id = exp['experiment_id']
        
        # 确定模型家族
        if 'gpt' in exp_id.lower():
            family = 'GPT-5.2'
        elif 'gemini' in exp_id.lower() or 'superassistant' in exp_id.lower():
            family = 'Gemini-2.5-Pro'
        elif 'baseline' == exp_id:
            family = 'Baseline'
        else:
            continue
        
        model_data[family]['times'].append(summary['total_wall_time'])
        model_data[family]['errors'].append(summary['avg_rel_error'])
        model_data[family]['labels'].append(exp_id)
    
    # 绘制每个模型家族
    all_times = []
    all_errors = []
    
    for family, data in model_data.items():
        if not data['times']:
            continue
        
        # 绘制散点
        ax.scatter(data['times'], data['errors'], 
                  c=data['color'], marker=data['marker'], s=150, 
                  alpha=0.7, edgecolors='black', linewidth=1.5,
                  label=family, zorder=3)
        
        all_times.extend(data['times'])
        all_errors.extend(data['errors'])
        
        # 标注该家族的最优点（最快的）
        if len(data['times']) > 0:
            best_idx = np.argmin(data['times'])
            best_time = data['times'][best_idx]
            best_error = data['errors'][best_idx]
            
            # 简化标签名称
            label_text = family
            if family == 'Baseline':
                label_text = 'Baseline'
            elif family == 'GPT-5.2':
                label_text = 'GPT-5.2\n(100% pass)'
            elif family == 'Gemini-2.5-Pro':
                label_text = f'Gemini-2.5-Pro\n({best_time:.2f}s, 90% pass)'
            
            # 调整标注位置避免重叠
            xytext_offset = (15, 10) if family != 'Gemini-2.5-Pro' else (15, -20)
            
            ax.annotate(label_text, (best_time, best_error), 
                       xytext=xytext_offset, textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor=data['color'], alpha=0.3, 
                                edgecolor=data['color'], linewidth=2),
                       arrowprops=dict(arrowstyle='->', 
                                      connectionstyle='arc3,rad=0.3',
                                      color=data['color'], lw=1.5))
    
    if not all_times:
        print("⚠️  没有有效的实验记录，无法绘制帕累托图")
        return
    
    # 找到帕累托前沿
    pareto_indices = []
    for i in range(len(all_times)):
        is_pareto = True
        for j in range(len(all_times)):
            if i != j:
                # 如果存在另一个点既更快又更准，当前点就不在帕累托前沿上
                if all_times[j] <= all_times[i] and all_errors[j] <= all_errors[i]:
                    if all_times[j] < all_times[i] or all_errors[j] < all_errors[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)
    
    # 绘制帕累托前沿连线
    if len(pareto_indices) > 1:
        pareto_times = [all_times[i] for i in sorted(pareto_indices, key=lambda x: all_times[x])]
        pareto_errors = [all_errors[i] for i in sorted(pareto_indices, key=lambda x: all_times[x])]
        ax.plot(pareto_times, pareto_errors, 'r--', alpha=0.5, linewidth=2, 
               label='Pareto Front', zorder=2)
    
    ax.set_xlabel('Total Wall Time (seconds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Average Relative Error (L2)', fontweight='bold', fontsize=12)
    ax.set_title('Time-Accuracy Trade-off: LLM Agent Comparison', fontweight='bold', pad=15, fontsize=14)
    
    # 使用 log scale 以便更好地显示
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    
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
    绘制每个 case 在不同实验中的性能对比
    智能选择: baseline + 每个模型家族的最优结果
    """
    if len(experiments) < 2:
        print("⚠️  实验记录少于 2 个，跳过 per-case 对比图")
        return
    
    # 按模型家族分组
    model_families = {}
    for exp in experiments:
        exp_id = exp['experiment_id']
        # 提取模型名称（去掉后缀）
        if 'gpt' in exp_id.lower():
            family = 'GPT-5.2'
        elif 'gemini' in exp_id.lower() or 'superassistant' in exp_id.lower():
            family = 'Gemini-2.5-Pro'
        elif 'baseline' == exp_id:
            family = 'Baseline'
        else:
            family = 'Other'
        
        if family not in model_families:
            model_families[family] = []
        model_families[family].append(exp)
    
    # 选择每个家族的最优结果（最短时间 + 100% pass rate）
    selected_experiments = []
    selected_labels = []
    
    # 先添加 Baseline
    if 'Baseline' in model_families:
        selected_experiments.append(model_families['Baseline'][0])
        selected_labels.append('Baseline')
    
    # 再添加其他模型的最优结果
    for family in ['GPT-5.2', 'Gemini-2.5-Pro', 'Other']:
        if family in model_families:
            # 优先选择 100% pass rate 的，然后选择最短时间的
            family_exps = model_families[family]
            best = min(family_exps, key=lambda e: (
                -e['summary']['pass_rate'],  # 优先高通过率
                e['summary']['total_wall_time']  # 其次低时间
            ))
            selected_experiments.append(best)
            selected_labels.append(f"{family} (best)")
    
    if len(selected_experiments) < 2:
        print("⚠️  没有足够的实验进行对比")
        return
    
    # 获取所有 case
    case_ids = list(selected_experiments[0]['per_case'].keys())
    
    # 准备绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x = np.arange(len(case_ids))
    width = 0.8 / len(selected_experiments)
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    # 子图 1: 耗时对比
    for idx, (exp, label) in enumerate(zip(selected_experiments, selected_labels)):
        times = [exp['per_case'][c]['wall_time'] for c in case_ids]
        offset = (idx - len(selected_experiments)/2 + 0.5) * width
        ax1.bar(x + offset, times, width, label=label, 
               color=colors[idx % len(colors)], alpha=0.8)
    
    ax1.set_ylabel('Wall Time (seconds)', fontweight='bold')
    ax1.set_title('Per-Case Performance Comparison: Wall Time', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_ids, rotation=45, ha='right')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 子图 2: 迭代次数对比
    for idx, (exp, label) in enumerate(zip(selected_experiments, selected_labels)):
        iters = [exp['per_case'][c]['iters'] for c in case_ids]
        offset = (idx - len(selected_experiments)/2 + 0.5) * width
        ax2.bar(x + offset, iters, width, label=label, 
               color=colors[idx % len(colors)], alpha=0.8)
    
    ax2.set_xlabel('Test Case', fontweight='bold')
    ax2.set_ylabel('Iterations', fontweight='bold')
    ax2.set_title('Per-Case Performance Comparison: Iterations', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(case_ids, rotation=45, ha='right')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = output_dir / f'per_case_comparison.{fmt}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Per-case 对比图已保存: {output_path}")
    print(f"   对比模型: {', '.join(selected_labels)}")
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

