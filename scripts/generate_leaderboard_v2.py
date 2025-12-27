#!/usr/bin/env python3
"""
PDEBench v2 排行榜生成器

生成两个独立的排行榜 + 按 PDE 类型的子榜单：
1. Fix Accuracy (速度榜) - 固定精度，比速度
2. Fix Time (精度榜) - 固定时间，比精度
3. PDE 类型子榜单 - 展示各类型专用指标
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import sys
from datetime import datetime
from collections import defaultdict
import numpy as np


def load_result(result_file: Path) -> Dict[str, Any]:
    """加载测试结果"""
    with open(result_file) as f:
        return json.load(f)


def load_case_config(cases_dir: Path, case_id: str) -> Dict[str, Any]:
    """加载 case 的 config.json"""
    config_path = cases_dir / case_id / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


def aggregate_results_by_agent(results_list: List[Dict], mode: str) -> List[Dict]:
    """
    将case级别的结果聚合为agent级别的统计。
    
    Args:
        results_list: case级别的结果列表
        mode: 'fix_accuracy' 或 'fix_time'
    
    Returns:
        按agent聚合的结果列表，每个包含summary
    """
    from collections import defaultdict
    
    # 按agent_name分组
    by_agent = defaultdict(list)
    for result in results_list:
        agent_name = result.get('agent_name', result.get('agent', 'unknown'))
        by_agent[agent_name].append(result)
    
    # 为每个agent生成汇总统计
    aggregated = []
    for agent_name, cases in by_agent.items():
        # 过滤有效和通过的结果
        valid_cases = [c for c in cases if c.get('status') not in ['NOT_SUBMITTED', 'ERROR']]
        passed_cases = [c for c in cases if c.get('status') == 'PASSED']
        
        # 计算tier统计
        tier_counts = {1: 0, 2: 0, 3: 0}
        for case in valid_cases:
            tier_levels = case.get('tier_levels', {})
            for level in tier_levels.get('passed', []):
                if level in tier_counts:
                    tier_counts[level] += 1
        
        # 计算tier通过率（0-1之间的小数）
        n_total = len(cases) if cases else 1
        level_pass_rates = {
            1: tier_counts[1] / n_total,
            2: tier_counts[2] / n_total,
            3: tier_counts[3] / n_total
        }
        
        # 加权分（百分制）：(L1通过数×15 + L2通过数×35 + L3通过数×50) / 总cases数
        weighted_score = (
            tier_counts[1] * 15.0 +
            tier_counts[2] * 35.0 +
            tier_counts[3] * 50.0
        ) / n_total
        
        # 提取scores
        scores = [c.get('score', 0) for c in valid_cases]
        
        # 构建聚合结果
        agent_result = {
            'agent_name': agent_name,
            'agent': agent_name,  # 兼容旧代码
            'test_mode': mode,
            'results': cases,  # 保留原始case结果
            'summary': {
                'total_cases': len(cases),
                'submitted': len([c for c in cases if c.get('status') != 'NOT_SUBMITTED']),
                'passed': len(passed_cases),
                'pass_rate': len(passed_cases) / len(cases) if cases else 0,
                'avg_score': float(np.mean(scores)) if scores else 0,
                'tier_statistics': {
                    'level_pass_rates': level_pass_rates,
                    'weighted_score': weighted_score,
                    'tier_counts': tier_counts
                }
            }
        }
        
        aggregated.append(agent_result)
    
    return aggregated
    """按 agent 和 PDE 类型聚合统计（单个 agent）"""
    stats = defaultdict(lambda: {
        "pass": 0,
        "total": 0
    })
    
    for case_result in result.get("results", []):
        case_id = case_result.get("case_id", "")
        config = load_case_config(cases_dir, case_id)
        tags = config.get("tags", {})
        
        # 按 PDE 类型统计
        for pde_type in tags.get("pde_type", ["unknown"]):
            stats[pde_type]["total"] += 1
            if case_result.get("status") == "PASSED":
                stats[pde_type]["pass"] += 1
    
    return dict(stats)


def aggregate_by_agent_and_pde_type(agent_result: Dict, cases_dir: Path) -> Dict[str, Dict]:
    """按 agent 和 PDE 类型聚合统计（单个 agent）"""
    stats = defaultdict(lambda: {
        "pass": 0,
        "total": 0
    })
    
    # 从agent_result中获取cases列表
    cases = agent_result.get('results', [])
    
    for case_result in cases:
        case_id = case_result.get("case_id", "")
        config = load_case_config(cases_dir, case_id)
        tags = config.get("tags", {})
        
        # 按 PDE 类型统计
        for pde_type in tags.get("pde_type", ["unknown"]):
            stats[pde_type]["total"] += 1
            if case_result.get("status") == "PASSED":
                stats[pde_type]["pass"] += 1
    
    return dict(stats)


def aggregate_by_agent_and_difficulty(agent_result: Dict, cases_dir: Path) -> Dict[str, Dict]:
    """按 agent 和难度挑战聚合统计（单个 agent）"""
    stats = defaultdict(lambda: {
        "pass": 0,
        "total": 0
    })
    
    # 从agent_result中获取cases列表
    cases = agent_result.get('results', [])
    
    for case_result in cases:
        case_id = case_result.get("case_id", "")
        config = load_case_config(cases_dir, case_id)
        tags = config.get("tags", {})
        difficulty_knobs = tags.get("difficulty_knobs", {})
        
        # 高反差挑战
        if difficulty_knobs.get("contrast", 0) >= 1e3:
            difficulty_label = f"high_contrast_1e{int(difficulty_knobs['contrast'] / 1e3)}k"
            stats[difficulty_label]["total"] += 1
            if case_result.get("status") == "PASSED":
                stats[difficulty_label]["pass"] += 1
        
        # 长时积分挑战
        if difficulty_knobs.get("long_time_factor", 0) > 1:
            stats["long_time_integration"]["total"] += 1
            if case_result.get("status") == "PASSED":
                stats["long_time_integration"]["pass"] += 1
    
    return dict(stats)


def aggregate_pde_type_leaderboards(
    speed_results: List[Dict], 
    accuracy_results: List[Dict],
    cases_dir: Path
) -> Dict[str, Dict]:
    """
    为每个 PDE 类型创建独立的排行榜，包含专用指标
    
    新设计：
    - 三梯度通过率（L1/L2/L3），合并速度榜和精度榜
    - 显示PDE类型专用指标（而不是中位误差和效率）
    """
    # PDE 类型映射及其专用指标
    pde_type_info = {
        "elliptic": {
            "display_name": "椭圆型方程 (Elliptic)",
            "metrics": [
                {"key": "efficiency_dof_per_sec", "name": "效率(DOF/s)", "format": ".0f"},
                {"key": "linear_iterations_mean", "name": "线性迭代", "format": ".1f"},
                {"key": "condition_number_estimate", "name": "条件数估计", "format": ".0f"}
            ]
        },
        "parabolic": {
            "display_name": "抛物型方程 (Parabolic)",
            "metrics": [
                {"key": "efficiency_workrate", "name": "WorkRate", "format": ".0f"},
                {"key": "cfl_number", "name": "CFL数", "format": ".2f"},
                {"key": "energy_decay_ratio", "name": "能量衰减", "format": ".3f"}
            ]
        },
        "hyperbolic": {
            "display_name": "双曲型方程 (Hyperbolic)",
            "metrics": [
                {"key": "cfl_number", "name": "CFL数", "format": ".2f"},
                {"key": "total_variation", "name": "总变差", "format": ".2f"},
                {"key": "energy_conservation_error", "name": "能量守恒误差", "format": ".2e"}
            ]
        },
        "incompressible_flow": {
            "display_name": "不可压缩流动 (Incompressible Flow)",
            "metrics": [
                {"key": "divergence_free_error", "name": "散度误差", "format": ".2e"},
                {"key": "pressure_iterations_mean", "name": "压力迭代", "format": ".1f"},
                {"key": "velocity_efficiency", "name": "速度效率", "format": ".0f"}
            ]
        },
        "mixed_type": {
            "display_name": "混合型方程 (Mixed Type)",
            "metrics": [
                {"key": "efficiency_dof_per_sec", "name": "效率(DOF/s)", "format": ".0f"},
                {"key": "cfl_number", "name": "CFL数", "format": ".2f"},
                {"key": "mixed_norm_error", "name": "混合范数误差", "format": ".2e"}
            ]
        },
        "dispersive": {
            "display_name": "色散方程 (Dispersive)",
            "metrics": [
                {"key": "mass_conservation_error", "name": "质量守恒", "format": ".2e"},
                {"key": "dispersion_error", "name": "色散误差", "format": ".2e"},
                {"key": "phase_velocity_error", "name": "相速度误差", "format": ".2e"}
            ]
        },
        "reaction_diffusion": {
            "display_name": "反应扩散 (Reaction-Diffusion)",
            "metrics": [
                {"key": "pattern_formation_quality", "name": "图案形成质量", "format": ".2f"},
                {"key": "reaction_balance", "name": "反应平衡", "format": ".2e"},
                {"key": "efficiency_workrate", "name": "WorkRate", "format": ".0f"}
            ]
        },
        "compressible_flow": {
            "display_name": "可压缩流动 (Compressible Flow)",
            "metrics": [
                {"key": "shock_resolution", "name": "激波分辨率", "format": ".2f"},
                {"key": "entropy_production", "name": "熵产生", "format": ".2e"},
                {"key": "mach_number", "name": "马赫数", "format": ".2f"}
            ]
        },
        "kinetic": {
            "display_name": "动理学方程 (Kinetic)",
            "metrics": [
                {"key": "velocity_space_resolution", "name": "速度空间分辨率", "format": ".0f"},
                {"key": "mass_conservation_error", "name": "质量守恒", "format": ".2e"},
                {"key": "efficiency_phase_space", "name": "相空间效率", "format": ".0f"}
            ]
        },
        "fractional": {
            "display_name": "分数阶方程 (Fractional)",
            "metrics": [
                {"key": "fractional_order", "name": "分数阶", "format": ".2f"},
                {"key": "nonlocal_operator_efficiency", "name": "非局部算子效率", "format": ".0f"},
                {"key": "memory_kernel_error", "name": "记忆核误差", "format": ".2e"}
            ]
        },
        "stochastic": {
            "display_name": "随机方程 (Stochastic)",
            "metrics": [
                {"key": "ensemble_size", "name": "集合规模", "format": ".0f"},
                {"key": "variance_estimate", "name": "方差估计", "format": ".2e"},
                {"key": "monte_carlo_efficiency", "name": "蒙特卡洛效率", "format": ".0f"}
            ]
        },
        "multiphysics": {
            "display_name": "多物理场耦合 (Multiphysics)",
            "metrics": [
                {"key": "coupling_iterations_mean", "name": "耦合迭代", "format": ".1f"},
                {"key": "energy_balance_error", "name": "能量平衡误差", "format": ".2e"},
                {"key": "interface_resolution", "name": "界面分辨率", "format": ".2f"}
            ]
        }
    }
    
    # 收集所有 PDE 类型
    all_pde_types = set()
    for result in speed_results + accuracy_results:
        for case_result in result.get("results", []):
            case_id = case_result.get("case_id", "")
            config = load_case_config(cases_dir, case_id)
            tags = config.get("tags", {})
            all_pde_types.update(tags.get("pde_type", []))
    
    # 为每个 PDE 类型构建排行榜
    pde_leaderboards = {}
    
    for pde_type in sorted(all_pde_types):
        if pde_type not in pde_type_info:
            continue  # 跳过未定义的类型
        
        info = pde_type_info[pde_type]
        agent_data = {}  # {agent_name: {...}}
        
        # 合并收集速度榜和精度榜数据
        for result in speed_results + accuracy_results:
            agent_name = result.get("agent", "Unknown")
            if agent_name not in agent_data:
                agent_data[agent_name] = {
                    "agent": agent_name,
                    "specialized_metrics_list": defaultdict(list)  # 按指标key存储值列表
                }
            
            for case_result in result.get("results", []):
                case_id = case_result.get("case_id", "")
                config = load_case_config(cases_dir, case_id)
                tags = config.get("tags", {})
                
                if pde_type in tags.get("pde_type", []):
                    # 收集专用指标（只有PASSED的case才有有效的specialized_metrics）
                    if case_result.get("status") == "PASSED":
                        spec_metrics = case_result.get("specialized_metrics", {})
                        for metric_def in info.get("metrics", []):
                            metric_key = metric_def["key"]
                            if metric_key in spec_metrics:
                                value = spec_metrics[metric_key]
                                # 只收集有效的数值
                                if isinstance(value, (int, float)) and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                                    agent_data[agent_name]["specialized_metrics_list"][metric_key].append(value)
        
        # 计算统计指标
        agents_list = []
        for agent_name, data in agent_data.items():
            # 计算专用指标的中位数
            specialized_metrics_median = {}
            has_data = False
            for metric_def in info.get("metrics", []):
                metric_key = metric_def["key"]
                values = data["specialized_metrics_list"].get(metric_key, [])
                if values:
                    specialized_metrics_median[metric_key] = np.median(values)
                    has_data = True
                else:
                    specialized_metrics_median[metric_key] = None
            
            # 只有至少有一个指标有数据的agent才加入列表
            if has_data:
                agents_list.append({
                    "agent": agent_name,
                    "specialized_metrics": specialized_metrics_median
                })
        
        # 按agent名称排序
        agents_list.sort(key=lambda x: x["agent"])
        
        pde_leaderboards[pde_type] = {
            "display_name": info["display_name"],
            "metrics": info.get("metrics", []),
            "agents": agents_list
        }
    
    return pde_leaderboards


def generate_html_leaderboard(
    speed_results: List[Dict],
    accuracy_results: List[Dict],
    output_file: Path,
    cases_dir: Path = None
):
    """生成HTML排行榜（包含 PDE 类型子榜单）"""
    
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDEBench v2 排行榜</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .meta {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .meta-item {{
            margin: 5px 0;
        }}
        
        .meta-label {{
            font-weight: 600;
            color: #495057;
        }}
        
        .meta-value {{
            color: #6c757d;
        }}
        
        .leaderboards {{
            padding: 40px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(700px, 1fr));
            gap: 40px;
        }}
        
        .leaderboard {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .leaderboard-header {{
            padding: 30px;
            color: white;
            text-align: center;
        }}
        
        .leaderboard-header.speed {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .leaderboard-header.accuracy {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        .leaderboard-header h2 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .leaderboard-header p {{
            font-size: 1em;
            opacity: 0.9;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        thead {{
            background: #f8f9fa;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }}
        
        th.center {{
            text-align: center;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #f1f3f5;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .rank {{
            font-size: 1.5em;
            font-weight: 700;
            text-align: center;
            width: 80px;
        }}
        
        .rank-1 {{
            color: #ffd700;
            text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
        }}
        
        .rank-2 {{
            color: #c0c0c0;
        }}
        
        .rank-3 {{
            color: #cd7f32;
        }}
        
        .agent-name {{
            font-weight: 600;
            font-size: 1.1em;
            color: #212529;
        }}
        
        .score {{
            text-align: center;
            font-size: 1.3em;
            font-weight: 700;
        }}
        
        .score-value {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .pass-rate {{
            text-align: center;
            font-size: 1.1em;
            color: #3498db;
            font-weight: 600;
        }}
        
        .expand-btn {{
            cursor: pointer;
            color: #667eea;
            font-size: 0.9em;
            text-decoration: none;
            margin-left: 10px;
        }}
        
        .expand-btn:hover {{
            text-decoration: underline;
        }}
        
        .hidden {{
            display: none;
        }}
        
        .agent-details {{
            margin-top: 10px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            font-size: 0.9em;
        }}
        
        .detail-section {{
            margin-bottom: 12px;
        }}
        
        .detail-section:last-child {{
            margin-bottom: 0;
        }}
        
        .detail-title {{
            font-weight: 600;
            color: #495057;
            margin-bottom: 6px;
        }}
        
        .detail-item {{
            display: inline-block;
            margin: 3px 8px 3px 0;
            padding: 4px 10px;
            background: white;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }}
        
        .detail-label {{
            font-weight: 600;
            color: #6c757d;
        }}
        
        .detail-value {{
            margin-left: 5px;
        }}
        
        .pass-good {{
            color: #27ae60;
            font-weight: 600;
        }}
        
        .pass-medium {{
            color: #f39c12;
            font-weight: 600;
        }}
        
        .pass-bad {{
            color: #e74c3c;
            font-weight: 600;
        }}
        
        .pde-type-leaderboards {{
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .pde-type-section {{
            margin-bottom: 40px;
        }}
        
        .pde-type-section:last-child {{
            margin-bottom: 0;
        }}
        
        .pde-type-header {{
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 8px 8px 0 0;
            font-size: 1.3em;
            font-weight: 600;
        }}
        
        .pde-type-table {{
            background: white;
            border-radius: 0 0 8px 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .subtitle {{
            background: #e9ecef;
            padding: 15px 30px;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }}
        
        footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}
        
        footer p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 PDEBench v2 排行榜</h1>
            <p>案例驱动的 PDE 求解器评测系统</p>
        </header>
        
        <div class="meta">
            <div class="meta-item">
                <span class="meta-label">最后更新:</span>
                <span class="meta-value">{last_updated}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">参赛 Agents:</span>
                <span class="meta-value">{total_agents}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">速度榜提交:</span>
                <span class="meta-value">{speed_count}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">精度榜提交:</span>
                <span class="meta-value">{accuracy_count}</span>
            </div>
        </div>
        
        <div class="leaderboards">
            {leaderboard_content}
        </div>
        
        {pde_type_section}
        
        <footer>
            <p>📊 评测方法说明：</p>
            <p style="margin-top: 5px;">
                <strong>🚀 速度榜 (Fix Accuracy)：</strong>固定精度要求，越快越好<br>
                <strong>🎯 精度榜 (Fix Time)：</strong>固定时间预算，精度越高越好
            </p>
            <p style="margin-top: 10px;">🎯 三档挑战：L1(低精度) < L2(中精度) < L3(高精度)，基于Oracle动态设定</p>
            <p style="margin-top: 10px; font-size: 0.95em; color: #6c757d;">
                <strong>三档计算方法：</strong>每个case先用Oracle求解器（N=oracle_resolution, P=oracle_degree）获取基准性能 E_ref（误差）和 T_ref（时间）。<br>
                <strong>精度三档：</strong> L1 = 100×E_ref（低精度），L2 = E_ref（中精度），L3 = 0.01×E_ref（高精度）<br>
                <strong>速度三档：</strong> Fast = 0.1×T_ref（快速），Medium = T_ref（中速），Slow = 10×T_ref（慢速）<br>
                <strong>加权分（百分制）：</strong> (L1通过数×15 + L2通过数×35 + L3通过数×50) / 总cases数，满分100分
            </p>
            <p style="margin-top: 10px;">PDEBench v2.0 - 案例驱动的PDE求解器评测系统</p>
        </footer>
    </div>
    
    <script>
        function toggleCaseBreakdown(id) {{
            const elem = document.getElementById(id);
            if (elem.classList.contains('hidden')) {{
                elem.classList.remove('hidden');
            }} else {{
                elem.classList.add('hidden');
            }}
        }}
    </script>
</body>
</html>
"""
    
    # 统计信息（兼容新旧格式）
    total_agents = len(set(
        [r.get('agent_name', r.get('agent', 'unknown')) for r in speed_results] +
        [r.get('agent_name', r.get('agent', 'unknown')) for r in accuracy_results]
    ))
    
    last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 生成两个主排行榜
    leaderboards_config = [
        ("speed", "🚀 速度榜", "Fix Accuracy Mode - 保证精度，越快越好", speed_results),
        ("accuracy", "🎯 精度榜", "Fix Time Mode - 时间限制，精度越高越好", accuracy_results),
    ]
    
    leaderboard_content = ""
    
    for board_class, title, desc, results in leaderboards_config:
        if not results:
            continue
            
        # 按加权分排序
        sorted_results = sorted(
            results, 
            key=lambda x: x['summary'].get('tier_statistics', {}).get('weighted_score', 0), 
            reverse=True
        )
        
        leaderboard_content += f"""
            <div class="leaderboard">
                <div class="leaderboard-header {board_class}">
                    <h2>{title}</h2>
                    <p>{desc}</p>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th class="center">排名</th>
                            <th>Agent</th>
                            <th class="center">加权分</th>
                            <th class="center">L1 (低精度)</th>
                            <th class="center">L2 (中精度)</th>
                            <th class="center">L3 (高精度)</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for rank, result in enumerate(sorted_results, 1):
            agent_name = result.get('agent_name', result.get('agent', 'unknown'))
            
            # 三档统计（level_pass_rates是0-1之间的小数）
            tier_stats = result['summary'].get('tier_statistics', {})
            level_pass_rates_raw = tier_stats.get('level_pass_rates') or {}
            level_pass_rates = {
                1: float(level_pass_rates_raw.get(1, level_pass_rates_raw.get('1', 0))) * 100,
                2: float(level_pass_rates_raw.get(2, level_pass_rates_raw.get('2', 0))) * 100,
                3: float(level_pass_rates_raw.get(3, level_pass_rates_raw.get('3', 0))) * 100
            }
            weighted_score = tier_stats.get('weighted_score', 0)
            
            # 排名样式
            rank_class = ''
            if rank == 1:
                rank_class = 'rank-1'
                rank_display = '🥇'
            elif rank == 2:
                rank_class = 'rank-2'
                rank_display = '🥈'
            elif rank == 3:
                rank_class = 'rank-3'
                rank_display = '🥉'
            else:
                rank_display = str(rank)
            
            # 生成多维度统计详情
            detail_id = f"details_{board_class}_{rank}"
            details_html = ""
            expand_link = ""
            
            if cases_dir and cases_dir.exists():
                # 获取该 agent 的 PDE 类型统计
                pde_stats = aggregate_by_agent_and_pde_type(result, cases_dir)
                difficulty_stats = aggregate_by_agent_and_difficulty(result, cases_dir)
                
                details_html = f'<div class="agent-details hidden" id="{detail_id}">'
                
                # PDE 类型统计
                if pde_stats:
                    details_html += '<div class="detail-section">'
                    details_html += '<div class="detail-title">📊 PDE 类型表现:</div>'
                    for pde_type, stats in pde_stats.items():
                        pass_rate = (stats['pass'] / max(stats['total'], 1)) * 100
                        if pass_rate >= 50:
                            color_class = 'pass-good'
                        elif pass_rate >= 30:
                            color_class = 'pass-medium'
                        else:
                            color_class = 'pass-bad'
                        details_html += (
                            f'<span class="detail-item">'
                            f'<span class="detail-label">{pde_type}</span>: '
                            f'<span class="detail-value {color_class}">{stats["pass"]}/{stats["total"]}</span> '
                            f'<span class="detail-value">({pass_rate:.0f}%)</span>'
                            f'</span>'
                        )
                    details_html += '</div>'
                
                # 难度挑战统计
                if difficulty_stats:
                    details_html += '<div class="detail-section">'
                    details_html += '<div class="detail-title">🔥 难度挑战:</div>'
                    for difficulty, stats in difficulty_stats.items():
                        pass_rate = (stats['pass'] / max(stats['total'], 1)) * 100
                        if pass_rate >= 50:
                            color_class = 'pass-good'
                        elif pass_rate >= 30:
                            color_class = 'pass-medium'
                        else:
                            color_class = 'pass-bad'
                        details_html += (
                            f'<span class="detail-item">'
                            f'<span class="detail-label">{difficulty}</span>: '
                            f'<span class="detail-value {color_class}">{stats["pass"]}/{stats["total"]}</span> '
                            f'<span class="detail-value">({pass_rate:.0f}%)</span>'
                            f'</span>'
                        )
                    details_html += '</div>'
                
                details_html += '</div>'
                
                expand_link = f'''
                            <a href="#" class="expand-btn" onclick="toggleCaseBreakdown('{detail_id}'); return false;">
                                📊 查看详细统计
                            </a>'''
            
            leaderboard_content += f"""
                        <tr>
                            <td class="rank {rank_class}">{rank_display}</td>
                            <td>
                                <div class="agent-name">{agent_name}</div>
                                {expand_link}
                                {details_html}
                            </td>
                            <td class="score">
                                <span class="score-value">{weighted_score:.2f}</span>
                            </td>
                            <td class="pass-rate">{level_pass_rates[1]:.1f}%</td>
                            <td class="pass-rate">{level_pass_rates[2]:.1f}%</td>
                            <td class="pass-rate">{level_pass_rates[3]:.1f}%</td>
                        </tr>
"""
        
        leaderboard_content += """
                    </tbody>
                </table>
            </div>
"""
    
    # 生成 PDE 类型子榜单
    pde_type_section = ""
    if cases_dir and cases_dir.exists():
        pde_leaderboards = aggregate_pde_type_leaderboards(speed_results, accuracy_results, cases_dir)
        
        if pde_leaderboards:
            pde_type_section = '<div class="pde-type-leaderboards">'
            pde_type_section += '<h2 style="text-align: center; margin-bottom: 30px; color: #495057; font-size: 2em;">📊 PDE 类型专项榜单</h2>'
            
            # 三列布局容器
            pde_type_section += '<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">'
            
            for pde_type, data in pde_leaderboards.items():
                if not data["agents"]:
                    continue
                
                # 构建表头：Agent + 三个专用指标
                metrics_headers = ""
                for metric_def in data.get("metrics", []):
                    metrics_headers += f'<th class="center">{metric_def["name"]}</th>\n                                    '
                
                pde_type_section += f'''
                <div class="pde-type-section" style="margin-bottom: 0;">
                    <div class="pde-type-header" style="font-size: 1.1em; padding: 15px 20px;">
                        {data["display_name"]}
                    </div>
                    <div class="pde-type-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Agent</th>
                                    {metrics_headers}
                                </tr>
                            </thead>
                            <tbody>
'''
                
                for agent_info in data["agents"]:
                    # 构建专用指标列
                    metrics_values = ""
                    for metric_def in data.get("metrics", []):
                        metric_key = metric_def["key"]
                        value = agent_info["specialized_metrics"].get(metric_key)
                        if value is not None:
                            format_str = metric_def.get("format", ".2f")
                            value_str = f"{value:{format_str}}"
                        else:
                            value_str = "N/A"
                        metrics_values += f'<td style="text-align: center;">{value_str}</td>\n                                    '
                    
                    pde_type_section += f'''
                                <tr>
                                    <td class="agent-name">{agent_info["agent"]}</td>
                                    {metrics_values}
                                </tr>
'''
                
                pde_type_section += '''
                            </tbody>
                        </table>
                    </div>
                </div>
'''
            
            pde_type_section += '</div></div>'  # 关闭grid容器和pde-type-leaderboards
    
    # 填充模板
    html = html_template.format(
        last_updated=last_updated,
        total_agents=total_agents,
        speed_count=len(speed_results),
        accuracy_count=len(accuracy_results),
        leaderboard_content=leaderboard_content,
        pde_type_section=pde_type_section
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description='Generate PDEBench v2 Leaderboard',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results',
        nargs='+',
        type=Path,
        required=True,
        help='Evaluation JSON files (e.g., results/*/evaluation.json)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('leaderboard_v2.html'),
        help='Output HTML file (default: leaderboard_v2.html)'
    )
    
    parser.add_argument(
        '--cases-dir',
        type=Path,
        default=Path('cases'),
        help='Cases directory (default: cases)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"📊 PDEBench v2 排行榜生成器")
    print(f"{'='*80}")
    print(f"结果文件: {', '.join(str(f) for f in args.results)}")
    print(f"输出文件: {args.output}")
    print(f"{'='*80}\n")
    
    # 收集结果（支持新格式）
    speed_results = []
    accuracy_results = []
    
    for result_file in args.results:
        if not result_file.exists():
            print(f"⚠️  文件不存在: {result_file}")
            continue
            
        try:
            data = load_result(result_file)
            
            # 检测格式：新格式（evaluation.json）还是旧格式（单次结果）
            if 'results' in data and isinstance(data['results'], dict):
                # 新格式：包含 fix_accuracy 和 fix_time 两个模式
                agent_name = data.get('agent_name', result_file.stem)
                
                # 处理 fix_accuracy 模式（速度榜）
                if 'fix_accuracy' in data['results']:
                    for case_result in data['results']['fix_accuracy']:
                        speed_result = {
                            'agent_name': agent_name,
                            'test_mode': 'fix_accuracy',
                            **case_result
                        }
                        speed_results.append(speed_result)
                    print(f"✅ 已加载 (速度榜): {agent_name} - {len(data['results']['fix_accuracy'])} cases")
                
                # 处理 fix_time 模式（精度榜）
                if 'fix_time' in data['results']:
                    for case_result in data['results']['fix_time']:
                        accuracy_result = {
                            'agent_name': agent_name,
                            'test_mode': 'fix_time',
                            **case_result
                        }
                        accuracy_results.append(accuracy_result)
                    print(f"✅ 已加载 (精度榜): {agent_name} - {len(data['results']['fix_time'])} cases")
            
            elif 'test_mode' in data:
                # 旧格式：单个文件包含一次提交
                mode = data.get('test_mode', '')
                if mode == 'fix_accuracy':
                    speed_results.append(data)
                    print(f"✅ 已加载 (速度榜): {result_file.name}")
                elif mode == 'fix_time':
                    accuracy_results.append(data)
                    print(f"✅ 已加载 (精度榜): {result_file.name}")
                else:
                    print(f"⚠️  已跳过 (未知模式): {result_file.name}")
            else:
                print(f"⚠️  已跳过 (未知格式): {result_file.name}")
        
        except Exception as e:
            print(f"❌ 加载失败 {result_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"📈 总结:")
    print(f"   速度榜 (Fix Accuracy): {len(speed_results)} 个case结果")
    print(f"   精度榜 (Fix Time): {len(accuracy_results)} 个case结果")
    print(f"{'='*80}\n")
    
    if not (speed_results or accuracy_results):
        print("❌ 没有找到有效的结果！")
        sys.exit(1)
    
    # 聚合case级别结果为agent级别
    print("📊 聚合结果...")
    speed_aggregated = aggregate_results_by_agent(speed_results, 'fix_accuracy')
    accuracy_aggregated = aggregate_results_by_agent(accuracy_results, 'fix_time')
    print(f"   速度榜: {len(speed_aggregated)} 个agent")
    print(f"   精度榜: {len(accuracy_aggregated)} 个agent\n")
    
    # 生成排行榜
    print("🎨 正在生成 HTML 排行榜（包含 PDE 类型专项榜单）...")
    generate_html_leaderboard(
        speed_results=speed_aggregated,
        accuracy_results=accuracy_aggregated,
        output_file=args.output,
        cases_dir=args.cases_dir if args.cases_dir.exists() else None
    )
    
    print(f"✅ 排行榜已生成: {args.output}")
    print(f"\n🌐 在浏览器中打开: file://{args.output.absolute()}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
