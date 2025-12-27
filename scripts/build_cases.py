#!/usr/bin/env python3
"""
Cases 构建脚本

从 pdebench/data/benchmark.jsonl 生成完整的 cases/ 目录

核心逻辑：
1. 读取源数据 (JSONL)
2. 运行 Oracle 获取基准性能
3. 计算动态难度分级
4. 生成 config.json, description.md, test_*.py
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# 添加 pdebench 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.templates.prompts import generate_description_md
from pdebench.templates.scripts import generate_test_script


def run_oracle(oracle_config: Dict[str, Any], case_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行 Oracle 求解器获取基准性能
    
    Returns:
        {
            'error': float,  # 参考误差
            'time': float    # 参考时间
        }
    """
    print(f"   🔮 Running Oracle for {case_id}...")
    
    try:
        from pdebench.oracle.core.generate import generate
        from pdebench.oracle.core.solve import solve_case
        from pdebench.oracle.core.evaluate import evaluate
        
        # 在临时目录运行 Oracle
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # 准备完整的 Oracle 配置（需要补充 targets 字段）
            # 因为 evaluate() 期望 case_spec['targets']
            oracle_config_full = dict(oracle_config)
            if 'targets' not in oracle_config_full:
                # 从 evaluation_config 推断 targets
                eval_cfg = entry.get('evaluation_config', {})
                oracle_config_full['targets'] = {
                    'metric': eval_cfg.get('target_metric', 'rel_L2_fe'),
                    'target_error': 1e-2  # 临时值，反正我们只要误差数据
                }
            
            # 运行三阶段 Oracle pipeline
            generate(oracle_config_full, tmppath)
            solve_case(oracle_config_full, tmppath)
            metrics = evaluate(oracle_config_full, tmppath)
            
            # 读取元数据获取时间
            with open(tmppath / 'meta.json') as f:
                meta = json.load(f)
            
            # 提取基准指标
            target_metric = oracle_config_full['targets']['metric']
            error_ref = metrics.get(target_metric, metrics.get('rel_L2_fe', 1e-3))
            time_ref = meta.get('wall_time_sec', 10.0)
            
            print(f"   ✅ Oracle baseline: E_ref={error_ref:.2e}, T_ref={time_ref:.3f}s")
            
            return {
                'error': float(error_ref),
                'time': float(time_ref)
            }
            
    except Exception as e:
        print(f"   ⚠️  Oracle failed, using default baseline: {e}")
        import traceback
        traceback.print_exc()
        # 如果 Oracle 失败，使用保守的默认值
        return {
            'error': 1e-2,
            'time': 10.0
        }


def calculate_difficulty_tiers(baseline_error: float, baseline_time: float) -> Dict[str, Any]:
    """
    基于 Oracle 基准动态计算难度分级
    
    Args:
        baseline_error: Oracle 的参考误差
        baseline_time: Oracle 的参考时间
        
    Returns:
        {
            'accuracy': {
                'level_1': {'target_error': ..., 'name': 'Low/Engineering'},
                'level_2': {'target_error': ..., 'name': 'Medium/Standard'},
                'level_3': {'target_error': ..., 'name': 'High/Scientific'}
            },
            'speed': {
                'fast': {'time_budget': ..., 'name': 'Real-time'},
                'medium': {'time_budget': ..., 'name': 'Interactive'},
                'slow': {'time_budget': ..., 'name': 'Batch'}
            }
        }
    """
    return {
        'accuracy': {
            'level_1': {
                'target_error': baseline_error * 100,
                'name': 'Low/Engineering'
            },
            'level_2': {
                'target_error': baseline_error * 1.0,
                'name': 'Medium/Standard'
            },
            'level_3': {
                'target_error': baseline_error * 0.01,
                'name': 'High/Scientific'
            }
        },
        'speed': {
            'fast': {
                'time_budget': baseline_time * 0.1,
                'name': 'Real-time'
            },
            'medium': {
                'time_budget': baseline_time * 1.0,
                'name': 'Interactive'
            },
            'slow': {
                'time_budget': baseline_time * 10.0,
                'name': 'Batch'
            }
        }
    }


def build_case(entry: Dict[str, Any], output_dir: Path, skip_oracle: bool = False):
    """
    构建单个 case
    
    Args:
        entry: 从 benchmark.jsonl 读取的条目
        output_dir: 输出目录 (cases/)
        skip_oracle: 是否跳过 Oracle 运行（用于快速测试）
    """
    case_id = entry['id']
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Building case: {case_id}")
    
    # Step 1: 运行 Oracle (或使用默认值)
    if not skip_oracle:
        baseline = run_oracle(entry['oracle_config'], case_id, entry)
    else:
        print(f"   ⚡ Skipping Oracle (using default baseline)")
        baseline = {'error': 1e-2, 'time': 10.0}
    
    # Step 2: 计算难度分级
    difficulty_tiers = calculate_difficulty_tiers(baseline['error'], baseline['time'])
    
    # Step 3: 构建完整 config.json
    full_config = {
        **entry,  # 包含所有源数据
        'baseline': {
            'error_ref': baseline['error'],
            'time_ref': baseline['time'],
            'description': 'Oracle baseline performance'
        },
        'difficulty_tiers': difficulty_tiers,
        'evaluation_config': {
            **entry['evaluation_config'],
            'target_error': difficulty_tiers['accuracy']['level_2']['target_error']  # 默认目标为 level_2
        }
    }
    
    with open(case_dir / 'config.json', 'w') as f:
        json.dump(full_config, f, indent=2)
    print(f"   ✅ config.json")
    
    # Step 4: 生成 description.md
    description = generate_description_md(
        entry,
        target_error=difficulty_tiers['accuracy']['level_2']['target_error'],
        difficulty_tiers=difficulty_tiers
    )
    with open(case_dir / 'description.md', 'w') as f:
        f.write(description)
    print(f"   ✅ description.md")
    
    # Step 5: 生成测试脚本
    for mode in ['fix_accuracy', 'fix_time']:
        script = generate_test_script(entry, mode)
        script_path = case_dir / f'test_{mode}.py'
        with open(script_path, 'w') as f:
            f.write(script)
        script_path.chmod(0o755)
        print(f"   ✅ test_{mode}.py")
    
    print(f"   ✨ Case built successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Build cases from benchmark.jsonl',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/benchmark.jsonl'),
        help='Path to benchmark.jsonl (default: data/benchmark.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('cases'),
        help='Output directory (default: cases/)'
    )
    
    parser.add_argument(
        '--cases',
        nargs='+',
        help='Build specific cases only (default: all)'
    )
    
    parser.add_argument(
        '--skip-oracle',
        action='store_true',
        help='Skip Oracle execution, use default baselines (faster for testing)'
    )
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    root_dir = Path(__file__).parent.parent
    data_file = root_dir / args.data
    output_dir = root_dir / args.output
    
    if not data_file.exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print(f"   Please run: python scripts/migrate_to_data.py first")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("🏗️  PDEBench Cases Builder")
    print(f"{'='*80}")
    print(f"📄 Data: {data_file}")
    print(f"📁 Output: {output_dir}")
    if args.skip_oracle:
        print(f"⚡ Mode: Fast (skipping Oracle)")
    print(f"{'='*80}")
    
    # 读取数据
    entries = []
    with open(data_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    print(f"\n📊 Found {len(entries)} cases in dataset")
    
    # 过滤 cases
    if args.cases:
        entries = [e for e in entries if e['id'] in args.cases]
        if not entries:
            print(f"❌ Error: None of the specified cases found")
            sys.exit(1)
        print(f"   Building {len(entries)} selected cases")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建所有 cases
    success_count = 0
    for entry in entries:
        try:
            build_case(entry, output_dir, args.skip_oracle)
            success_count += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully built {success_count}/{len(entries)} cases")
    print(f"{'='*80}\n")
    
    if success_count > 0:
        print("📖 Usage example:")
        example_case = entries[0]['id']
        print(f"   cd {output_dir}/{example_case}")
        print(f"   python test_fix_accuracy.py --agent-script /path/to/solver.py")


if __name__ == '__main__':
    main()

