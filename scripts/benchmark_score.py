#!/usr/bin/env python3
"""
PDEBench 性能评估脚本

用于量化求解器性能，主要指标：
1. 总耗时（越低越好）
2. 通过率（必须 100%）
3. 平均迭代次数

用法：
    python scripts/benchmark_score.py [--output report.json] [--keep-artifacts]
    python scripts/benchmark_score.py --log-history --experiment-id "gpt4_run1"
"""
import sys
import json
import subprocess
import time
import argparse
from pathlib import Path
import shutil
import datetime
import numpy as np


def run_benchmark(keep_artifacts=False, output_file=None, log_history=False, experiment_id=None):
    """运行完整的 benchmark 套件"""
    repo_root = Path(__file__).parent.parent
    demo_dir = repo_root / "cases" / "demo"
    cases = list(demo_dir.glob("*.json"))
    
    # 按文件名排序，保证顺序一致
    cases.sort()
    
    print("=" * 80)
    print("🚀 PDEBench 性能评估")
    print("=" * 80)
    print(f"测试用例数量: {len(cases)}")
    print(f"项目路径: {repo_root}")
    print("=" * 80)
    print()
    
    # 表头
    print(f"{'Case ID':<25} | {'状态':<8} | {'耗时(s)':<10} | {'迭代':<8} | {'备注'}")
    print("-" * 80)
    
    results = []
    total_wall_time = 0.0
    total_iters = 0
    passed_cases = 0
    failed_cases = []
    
    # 结果输出目录
    artifacts_dir = repo_root / "artifacts_bench"
    if artifacts_dir.exists() and not keep_artifacts:
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    for case in cases:
        case_id = case.stem
        outdir = artifacts_dir / case_id
        
        # 运行 CLI run 命令
        cmd = [
            sys.executable, "-m", "pdebench.cli", "run",
            str(case),
            "--outdir", str(outdir)
        ]
        
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=repo_root,
            timeout=60  # 单个 case 最多 60 秒
        )
        elapsed_time = time.time() - start_time
        
        # 初始化结果记录
        case_result = {
            "case_id": case_id,
            "status": "UNKNOWN",
            "wall_time": 10.0,  # 默认惩罚时间
            "iters": -1,
            "note": "",
            "metrics": {}
        }
        
        if result.returncode == 0:
            try:
                # 读取 metrics.json
                metrics_file = outdir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                    
                    case_result["metrics"] = metrics
                    
                    if metrics["validity"]["pass"]:
                        case_result["status"] = "PASS"
                        case_result["wall_time"] = metrics["cost"]["wall_time_sec"]
                        case_result["iters"] = metrics["cost"]["iters"]
                        case_result["note"] = f"res={metrics.get('rel_res', -1):.2e}"
                        
                        passed_cases += 1
                        total_wall_time += case_result["wall_time"]
                        total_iters += case_result["iters"]
                    else:
                        case_result["status"] = "FAIL"
                        case_result["note"] = metrics["validity"]["reason"][:40]
                        failed_cases.append(case_id)
                else:
                    case_result["status"] = "NO_METRICS"
                    case_result["note"] = "metrics.json not found"
                    failed_cases.append(case_id)
            except Exception as e:
                case_result["status"] = "ERROR"
                case_result["note"] = str(e)[:40]
                failed_cases.append(case_id)
        else:
            # 运行崩溃
            case_result["status"] = "CRASH"
            # 提取最后一行错误信息
            stderr_lines = result.stderr.strip().split('\n')
            if stderr_lines:
                # 找到最后一个非空行
                for line in reversed(stderr_lines):
                    if line.strip():
                        case_result["note"] = line.strip()[-40:]
                        break
            else:
                case_result["note"] = "Unknown error"
            failed_cases.append(case_id)
        
        results.append(case_result)
        
        # 打印结果行
        status_emoji = "✅" if case_result["status"] == "PASS" else "❌"
        print(f"{status_emoji} {case_result['case_id']:<23} | "
              f"{case_result['status']:<8} | "
              f"{case_result['wall_time']:<10.4f} | "
              f"{case_result['iters']:<8} | "
              f"{case_result['note']}")
    
    print("-" * 80)
    print()
    
    # 汇总统计
    success_rate = passed_cases / len(cases) * 100
    avg_iters = total_iters / passed_cases if passed_cases > 0 else 0
    
    print("=" * 80)
    print("🏆 最终得分摘要")
    print("=" * 80)
    print(f"📊 总耗时 (越低越好):  {total_wall_time:.4f} 秒")
    print(f"✓  通过率:             {passed_cases}/{len(cases)} ({success_rate:.1f}%)")
    print(f"🔄 平均迭代次数:       {avg_iters:.1f}")
    
    if failed_cases:
        print(f"❌ 失败的 Cases:       {', '.join(failed_cases)}")
    
    print("=" * 80)
    
    # 保存详细报告到 JSON
    report = {
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "failed_cases": len(failed_cases),
            "success_rate": success_rate,
            "total_wall_time": total_wall_time,
            "avg_iters": avg_iters,
        },
        "failed_list": failed_cases,
        "details": results
    }
    
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 详细报告已保存到: {output_path}")
    
    # 记录到历史日志（用于绘图和追踪）
    if log_history:
        log_experiment_step(
            results=results,
            total_wall_time=total_wall_time,
            passed_cases=passed_cases,
            total_cases=len(cases),
            avg_iters=avg_iters,
            experiment_id=experiment_id,
            repo_root=repo_root
        )
    
    # 返回状态码
    if failed_cases:
        print("\n⚠️  有测试失败，返回状态码 1")
        return 1
    else:
        print("\n🎉 所有测试通过！")
        return 0


def log_experiment_step(results, total_wall_time, passed_cases, total_cases, avg_iters, experiment_id, repo_root):
    """将实验结果追加到历史日志文件"""
    
    # 计算平均相对误差（几何平均，对 log-scale 更合理）
    rel_errors = []
    for r in results:
        if r["status"] == "PASS" and "rel_L2_fe" in r["metrics"]:
            rel_errors.append(r["metrics"]["rel_L2_fe"])
    
    if rel_errors:
        # 几何平均：exp(mean(log(errors)))
        avg_rel_error = np.exp(np.mean(np.log(rel_errors)))
    else:
        avg_rel_error = float('inf')
    
    # 生成实验 ID
    if experiment_id is None:
        experiment_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 构建日志条目
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "summary": {
            "total_wall_time": total_wall_time,
            "avg_rel_error": avg_rel_error,
            "pass_rate": passed_cases / total_cases,
            "passed_cases": passed_cases,
            "total_cases": total_cases,
            "avg_iters": avg_iters
        },
        "per_case": {
            r["case_id"]: {
                "status": r["status"],
                "wall_time": r["wall_time"],
                "iters": r["iters"],
                "rel_L2_fe": r["metrics"].get("rel_L2_fe", None) if r["status"] == "PASS" else None,
                "rel_res": r["metrics"].get("rel_res", None) if r["status"] == "PASS" else None
            }
            for r in results
        }
    }
    
    # 追加到 JSONL 文件
    log_file = repo_root / "experiment_history.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"\n📈 实验记录已追加到: {log_file}")
    print(f"   实验 ID: {experiment_id}")
    print(f"   平均相对误差: {avg_rel_error:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="PDEBench 性能评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", "-o",
        help="保存详细报告的 JSON 文件路径",
        default=None
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="保留 artifacts_bench 目录（不清理旧数据）"
    )
    parser.add_argument(
        "--log-history",
        action="store_true",
        help="将结果追加到 experiment_history.jsonl 用于后续绘图分析"
    )
    parser.add_argument(
        "--experiment-id",
        help="实验标识符（用于区分不同的优化尝试），默认自动生成时间戳",
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = run_benchmark(
            keep_artifacts=args.keep_artifacts,
            output_file=args.output,
            log_history=args.log_history,
            experiment_id=args.experiment_id
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()

