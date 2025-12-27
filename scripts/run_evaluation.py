#!/usr/bin/env python3
"""
PDEBench Unified Evaluation Entry Point

This script provides a standardized interface for evaluating any agent
(code agents, LLMs, or manual submissions) on the PDEBench benchmark.

Usage:
    # Evaluate a single agent
    python run_evaluation.py \\
        --agent-name gpt-4 \\
        --agent-dir results/gpt-4 \\
        --output results/gpt-4/evaluation.json

    # Evaluate with specific modes
    python run_evaluation.py \\
        --agent-name claude \\
        --agent-dir results/claude \\
        --modes fix_accuracy fix_time \\
        --output results/claude/evaluation.json

    # Evaluate specific cases only
    python run_evaluation.py \\
        --agent-name gpt-4 \\
        --agent-dir results/gpt-4 \\
        --cases poisson_simple heat_simple \\
        --output results/gpt-4/evaluation.json

Expected directory structure:
    results/{agent_name}/
    ├── poisson_simple/
    │   └── solver.py
    ├── heat_grid_target/
    │   └── solver.py
    └── ...

Output format:
    evaluation.json contains:
    - agent_name
    - evaluation_date
    - summary statistics (pass rates, scores, tiers)
    - detailed results for each case and mode
"""

import argparse
import json
import sys
from pathlib import Path

# Add pdebench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.harness import BatchEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='PDEBench Unified Evaluation Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--agent-name',
        type=str,
        required=True,
        help='Name of the agent being evaluated (e.g., "gpt-4", "claude", "manual")'
    )
    
    parser.add_argument(
        '--agent-dir',
        type=Path,
        required=True,
        help='Directory containing agent submissions (e.g., results/gpt-4/)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output file path for evaluation results (JSON)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--cases-dir',
        type=Path,
        default=Path('cases'),
        help='Directory containing case definitions (default: cases/)'
    )
    
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['fix_accuracy', 'fix_time'],
        default=['fix_accuracy', 'fix_time'],
        help='Evaluation modes to run (default: both)'
    )
    
    parser.add_argument(
        '--cases',
        nargs='+',
        default=None,
        help='Specific case IDs to evaluate (default: all cases)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per case in seconds (default: 300)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.agent_dir.exists():
        print(f"❌ Error: Agent directory not found: {args.agent_dir}")
        sys.exit(1)
    
    if not args.cases_dir.exists():
        print(f"❌ Error: Cases directory not found: {args.cases_dir}")
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Print banner
    print("\n" + "="*80)
    print("PDEBench Unified Evaluation System")
    print("="*80)
    print(f"Agent: {args.agent_name}")
    print(f"Agent Directory: {args.agent_dir}")
    print(f"Cases Directory: {args.cases_dir}")
    print(f"Modes: {', '.join(args.modes)}")
    if args.cases:
        print(f"Cases: {', '.join(args.cases)}")
    else:
        print("Cases: all")
    print(f"Timeout: {args.timeout}s per case")
    print(f"Output: {args.output}")
    print("="*80 + "\n")
    
    # Create evaluator
    try:
        evaluator = BatchEvaluator(args.cases_dir)
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluator.evaluate_agent(
            agent_name=args.agent_name,
            agent_dir=args.agent_dir,
            modes=args.modes,
            case_filter=args.cases,
            timeout_sec=args.timeout
        )
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to: {args.output}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        sys.exit(1)
    
    # Print final status
    summary = results['summary']
    total_passed = sum(
        summary.get(mode, {}).get('passed', 0)
        for mode in args.modes
    )
    total_evaluated = sum(
        summary.get(mode, {}).get('submitted', 0)
        for mode in args.modes
    )
    
    print("\n" + "="*80)
    print("✅ Evaluation Complete")
    print("="*80)
    print(f"Total Evaluated: {total_evaluated}")
    print(f"Total Passed: {total_passed}")
    print(f"Overall Pass Rate: {total_passed/total_evaluated*100:.1f}%" if total_evaluated > 0 else "N/A")
    print("="*80 + "\n")
    
    # Exit with appropriate code
    if total_passed == total_evaluated and total_evaluated > 0:
        sys.exit(0)  # Perfect score
    elif total_passed > 0:
        sys.exit(0)  # Partial success
    else:
        sys.exit(1)  # All failed


if __name__ == '__main__':
    main()

