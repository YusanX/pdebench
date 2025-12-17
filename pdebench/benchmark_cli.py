"""Command-line interface for the PDE Benchmark system.

This CLI provides commands for:
- Building datasets from Oracle cases
- Evaluating agent scripts
- Generating agent script templates
"""

import argparse
import sys
from pathlib import Path


def cmd_build_dataset(args):
    """Build dataset from Oracle cases."""
    from pdebench.scripts.build_dataset import main as build_main
    
    # Override sys.argv for the build script
    sys.argv = [
        'build_dataset.py',
        '--cases-dir', str(args.cases_dir),
        '--output', str(args.output),
    ]
    
    if args.filter_level:
        sys.argv.extend(['--filter-level', args.filter_level])
    if args.filter_pde:
        sys.argv.extend(['--filter-pde', args.filter_pde])
    
    # Import and call the build script
    import subprocess
    result = subprocess.run([
        sys.executable,
        str(Path(__file__).parent.parent / 'scripts' / 'build_dataset.py'),
        '--cases-dir', str(args.cases_dir),
        '--output', str(args.output),
    ] + (['--filter-level', args.filter_level] if args.filter_level else [])
      + (['--filter-pde', args.filter_pde] if args.filter_pde else []))
    
    sys.exit(result.returncode)


def cmd_evaluate(args):
    """Evaluate agent on benchmark."""
    import subprocess
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / 'scripts' / 'evaluate_agent.py'),
        '--dataset', str(args.dataset),
        '--outdir', str(args.outdir),
    ]
    
    if args.agent_script:
        cmd.extend(['--agent-script', str(args.agent_script)])
    if args.mock_agent:
        cmd.append('--mock-agent')
    if args.limit:
        cmd.extend(['--limit', str(args.limit)])
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def cmd_template(args):
    """Generate agent script template."""
    from pdebench.datasets.schema import load_dataset
    from pdebench.sandbox.executor import create_agent_script_template
    
    # Load dataset to get the entry
    entries = load_dataset(str(args.dataset))
    
    # Find the specific case
    entry = None
    for e in entries:
        if e.id == args.case_id:
            entry = e
            break
    
    if entry is None:
        print(f"Error: Case '{args.case_id}' not found in dataset")
        sys.exit(1)
    
    # Generate template
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_agent_script_template(
        prompt=entry.prompt,
        requirements=entry.requirements,
        output_path=output_path
    )
    
    print(f"✓ Generated template script: {output_path}")
    print(f"\nTo implement the solver, edit the script and fill in the TODO sections.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PDEBench: AI-driven Scientific Coding Benchmark',
        prog='python -m pdebench.benchmark_cli'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    subparsers.required = True
    
    # Build dataset command
    parser_build = subparsers.add_parser(
        'build-dataset',
        help='Build benchmark dataset from Oracle cases'
    )
    parser_build.add_argument(
        '--cases-dir',
        type=Path,
        default=Path('cases/demo'),
        help='Directory containing Oracle case JSON files'
    )
    parser_build.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file path'
    )
    parser_build.add_argument(
        '--filter-level',
        type=str,
        help='Only include cases of specified level (e.g., "2.1")'
    )
    parser_build.add_argument(
        '--filter-pde',
        type=str,
        help='Only include cases of specified PDE type'
    )
    parser_build.set_defaults(func=cmd_build_dataset)
    
    # Evaluate command
    parser_eval = subparsers.add_parser(
        'evaluate',
        help='Evaluate agent on benchmark dataset'
    )
    parser_eval.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to dataset JSONL file'
    )
    parser_eval.add_argument(
        '--agent-script',
        type=Path,
        help='Path to agent script'
    )
    parser_eval.add_argument(
        '--mock-agent',
        action='store_true',
        help='Use mock agent (Oracle solver) for testing'
    )
    parser_eval.add_argument(
        '--outdir',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    parser_eval.add_argument(
        '--limit',
        type=int,
        help='Limit number of cases to evaluate'
    )
    parser_eval.set_defaults(func=cmd_evaluate)
    
    # Template command
    parser_template = subparsers.add_parser(
        'template',
        help='Generate agent script template for a case'
    )
    parser_template.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to dataset JSONL file'
    )
    parser_template.add_argument(
        '--case-id',
        type=str,
        required=True,
        help='Case ID to generate template for'
    )
    parser_template.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output script path'
    )
    parser_template.set_defaults(func=cmd_template)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

