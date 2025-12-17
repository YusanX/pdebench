# 大模型集成指南

本指南说明如何使用真实的大型语言模型（如 GPT-4、Claude、DeepSeek 等）通过 SWE-agent 或其他框架来测试 PDEBench。

## 方案 1: 使用 SWE-agent（推荐）

### 1.1 安装 SWE-agent

```bash
# 在 pdebench 目录外克隆 SWE-agent
cd /Users/yusan/agent
git clone https://github.com/princeton-nlp/SWE-agent.git
cd SWE-agent
pip install -e .
```

### 1.2 创建 PDEBench 任务适配器

为 SWE-agent 创建任务描述文件：

```bash
cd /Users/yusan/agent/pdebench
python scripts/create_swe_tasks.py --dataset datasets/level_2_1_basic.jsonl --output swe_tasks/
```

这将生成结构如下的任务文件：

```json
{
  "instance_id": "pdebench__poisson_simple",
  "repo": "pdebench",
  "base_commit": "main",
  "problem_statement": "Solve the Poisson equation...",
  "hints_text": "",
  "test_patch": "",
  "version": "1.0"
}
```

### 1.3 配置 LLM API

```bash
# 设置 API 密钥
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export DEEPSEEK_API_KEY="your-key-here"
```

### 1.4 运行 SWE-agent

```bash
cd /Users/yusan/agent/SWE-agent

# 使用 GPT-4
python run.py \
  --model_name gpt4 \
  --data_path /Users/yusan/agent/pdebench/swe_tasks/poisson_simple.json \
  --config_file config/default.yaml \
  --output_dir /Users/yusan/agent/pdebench/results/swe_gpt4

# 使用 Claude
python run.py \
  --model_name claude-3-opus-20240229 \
  --data_path /Users/yusan/agent/pdebench/swe_tasks/poisson_simple.json \
  --output_dir /Users/yusan/agent/pdebench/results/swe_claude
```

### 1.5 提取 Agent 生成的代码

SWE-agent 会生成完整的代码修改，我们需要提取生成的求解器脚本：

```bash
python scripts/extract_swe_output.py \
  --swe-output results/swe_gpt4/trajectories/poisson_simple \
  --output-script generated_solvers/gpt4_poisson_simple.py
```

### 1.6 评估生成的代码

```bash
python scripts/evaluate_agent.py \
  --dataset datasets/level_2_1_basic.jsonl \
  --agent-script generated_solvers/gpt4_poisson_simple.py \
  --outdir results/eval_gpt4
```

## 方案 2: 直接 API 调用（更灵活）

### 2.1 创建 LLM 调用包装器

```bash
cd /Users/yusan/agent/pdebench
python scripts/run_llm_benchmark.py \
  --dataset datasets/level_2_1_basic.jsonl \
  --model gpt-4 \
  --provider openai \
  --outdir results/llm_gpt4_direct \
  --limit 5
```

这个脚本会：
1. 从数据集读取每个问题
2. 构造 prompt 发送给 LLM
3. 解析 LLM 返回的代码
4. 自动执行和评估

### 2.2 配置文件示例

创建 `configs/llm_providers.yaml`：

```yaml
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    models:
      - gpt-4
      - gpt-4-turbo
      - gpt-3.5-turbo
    
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    models:
      - claude-3-opus-20240229
      - claude-3-sonnet-20240229
  
  deepseek:
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com/v1
    models:
      - deepseek-coder

prompt_template: |
  You are an expert computational physicist. Your task is to implement a finite element solver using FEniCSx (dolfinx).
  
  {problem_statement}
  
  Please generate a complete, runnable Python script that:
  {requirements}
  
  Output only the Python code, enclosed in ```python code blocks.
```

## 方案 3: 批量评估多个模型

### 3.1 创建批量评估脚本

```bash
python scripts/batch_evaluate_llms.py \
  --dataset datasets/level_2_1_basic.jsonl \
  --models gpt-4 claude-3-opus deepseek-coder \
  --outdir results/multi_model_comparison \
  --runs-per-model 3
```

### 3.2 查看对比结果

```bash
python scripts/compare_results.py \
  --result-dirs results/llm_gpt4 results/llm_claude results/llm_deepseek \
  --output results/comparison_report.html
```

生成的报告包含：
- 各模型成功率对比
- 精度分布图
- 运行时间分析
- 代码质量评估

## 实现细节

### 创建 SWE-agent 任务生成器

创建 `scripts/create_swe_tasks.py`：

```python
#!/usr/bin/env python3
"""Convert PDEBench dataset to SWE-agent task format."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pdebench.datasets.schema import load_dataset


def create_swe_task(entry, repo_path):
    """Convert a dataset entry to SWE-agent task format."""
    
    # 构造问题陈述
    problem_statement = f"""# PDEBench Task: {entry.id}

{entry.prompt}

## Technical Requirements
{chr(10).join(f"- {req}" for req in entry.requirements)}

## Expected Output Files
1. `agent_solver.py` - Your implementation
2. The script should accept: --resolution N --degree P --outdir DIR
3. Output files: solution.npz and meta.json

## Workspace Setup
Your code will be tested in an isolated environment with:
- Python 3.10
- FEniCSx (dolfinx) installed
- PETSc and MPI available

## Validation
Your solution will be compared against a reference solution using relative L2 error.
Target: {entry.evaluation_config.get('target_metric')} ≤ {entry.evaluation_config.get('target_error')}
"""
    
    return {
        "instance_id": f"pdebench__{entry.id}",
        "repo": str(repo_path.absolute()),
        "base_commit": "HEAD",
        "problem_statement": problem_statement,
        "hints_text": "",
        "created_at": "2024-12-17",
        "version": "1.0",
        "FAIL_TO_PASS": ["test_solution_accuracy"],
        "PASS_TO_PASS": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-path', type=Path, default=Path.cwd())
    args = parser.parse_args()
    
    # Load dataset
    entries = load_dataset(str(args.dataset))
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Convert each entry
    for entry in entries:
        task = create_swe_task(entry, args.repo_path)
        
        output_file = args.output / f"{entry.id}.json"
        with open(output_file, 'w') as f:
            json.dump(task, f, indent=2)
        
        print(f"✓ Created: {output_file}")
    
    print(f"\n✅ Generated {len(entries)} SWE-agent tasks")


if __name__ == '__main__':
    main()
```

### 创建直接 LLM 调用脚本

创建 `scripts/run_llm_benchmark.py`：

```python
#!/usr/bin/env python3
"""Run LLM benchmark with direct API calls."""

import argparse
import json
import os
import re
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.datasets.schema import load_dataset
from pdebench.sandbox.executor import execute_agent_script_with_oracle
from pdebench.evaluation.validator import validate_solution


def call_llm(prompt, model, provider):
    """Call LLM API and return generated code."""
    
    if provider == 'openai':
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in computational physics and finite element methods."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
    
    elif provider == 'anthropic':
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def extract_code(llm_response):
    """Extract Python code from LLM response."""
    # Find code blocks
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    if matches:
        return matches[0]
    
    # Fallback: return entire response
    return llm_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--provider', choices=['openai', 'anthropic'], required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    
    # Load dataset
    entries = load_dataset(str(args.dataset))
    if args.limit:
        entries = entries[:args.limit]
    
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, entry in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] Processing: {entry.id}")
        
        case_dir = args.outdir / entry.id
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Call LLM
        print("  📡 Calling LLM...")
        try:
            llm_response = call_llm(entry.prompt, args.model, args.provider)
            code = extract_code(llm_response)
            
            # Save generated code
            script_path = case_dir / 'generated_solver.py'
            with open(script_path, 'w') as f:
                f.write(code)
            
            # Save full response
            with open(case_dir / 'llm_response.txt', 'w') as f:
                f.write(llm_response)
            
            print(f"  ✓ Generated code ({len(code)} chars)")
            
            # Execute and evaluate
            print("  🔧 Executing...")
            exec_result, agent_out, oracle_out = execute_agent_script_with_oracle(
                script_path=script_path,
                oracle_config=entry.oracle_config,
                base_outdir=case_dir,
                evaluation_config=entry.evaluation_config
            )
            
            if exec_result.success:
                print("  ✓ Execution successful")
                
                validation = validate_solution(agent_out, oracle_out, entry.evaluation_config)
                print(f"  {'✓' if validation.is_valid else '✗'} {validation.reason}")
                
                result = {
                    'case_id': entry.id,
                    'success': validation.is_valid,
                    'execution': exec_result.to_dict(),
                    'validation': validation.to_dict()
                }
            else:
                print(f"  ✗ Execution failed: {exec_result.error_message}")
                result = {
                    'case_id': entry.id,
                    'success': False,
                    'execution': exec_result.to_dict(),
                    'validation': None
                }
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            result = {
                'case_id': entry.id,
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
        
        # Save intermediate results
        with open(case_dir / 'result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        # Rate limiting
        time.sleep(1)
    
    # Generate summary
    summary = {
        'model': args.model,
        'provider': args.provider,
        'total_cases': len(results),
        'successful': sum(1 for r in results if r.get('success', False)),
        'results': results
    }
    
    with open(args.outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Completed: {summary['successful']}/{summary['total_cases']} passed")


if __name__ == '__main__':
    main()
```

## 运行示例

```bash
# 1. 生成 SWE-agent 任务
python scripts/create_swe_tasks.py \
  --dataset datasets/level_2_1_basic.jsonl \
  --output swe_tasks/

# 2. 直接 API 调用测试
export OPENAI_API_KEY="sk-..."
python scripts/run_llm_benchmark.py \
  --dataset datasets/level_2_1_basic.jsonl \
  --model gpt-4 \
  --provider openai \
  --outdir results/gpt4_test \
  --limit 2

# 3. 查看结果
cat results/gpt4_test/summary.json | jq '.successful, .total_cases'
```

## 预期结果

不同模型的预期表现：

| 模型 | Level 2.1 预期通过率 | Level 2.2 预期通过率 |
|------|---------------------|---------------------|
| GPT-4 | 70-90% | 30-50% |
| Claude-3-Opus | 60-80% | 20-40% |
| GPT-3.5 | 30-50% | 5-15% |
| DeepSeek-Coder | 40-60% | 10-25% |

Level 2.2 难度较高，因为需要识别数值稳定性问题（如高 Péclet 数需要 SUPG）。

