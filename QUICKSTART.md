# PDEBench 快速入门指南

## 5 分钟上手

### 1. 安装环境

```bash
conda create -n pdebench python=3.10
conda activate pdebench
conda install -c conda-forge fenics-dolfinx mpich petsc4py
cd pdebench && pip install -e .
```

### 2. 生成数据集

```bash
# 生成基础难度数据集（11 个案例）
python scripts/build_dataset.py \
    --output datasets/level_2_1_basic.jsonl \
    --filter-level "2.1"
```

### 3. 测试系统

```bash
# 使用 Mock Agent 验证系统（应该 100% 通过）
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
    --mock-agent \
    --outdir results/test \
    --limit 2
```

预期看到：
```
✓ Agent execution: Success
✓ Validation: Pass
  rel_L2_grid=4.718e-14 ≤ target=1.000e-02
```

## 查看数据集

```bash
# 查看一个问题描述
python -c "
import json
with open('datasets/level_2_1_basic.jsonl') as f:
    entry = json.loads(f.readline())
    print(entry['prompt'])
"
```

## 实现你的第一个 Agent 求解器

### 问题描述示例

```
Solve the Poisson equation on a unit square domain [0,1]×[0,1]:

  -∇·(κ ∇u) = f   in Ω
  u = g           on ∂Ω

**Requirements:**
1. Use dolfinx (FEniCSx)
2. Accept --resolution N and --degree P arguments
3. Save solution to solution.npz with fields: x, y, u
4. Save metadata to meta.json
```

### 最小实现模板

```python
#!/usr/bin/env python3
import argparse
import json
import numpy as np
from dolfinx import mesh, fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--degree', type=int, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    
    # 1. 创建网格
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, args.resolution, args.resolution
    )
    
    # 2. 定义函数空间
    V = fem.functionspace(msh, ("Lagrange", args.degree))
    
    # 3. 定义变分问题
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    
    # 源项和系数
    kappa = fem.Constant(msh, 1.0)
    f = 2 * np.pi**2 * ufl.sin(np.pi*x[0]) * ufl.sin(np.pi*x[1])
    
    # 双线性形式和线性形式
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # 4. 边界条件
    u_exact = lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact)
    
    facets = mesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 5. 求解
    problem = fem_petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "jacobi"}
    )
    uh = problem.solve()
    
    # 6. 输出
    x_grid = np.linspace(0, 1, 100)
    y_grid = np.linspace(0, 1, 100)
    u_grid = np.zeros((100, 100))
    
    for j, yv in enumerate(y_grid):
        for i, xv in enumerate(x_grid):
            point = np.array([[xv, yv, 0.0]])
            u_grid[j, i] = uh.eval(point, [0])[0]
    
    np.savez(f"{args.outdir}/solution.npz", x=x_grid, y=y_grid, u=u_grid)
    
    with open(f"{args.outdir}/meta.json", 'w') as f:
        json.dump({
            'wall_time_sec': 0.0,
            'solver_info': {'ksp_type': 'cg', 'pc_type': 'jacobi', 'iters': 0}
        }, f)

if __name__ == '__main__':
    main()
```

### 测试你的求解器

```bash
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
    --agent-script my_solver.py \
    --outdir results/my_first_run \
    --limit 1
```

## 常见问题

### Q: 如何查看详细的错误信息？

检查 `results/case_id/result.json`：
```bash
cat results/my_first_run/poisson_simple/result.json | jq .
```

### Q: 如何调试 Agent 脚本？

手动运行 Agent 脚本：
```bash
python my_solver.py --resolution 32 --degree 1 --outdir test_output
ls test_output/  # 应该看到 solution.npz 和 meta.json
```

### Q: Mock Agent 是如何工作的？

Mock Agent 直接调用 Oracle 求解器，用于验证评估流程本身。它的误差应该接近机器精度（~1e-14）。

## 下一步

- 阅读完整 [README.md](README.md) 了解架构细节
- 查看 [PROMPT_FOR_REFACTORING.md](PROMPT_FOR_REFACTORING.md) 了解设计理念
- 尝试 Level 2.2（稳定性挑战）案例

## 需要帮助？

- 查看 Oracle 求解器实现：`pdebench/oracle/solvers/`
- 参考生成的数据集：`datasets/*.jsonl`
- 检查评估报告：`results/*/summary.json`

