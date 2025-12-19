# Task: Solve PDE Benchmark Cases

You are an expert computational physicist participating in the PDEBench challenge. Your goal is to achieve the highest possible accuracy and efficiency on a series of PDE problems.

## 🏆 Evaluation Metrics (Leaderboard)

Your solution will be evaluated on a **Pareto Frontier** basis. We do NOT just look at a single run; we sweep your solver across different resolutions (N=16..128) and degrees (P=1..2).

**Priorities:**
1.  **Convergence Capability (MUST HAVE)**: Your solver **MUST** be able to reach high precision (Relative Error < 1e-4) when provided with fine meshes. If your algorithm hits an "accuracy ceiling" (e.g., error stays at 1e-2 even with N=128), it will be marked as **FAILED**.
2.  **Pareto Efficiency**: Among valid high-precision solvers, the one that runs faster is better.

**Success Criteria:**
- **Pass@1e-4**: Elite (Machine Precision) - *Target Goal*
- **Pass@1e-3**: Excellent (Engineering Precision)
- **Pass@1e-2**: Good (Basic Pass)

## 🛠️ Workflow

### 1. 📚 Step 0: Knowledge Retrieval (MANDATORY)

Before writing any code, you **MUST** read the file `DOLFINX_GUIDE.md` located in the root directory.
This file contains **CRITICAL** syntax patterns for `dolfinx v0.10.0` that differ from older versions.

**Key things to look for in the guide:**
- How to perform **Point Evaluation** (Interpolation) correctly without crashing.
- How to create `FunctionSpace` and `BoundingBoxTree` using the new API.

### 2. Read the Problem
Read the first case from `datasets/level_2_1_basic.jsonl`. Extract the `prompt` field which contains:
- PDE type (Poisson, Heat, etc.)
- Domain and boundary conditions
- Coefficients (kappa, source term f)
- **Required Output Grid** (nx, ny)

### 3. Implement Solver (`agent_solver.py`)
Create a Python script that:
- Uses `dolfinx` (FEniCSx) **v0.10.0** for FEM implementation.
- **MANDATORY CLI INTERFACE**:
  - Must use `argparse` to accept: `--resolution N` (int), `--degree P` (int), `--outdir DIR` (str).
  - **CRITICAL**: Pass `args.resolution` DIRECTLY to mesh generation (e.g., `mesh.create_unit_square(..., args.resolution, args.resolution)`).
  - **CRITICAL**: Pass `args.degree` DIRECTLY to function space (e.g., `FunctionSpace(mesh, ("CG", args.degree))`).
  - **DO NOT** hard-code these values! Your solver will be externally swept to generate Pareto fronts.
- **CRITICAL - Physics Parameters**:
  - **You MUST extract ALL physics parameters from the problem description** (kappa, source term f, boundary conditions, initial conditions, T_final, dt, etc.).
  - **DO NOT invent or guess parameters**. If the prompt says `kappa=0.5`, you write `kappa=0.5`. If it says `T_final=0.2`, you write `T_final=0.2`.
- **Ensures Stability**: 
  - For time-dependent problems, ensure your `dt` is linked to `args.resolution` (CFL condition) or use an unconditional stable scheme (e.g., Backward Euler).
  - For convection-dominated problems, consider stabilization (e.g., SUPG) to avoid oscillations that ruin high-precision targets.
- **Interpolates** the solution onto the specified grid (nx × ny).
- Saves outputs: `solution.npz` (x, y, u) and `meta.json`.

**Critical Requirement:** The solver must separate **Model Setup** time from **Solve** time to ensure fair efficiency comparison.

### 4. Self-Correction Loop (Max 5 Attempts PER CASE)

You have a maximum of **5 attempts** for **EACH** case. The counter resets when you move to a new case.

1. **Research & Implement**: 
   - If previous attempt failed on API error: **Re-read `DOLFINX_GUIDE.md`** or locate the error and fix it.
   - Write or update `agent_solver.py`.

2. **Evaluate**: Run the evaluation command:
```bash
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
       --agent-script agent_solver.py \
       --outdir results/YOUR_MODEL_NAME \
       --limit 1
   ```
   *(Replace `YOUR_MODEL_NAME` with your actual model name, e.g., `gpt-4`, `claude-3`)*

3. **Analyze**: Check the result table.
   - ❌ **Fail**: Read the error message, fix the code, and retry (Attempt N+1).
   - ✅ **Pass (Low Precision)**: If `Pass@1e-2`, try to optimize mesh/degree to reach `Pass@1e-4` (if attempts remain).
   - 🌟 **Pass (High Precision)**: If `Pass@1e-4`, you are done! Move to the next case.

**STOP CONDITION**: 
- If you reach **Attempt #5** in each case and still fail, **STOP** trying this case. Mark it as FAILED and proceed to the next case. 
- Do NOT loop indefinitely on a single hard case.

## 📝 Output Directory Structure

Your results will be saved automatically to:
`results/YOUR_MODEL_NAME/CASE_ID/`

Example: `results/gemini-pro/heat_grid_target/`

## 🚫 Common Pitfalls
- **Version Mismatch**: You MUST use **dolfinx v0.10.0**. 
- **BoundingBoxTree**: In v0.8+, do NOT use `BoundingBoxTree(mesh, dim)`. Use `dolfinx.geometry.bb_tree(mesh, dim)`.
- **Legacy Syntax**: Do NOT use `dolfin` (old FEniCS).
- **Grid Mismatch**: You MUST interpolate the solution to the exact grid specified in the prompt.
- **Dependency Hell**: Assume `dolfinx`, `petsc4py`, `mpi4py`, `numpy` are installed. Do not install new packages.
