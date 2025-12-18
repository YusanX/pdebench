# Task: Solve PDE Benchmark Cases

You are an expert computational physicist participating in the PDEBench challenge. Your goal is to achieve the highest possible accuracy and efficiency on a series of PDE problems.

## 🏆 Evaluation Metrics (Leaderboard)

Your solution will be evaluated on three criteria (in order of importance):

1. **Pass Rate (Pass@ε)**: Success rate at strict accuracy thresholds (1e-4, 1e-3, 1e-2).
2. **Efficiency (Time)**: Total execution time for correct solutions.
3. **Accuracy (Error)**: Average relative L2 error.

**Success Criteria:**
- **Pass@1e-4**: Elite (Machine Precision)
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
- Accepts standard CLI arguments: `--resolution N`, `--degree P`, `--outdir DIR`.
- **Hard-codes** the physics parameters from the prompt.
- **Interpolates** the solution onto the specified grid (nx × ny).
- Saves outputs: `solution.npz` (x, y, u) and `meta.json`.

**Critical Requirement:** The solver must separate **Model Setup** time from **Solve** time to ensure fair efficiency comparison.

### 4. Self-Correction Loop (Max 5 Attempts PER CASE)

You have a maximum of **5 attempts** for **EACH** case. The counter resets when you move to a new case.

1. **Research & Implement**: 
   - If previous attempt failed on API error: **Re-read `DOLFINX_GUIDE.md`** or search online (if available).
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
