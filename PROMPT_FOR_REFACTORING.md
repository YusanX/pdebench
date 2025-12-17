# Code Agent for PDE Solving: Project Refactoring Plan (NeurIPS Target)

## 1. Project Vision: Redefining AI for Science Benchmarks
**Goal:** Create the *gold standard* benchmark for evaluating Large Language Models (LLMs) on end-to-end scientific modeling and simulation.
**Shift:** Move from low-level "Solver Auto-tuning" (HPC-focused) to high-level "Physics-to-Code Generation" (Reasoning-focused).

**Why NeurIPS?**
- **Novelty:** First benchmark to test if LLMs can "think like a computational physicist" (modeling -> weak form -> discretization -> validation).
- **Challenge:** Incorporates numerical stability awareness (e.g., SUPG for convection, LBB for Stokes) which breaks standard code agents.
- **Rigor:** Uses dual-source ground truth (Manufactured Solutions + Fine-Grid Oracles) and pareto-frontier evaluation (Accuracy vs. Cost).

---

## 2. The Task: "Physics-to-Simulation"
The Agent is given a **Problem Description** (natural language + constraints) and must generate a **Complete Python Script** using FEniCSx.

### Task Levels (Curriculum Learning)
1.  **Level 1: Operator Translation (Unit Test)**
    *   *Input:* "Implement the weak form of the Navier-Stokes momentum equation."
    *   *Output:* UFL (Unified Form Language) code snippet.
    *   *Metric:* Symbolic equivalence check.

2.  **Level 2: End-to-End Simulation (The Core)**
    *   **2.1 Basic:** Poisson / Heat (Linear, Symmetric). Test basic syntax & BCs.
    *   **2.2 Stability:** Convection-Diffusion (High Peclet) / Stokes. Test numerical awareness (SUPG, Mixed Elements).
    *   **2.3 Complex:** Navier-Stokes (Non-linear, Transient). Test operator splitting & convergence handling.

3.  **Level 3: Discovery (Optional/Future)**
    *   Inverse problems (parameter estimation) given observation data.

---

## 3. Dataset & Prompt Engineering
**Format:** `jsonl` datasets where each entry enforces an **Interface Contract**.

```json
{
  "id": "stokes_cavity_01",
  "level": "2.2",
  "prompt": "Solve the Stokes flow in a unit square... [Physics Desc].\n\nRequirements:\n1. Use dolfinx.\n2. Expose arguments `--resolution` and `--degree`.\n3. Save result to `solution.npz`.",
  "oracle_config": { ... } // Hidden config to generate Ground Truth
}
```

**Evaluation Schema:**
*   **Correctness:** Does it run? Does it conserve mass?
*   **Accuracy:** $L_2$ error vs. Oracle (Interpolated on reference grid).
*   **Pareto Efficiency:** Runtime vs. Accuracy curve (Agent must support variable resolution).

---

## 4. Architecture Refactoring Instructions (For Claude)

### Phase 1: Clean & Decouple (The "Oracle" Split)
*   **Action:** Move existing `pdebench/solvers/*.py` to `pdebench/oracle/`.
*   **Reason:** These are no longer library code for the agent; they are strictly for generating Ground Truth `truth.npz`.
*   **Refactor:** Ensure Oracle solvers accept a standard `case_spec` dict but are NOT importable by the Agent's generated code.

### Phase 2: Dataset Construction
*   **Action:** Create `pdebench/datasets/`.
*   **Task:** Write a script `scripts/build_dataset.py` that **mines existing Oracle configs from `pdebench/cases/demo/*.json`**.
    *   *Mechanism:* Use Jinja2 templates to "reverse engineer" natural language prompts from these JSON configs.
    *   *Example:* Convert `{"pde": "poisson", "bc": "u=0"}` -> "Solve Poisson equation with zero boundary conditions...".
*   **Output:** `datasets/level_2_1_poisson.jsonl` (generated from existing Poisson demos), `datasets/level_2_2_stokes.jsonl` (future).

### Phase 3: The Execution Sandbox
*   **Action:** Implement `pdebench/sandbox/executor.py`.
*   **Key Features:**
    *   **Isolation:** Run generated code in a subprocess (or Docker).
    *   **Resource Limits:** strict timeout (60s) and memory limit (4GB).
    *   **CLI Injection:** Automatically pass `--resolution 64` to Agent scripts to test scalability.

### Phase 4: Mesh-Agnostic Validator
*   **Action:** Implement `pdebench/evaluation/validator.py`.
*   **Challenge:** Agent grid != Oracle grid.
*   **Solution:** Use `dolfinx`'s `create_nonmatching_meshes_interpolation_data` to map Agent solution $u_h$ to Oracle function space $V_{ref}$.
*   **Metric:** Compute $\|u_h - u_{ref}\|_{L_2}$ on the Oracle's fine mesh.

---

## 5. Execution Roadmap

1.  **Step 0 (Setup):** Confirm FEniCSx environment is robust.
2.  **Step 1 (Refactor):** Move Solvers -> Oracle.
3.  **Step 2 (Data):** Generate `datasets/pilot_poisson.jsonl` (3-5 samples).
4.  **Step 3 (Pipeline):** Implement `scripts/evaluate_agent.py` that runs the full loop:
    `Prompt -> [Mock Agent] -> Script -> Sandbox -> Validator -> Score`.
5.  **Step 4 (Scale):** Add Stokes/Navier-Stokes oracles and prompts.

---

**Prompt for Claude:**
"Please adopt the role of a Senior Research Engineer. Review the 'Project Refactoring Plan' above. We are pivoting `pdebench` to be a NeurIPS-tier benchmark for AI-driven scientific coding. Start by **Refactoring the Directory Structure** (Phase 1) and establishing the **Dataset Schema** (Phase 2). Do not delete existing solvers, but encapsulate them as 'Oracles'. Let's begin with Step 1."

