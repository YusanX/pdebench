# Agent Task: Physics-to-Code Generation Benchmark (PDEBench)

## Goal
Solve the PDE problems described in the dataset by generating a **complete, executable Python script** for each case. Your script must use the `dolfinx` library to solve the problem and save the result.

## Workflow (Must Follow)
1.  **Read the Problem:** The system will provide you with a problem description (Prompt) from `datasets/level_2_1_basic.jsonl`.
2.  **Generate Code:** Write a Python script that solves the problem.
    *   **Strict Requirement:** Your script MUST accept command-line arguments `--resolution` (int) and `--degree` (int).
    *   **Strict Requirement:** Your script MUST save the solution to `solution.npz` containing arrays `x`, `y`, and `u` (interpolated on a regular grid defined by the problem bounds).
    *   **Library:** Use `dolfinx` (FEniCSx) and `petsc4py`.
    *   **Version Hint:** The environment uses `dolfinx` v0.8.0+.
    *   Use `dolfinx.geometry.bb_tree(domain, domain.topology.dim)` instead of `BoundingBoxTree(...)`.
    *   Use `dolfinx.geometry.compute_collisions_points` for point collision detection.
    *   Ensure `LinearProblem` initialization includes `petsc_options_prefix`.
3.  **Submit & Evaluate:** The system will run your script in a sandbox and compare your result against a hidden Oracle solution.

## Benchmark Execution Command
To run the benchmark using `swe-agent` (or similar tools calling this task), use the following command structure. This command runs the evaluation pipeline which acts as the harness for your generated code.

```bash
python scripts/evaluate_agent.py \
    --dataset datasets/level_2_1_basic.jsonl \
    --outdir results/agent_evaluation_$(date +%Y%m%d_%H%M%S) \
    --agent-script <PATH_TO_YOUR_GENERATED_SCRIPT>
```

**Note:** In a real "Code Agent" scenario, you (the Agent) are the one writing `<PATH_TO_YOUR_GENERATED_SCRIPT>`. The harness (`evaluate_agent.py`) will iteratively:
1.  Read a case from `--dataset`.
2.  Present the prompt to you (simulated or real).
3.  Take your code, save it to a file.
4.  Run it via `python your_script.py --resolution 32 --degree 1`.
5.  Validate the output `solution.npz`.

## Interactive Testing Mode (for You, the Agent)
If you want to test your ability to solve *one* specific case manually:
1.  Read the first line of `datasets/level_2_1_basic.jsonl` to get the Prompt.
2.  Write a script `my_solver.py`.
3.  Run it yourself: `python my_solver.py --resolution 32 --degree 1`.
4.  Check if `solution.npz` is generated.

## Success Criteria
*   **Executability:** The script runs without error.
*   **Accuracy:** Relative L2 error vs. Oracle < 0.05 (typically).
*   **Format:** The output `solution.npz` has the correct shape and keys.

