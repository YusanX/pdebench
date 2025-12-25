"""
PDEBench Case-Driven Testing Framework (SWE-bench Style)

标准化的测试接口，每个case使用独立的测试脚本。
包含所有 PDE 类型的测试基类。
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from abc import ABC, abstractmethod
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pdebench.sandbox.executor import execute_agent_script
from pdebench.oracle import generate


class BaseCaseTest(ABC):
    """
    案例测试基类 - 每个case继承此类实现独立的测试逻辑
    
    核心设计原则：
    1. 单一指标优化：每个test只优化一个目标（时间或精度），固定其他参数
    2. 独立性：每个case有独立的测试脚本，互不干扰
    3. 标准化：所有同类PDE遵循相同的API接口
    """
    
    def __init__(self, case_dir: Path, agent_dir: Optional[Path] = None):
        self.case_dir = Path(case_dir)
        self.config = self._load_config()
        self.case_id = self.config['id']
        
        # 如果提供了agent_dir，结果保存到agent目录下；否则保存到case目录下
        if agent_dir:
            self.output_dir = agent_dir / self.case_id / 'test_output'
        else:
            self.output_dir = self.case_dir / 'test_output'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_output_dir = self.output_dir / 'agent_output'
        self.oracle_output_dir = self.output_dir / 'oracle_output'
        self.agent_output_dir.mkdir(parents=True, exist_ok=True)
        self.oracle_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载case配置"""
        config_file = self.case_dir / 'config.json'
        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")
        
        with open(config_file) as f:
            return json.load(f)
    
    @abstractmethod
    def get_pde_type(self) -> str:
        """返回PDE类型"""
        pass
    
    def run_test(
        self, 
        agent_script: Path,
        test_mode: Literal['fix_accuracy', 'fix_time'],
        timeout_sec: int = 300
    ) -> Dict[str, Any]:
        """
        运行测试
        
        Args:
            agent_script: Agent生成的求解器脚本
            test_mode: 测试模式
                - 'fix_accuracy': 固定精度目标，优化运行时间
                - 'fix_time': 固定时间预算，优化精度
            timeout_sec: 超时时间
        
        Returns:
            测试结果字典
        """
        print(f"\n{'='*80}")
        print(f"🧪 Case: {self.case_id}")
        print(f"📋 Test Mode: {test_mode}")
        print(f"{'='*80}\n")
        
        # 获取测试参数
        test_params = self._get_test_params(test_mode)
        
        # 生成Oracle参考解
        print("🔮 Generating oracle reference solution...")
        oracle_config = self.config['oracle_config']
        generate(oracle_config, self.oracle_output_dir)
        print(f"   ✅ Oracle saved to {self.oracle_output_dir}")
        
        # 执行Agent脚本
        print(f"\n🤖 Executing agent script: {agent_script.name}")
        print(f"   Parameters: {test_params}")
        
        agent_result = execute_agent_script(
            script_path=agent_script,
            outdir=self.agent_output_dir,
            timeout_sec=timeout_sec,
            **test_params
        )
        
        if not agent_result.success:
            result = {
                'case_id': self.case_id,
                'test_mode': test_mode,
                'status': 'FAILED',
                'error': agent_result.error_message,
                'score': 0.0,
                'tier_levels': {'passed': [], 'total': 3, 'level_details': {f'level_{i}': False for i in [1, 2, 3]}}
            }
            # 保存结果
            result_file = self.output_dir / f'result_{test_mode}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            self._print_result(result)
            return result
        
        print(f"   ✅ Agent execution completed in {agent_result.t_agent_run:.3f}s")
        
        # 计算误差
        error = self._compute_error()
        
        if np.isnan(error):
            result = {
                'case_id': self.case_id,
                'test_mode': test_mode,
                'status': 'FAILED',
                'error': 'Error computation returned NaN',
                'score': 0.0,
                'tier_levels': {'passed': [], 'total': 3, 'level_details': {f'level_{i}': False for i in [1, 2, 3]}}
            }
            # 保存结果
            result_file = self.output_dir / f'result_{test_mode}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            self._print_result(result)
            return result
        
        print(f"   📊 Relative L2 Error: {error:.6e}")
        
        # 根据测试模式计算得分
        score = self._compute_score(
            mode=test_mode,
            runtime=agent_result.t_agent_run,
            error=error,
            target_error=self.config['evaluation_config']['target_error'],
            time_budget=self.config['evaluation_config'].get('time_budget', 60.0)
        )
        
        # 计算通过了哪些 tier levels
        passed_levels = self._check_tier_levels(
            mode=test_mode,
            runtime=agent_result.t_agent_run,
            error=error
        )
        
        result = {
            'case_id': self.case_id,
            'test_mode': test_mode,
            'status': 'PASSED' if score > 0 else 'FAILED',
            'runtime_sec': agent_result.t_agent_run,
            'error': float(error),
            'target_error': self.config['evaluation_config']['target_error'],
            'score': score,
            'test_params': test_params,
            'tier_levels': passed_levels  # 新增：通过的等级
        }
        
        # 计算专用指标（如果子类实现了）
        try:
            specialized_metrics = self._compute_specialized_metrics(result)
            if specialized_metrics:
                result['specialized_metrics'] = specialized_metrics
        except NotImplementedError:
            pass  # 基类不强制要求实现
        
        # 保存结果
        result_file = self.output_dir / f'result_{test_mode}.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        self._print_result(result)
        
        return result
    
    @abstractmethod
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        获取测试参数（子类实现）
        
        根据测试模式返回不同的参数：
        - fix_accuracy模式：使用推荐的高精度配置
        - fix_time模式：使用推荐的快速配置
        """
        pass
    
    def _compute_error(self) -> float:
        """计算误差（使用标准化的验证器）"""
        from pdebench.evaluation.validator import validate_solution
        
        validation_result = validate_solution(
            agent_outdir=self.agent_output_dir,
            oracle_outdir=self.oracle_output_dir,
            evaluation_config=self.config['evaluation_config'],
            oracle_config=self.config['oracle_config']
        )
        
        return validation_result.rel_L2_error
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算 PDE 类型专用指标（子类可选实现）
        
        Returns:
            专用指标字典，如：
            - 双曲：overshoot, conservation_error
            - Stokes: divergence_error
            - 非线性：newton_iterations
        """
        raise NotImplementedError("Subclass should implement if specialized metrics are needed")
    
    def _read_solver_info(self) -> Dict[str, Any]:
        """
        读取求解器信息（从 meta.json）
        
        Returns:
            求解器信息字典，可能包含：
            - linear_solver_type: 线性求解器类型 (CG, GMRES, etc.)
            - preconditioner_type: 预条件器类型 (AMG, ILU, etc.)
            - linear_iterations_mean: 平均线性迭代次数
            - linear_iterations_max: 最大线性迭代次数
            - nonlinear_iterations: 非线性迭代次数（如果适用）
        """
        solver_info = {}
        
        try:
            meta_file = self.agent_output_dir / 'meta.json'
            if not meta_file.exists():
                return solver_info
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            # 读取线性求解器信息
            if 'linear_solver' in meta:
                ls = meta['linear_solver']
                if isinstance(ls, dict):
                    solver_info['linear_solver_type'] = ls.get('type', 'unknown')
                    solver_info['preconditioner_type'] = ls.get('preconditioner', 'none')
                    
                    # 迭代次数
                    if 'iterations' in ls:
                        iters = ls['iterations']
                        if isinstance(iters, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iters))
                            solver_info['linear_iterations_max'] = int(np.max(iters))
                            solver_info['linear_iterations_total'] = int(np.sum(iters))
                        else:
                            solver_info['linear_iterations'] = iters
            
            # 读取非线性求解器信息
            if 'nonlinear_solver' in meta:
                ns = meta['nonlinear_solver']
                if isinstance(ns, dict):
                    solver_info['nonlinear_solver_type'] = ns.get('type', 'unknown')
                    
                    if 'iterations' in ns:
                        iters = ns['iterations']
                        if isinstance(iters, list):
                            solver_info['nonlinear_iterations_mean'] = float(np.mean(iters))
                            solver_info['nonlinear_iterations_max'] = int(np.max(iters))
                        else:
                            solver_info['nonlinear_iterations'] = iters
            
            # 读取离散化方法
            if 'discretization_method' in meta:
                solver_info['discretization_method'] = meta['discretization_method']
            
            # 读取时间积分方法（如果是瞬态问题）
            if 'time_integrator' in meta:
                solver_info['time_integrator'] = meta['time_integrator']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info
    
    def _compute_score(
        self,
        mode: str,
        runtime: float,
        error: float,
        target_error: float,
        time_budget: float
    ) -> float:
        """
        计算得分（0-100）
        
        fix_accuracy模式：固定精度，比速度
        - 必须达到target_error才有分
        - 得分 = 100 * (time_budget / runtime)，越快得分越高
        
        fix_time模式：固定时间，比精度
        - 必须在time_budget内完成才有分
        - 得分 = 100 * (1 - min(error/target_error, 1.0))，误差越小得分越高
        """
        if mode == 'fix_accuracy':
            # 固定精度模式：必须达到精度要求
            if error > target_error:
                print(f"   ❌ Failed: error {error:.2e} > target {target_error:.2e}")
                return 0.0
            
            # 计算速度得分：越快越好
            if runtime > time_budget:
                score = 50.0 * (time_budget / runtime)  # 超时但正确，给部分分
            else:
                score = 100.0 * (time_budget / max(runtime, 0.1))  # 快于预算，高分
                score = min(score, 100.0)
            
            print(f"   ✅ Passed: Accuracy target met in {runtime:.3f}s (budget: {time_budget:.1f}s)")
            print(f"   🎯 Score: {score:.1f}/100")
            return score
        
        elif mode == 'fix_time':
            # 固定时间模式：必须在时间预算内
            if runtime > time_budget:
                print(f"   ❌ Failed: runtime {runtime:.3f}s > budget {time_budget:.1f}s")
                return 0.0
            
            # 计算精度得分：误差越小越好
            error_ratio = error / target_error
            if error_ratio >= 1.0:
                score = 10.0  # 超过目标误差，给基础分
            else:
                # 误差越小，得分越高（非线性）
                score = 100.0 * (1.0 - error_ratio)
            
            print(f"   ✅ Passed: Achieved error {error:.2e} within {runtime:.3f}s")
            print(f"   🎯 Score: {score:.1f}/100")
            return score
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _check_tier_levels(self, mode: str, runtime: float, error: float) -> Dict[str, Any]:
        """检查通过了哪些 tier levels"""
        if 'tiers' not in self.config:
            return {'passed': [], 'total': 0}
        
        tiers = self.config['tiers']
        passed = []
        
        if mode == 'fix_accuracy':
            # 速度榜：检查在不同时间限制下能否达到中等精度
            target_error = tiers['accuracy']['level_2']['target_error']
            if error <= target_error:
                # 达到精度要求，检查速度等级
                if runtime <= tiers['speed']['fast']['time_budget']:
                    passed = [1, 2, 3]  # 通过所有等级
                elif runtime <= tiers['speed']['medium']['time_budget']:
                    passed = [1, 2]  # 通过 L1, L2
                elif runtime <= tiers['speed']['slow']['time_budget']:
                    passed = [1]  # 只通过 L1
        
        elif mode == 'fix_time':
            # 精度榜：检查在时间限制内能达到哪个精度等级
            time_budget = tiers['speed']['medium']['time_budget']
            if runtime <= time_budget:
                if error <= tiers['accuracy']['level_3']['target_error']:
                    passed = [1, 2, 3]  # 高精度
                elif error <= tiers['accuracy']['level_2']['target_error']:
                    passed = [1, 2]  # 中精度
                elif error <= tiers['accuracy']['level_1']['target_error']:
                    passed = [1]  # 低精度
        
        return {
            'passed': passed,
            'total': 3,
            'level_details': {
                f'level_{i}': (i in passed) for i in [1, 2, 3]
            }
        }
    
    def _print_result(self, result: Dict[str, Any]):
        """打印测试结果"""
        print(f"\n{'─'*80}")
        print(f"📊 Test Result: {result['case_id']}")
        print(f"{'─'*80}")
        print(f"Status: {result['status']}")
        
        # 只在字段存在时打印
        if 'runtime_sec' in result:
            print(f"Runtime: {result['runtime_sec']:.3f}s")
        if 'error' in result and isinstance(result['error'], (int, float)):
            print(f"Error: {result['error']:.6e}")
        elif 'error' in result:
            print(f"Error: {result['error']}")
        if 'target_error' in result:
            print(f"Target Error: {result['target_error']:.6e}")
        print(f"Score: {result.get('score', 0.0):.1f}/100")
        
        # 打印 tier levels
        if 'tier_levels' in result:
            levels = result['tier_levels']
            passed = levels['passed']
            total = levels['total']
            print(f"Tier Levels: {len(passed)}/{total} passed {passed}")
        
        # 打印专用指标
        if 'specialized_metrics' in result:
            print(f"Specialized Metrics: {result['specialized_metrics']}")
        
        print(f"{'─'*80}\n")


# ============================================================================
# 基础 PDE 类型测试类
# ============================================================================

class EllipticCaseTest(BaseCaseTest):
    """
    椭圆型方程测试基类 (Standard Elliptic Test)
    
    主要用于泊松方程 (Poisson) 等标准椭圆型问题。
    """
    
    def get_pde_type(self) -> str:
        return 'elliptic'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        泊松方程参数推荐：
        - fix_accuracy: 高精度配置 (N=128, P=2)
        - fix_time: 快速配置 (N=32, P=1)
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 128,
                'degree': 2
            }
        elif mode == 'fix_time':
            return {
                'resolution': 32,
                'degree': 1
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算椭圆型方程专用指标：
        - efficiency_dof_per_sec: 求解效率 DOF/s
        - dof: 自由度数量
        - solver_iterations: 线性求解器迭代次数
        - convergence_rate_estimate: 收敛阶估计 p_est = log(E1/E2)/log(h1/h2)
        - condition_number_estimate: 条件数估计（从迭代次数推断）
        """
        metrics = {}
        
        try:
            # 1. 计算 DOF（更准确的估计）
            resolution = result.get('test_params', {}).get('resolution', 0)
            degree = result.get('test_params', {}).get('degree', 1)
            
            # 2D 三角形网格：DOF ≈ (N+1)^2 for P1, 更复杂对于 P2+
            # 简化估计：DOF ≈ N^2 * (degree+1)*(degree+2)/2 (2D 三角形上的 P^k 元)
            if degree == 1:
                dof = resolution ** 2
            elif degree == 2:
                dof = (2 * resolution + 1) ** 2  # P2 大约有 (2N+1)^2 个节点
            else:
                dof = resolution ** 2 * degree ** 2  # 粗略估计
            
            metrics['dof'] = int(dof)
            metrics['resolution'] = int(resolution)
            metrics['degree'] = int(degree)
            
            # 2. 计算效率 DOF/s
            runtime = result.get('runtime_sec', 0)
            if runtime > 0:
                efficiency = dof / runtime
                metrics['efficiency_dof_per_sec'] = float(efficiency)
            
            # 3. 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
                
                # 条件数估计（从 CG 迭代次数推断）
                if 'linear_iterations_mean' in solver_info:
                    iters = solver_info['linear_iterations_mean']
                    # 对于 SPD 系统，CG 迭代次数 ~ sqrt(κ)
                    if iters > 0:
                        kappa_estimate = iters ** 2
                        metrics['condition_number_estimate'] = float(kappa_estimate)
            
            # 4. 收敛阶估计（需要多分辨率数据）
            # 如果有历史误差数据，可以估计收敛阶
            # 这里暂时不实现，需要存储多次运行的结果
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics


class ParabolicCaseTest(BaseCaseTest):
    """
    抛物型方程测试基类 (Standard Parabolic Test)
    
    主要用于热方程 (Heat Equation) 等标准抛物型问题。
    """
    
    def get_pde_type(self) -> str:
        return 'parabolic'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        热方程参数推荐：
        - fix_accuracy: 高精度配置 (N=128, P=2, dt=0.001)
        - fix_time: 快速配置 (N=32, P=1, dt=0.01)
        """
        oracle_time_config = self.config['oracle_config']['pde']['time']
        dt_oracle = oracle_time_config['dt']
        
        if mode == 'fix_accuracy':
            return {
                'resolution': 128,
                'degree': 2,
                'dt': dt_oracle * 0.5  # 更小的时间步长
            }
        elif mode == 'fix_time':
            return {
                'resolution': 32,
                'degree': 1,
                'dt': dt_oracle * 2.0  # 更大的时间步长（但保持稳定）
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算抛物型方程专用指标：
        - efficiency_workrate: 工作速率 (DOF × N_steps) / T_total
        - energy_decay_rate: 能量衰减率（L2范数应单调下降）
        - max_principle_check: 最大值原理检查
        - time_stepping_efficiency: 时间步进效率
        - cfl_number: CFL 数（隐式稳定性指标）
        """
        metrics = {}
        
        try:
            # 1. 计算 DOF 和时间步数
            resolution = result.get('test_params', {}).get('resolution', 0)
            degree = result.get('test_params', {}).get('degree', 1)
            
            # 修正 DOF 计算
            if degree == 1:
                dof = resolution ** 2
            elif degree == 2:
                dof = (2 * resolution + 1) ** 2
            else:
                dof = resolution ** 2 * degree ** 2
            
            oracle_time_config = self.config['oracle_config']['pde']['time']
            t_end = oracle_time_config['t_end']
            dt = result.get('test_params', {}).get('dt', oracle_time_config['dt'])
            n_steps = int(np.ceil(t_end / dt))
            
            metrics['dof'] = int(dof)
            metrics['n_steps'] = n_steps
            metrics['dt'] = float(dt)
            metrics['t_end'] = float(t_end)
            
            # 2. 计算 WorkRate
            runtime = result.get('runtime_sec', 0)
            if runtime > 0:
                workrate = (dof * n_steps) / runtime
                metrics['efficiency_workrate'] = float(workrate)
                
                # 每步平均时间
                time_per_step = runtime / n_steps
                metrics['time_per_step'] = float(time_per_step)
            
            # 3. CFL 数（网格尺寸，时间步长）
            h = 1.0 / resolution
            kappa = oracle_time_config.get('kappa', 1.0)
            # 热方程 CFL: dt / (h^2 / κ)
            cfl = kappa * dt / (h ** 2)
            metrics['cfl_number'] = float(cfl)
            if cfl > 0.5:  # 显式格式的稳定性极限
                metrics['cfl_warning'] = f"CFL={cfl:.2f} > 0.5 (显式格式不稳定)"
            
            # 4. 读取解的时间历史
            u_history_file = self.agent_output_dir / 'u_history.npy'
            if u_history_file.exists():
                u_history = np.load(u_history_file)
                
                # 能量（L2范数）历史
                energy_history = np.array([np.linalg.norm(u_history[i].flatten()) for i in range(len(u_history))])
                
                # 检查能量单调性
                energy_diffs = np.diff(energy_history)
                n_violations = np.sum(energy_diffs > 1e-10)
                
                metrics['energy_monotone'] = bool(n_violations == 0)
                metrics['energy_violations'] = int(n_violations)
                
                # 能量衰减率（指数拟合 E(t) ~ E0 * exp(-λt)）
                if energy_history[0] > 1e-14:
                    decay_ratio = (energy_history[0] - energy_history[-1]) / energy_history[0]
                    metrics['energy_decay_ratio'] = float(decay_ratio)
                    
                    # 估计衰减率 λ
                    if energy_history[-1] > 1e-14:
                        lambda_estimate = -np.log(energy_history[-1] / energy_history[0]) / t_end
                        metrics['decay_rate_lambda'] = float(lambda_estimate)
                
                # 5. 最大值原理检查
                # 齐次 Dirichlet 边界：max|u(t)| ≤ max|u(0)|
                initial_max = np.max(np.abs(u_history[0]))
                all_max = np.array([np.max(np.abs(u_history[i])) for i in range(len(u_history))])
                final_max = all_max[-1]
                global_max = np.max(all_max)
                
                metrics['initial_max'] = float(initial_max)
                metrics['final_max'] = float(final_max)
                metrics['global_max'] = float(global_max)
                
                # 检查违反（允许小的数值误差）
                tolerance = initial_max * 0.01  # 1% 容差
                if global_max > initial_max + tolerance:
                    metrics['max_principle_violated'] = True
                    metrics['max_principle_overshoot'] = float(global_max - initial_max)
                else:
                    metrics['max_principle_violated'] = False
            
            # 6. 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics


# ConvectionDiffusionCaseTest 已移除 - 应使用 MixedTypeCaseTest 或 EllipticCaseTest + 对应标签


# ============================================================================
# 扩展 PDE 类型测试类 (Phase 2)
# ============================================================================

class IncompressibleFlowCaseTest(BaseCaseTest):
    """鞍点问题（Stokes 方程）测试"""
    
    def get_pde_type(self) -> str:
        return 'incompressible_flow'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        Stokes 方程参数推荐：
        - Taylor-Hood 元 (P2-P1) 满足 inf-sup 条件
        - 需要块预条件器 (field-split)
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 64,
                'degree': 2  # 速度空间阶数；压力为 degree-1
            }
        elif mode == 'fix_time':
            return {
                'resolution': 32,
                'degree': 1  # 速度 P1，压力 P0（MINI element 或 stabilized）
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算不可压流（Stokes/NS）专用指标：
        - divergence_L2: ||∇·u||_L2（散度自由度）
        - mass_flux_error: ∫_Ω ∇·u dx（全局质量守恒）
        - pressure_mean_check: 压力均值（零空间处理）
        - inf_sup_stability: inf-sup 稳定性指标
        - velocity_gradient_L2: ||∇u||_L2（剪切率）
        """
        metrics = {}
        
        try:
            # 1. 读取速度场
            agent_u_file = self.agent_output_dir / 'u.npy'
            if agent_u_file.exists():
                u = np.load(agent_u_file)
                
                # 速度场 L2 范数
                metrics['velocity_L2'] = float(np.linalg.norm(u))
                
                # 计算散度（有限差分）
                if u.ndim >= 3:  # (nx, ny, dim) 或 (nx, ny, nz, dim)
                    div_u = self._compute_divergence_fd(u)
                    
                    # 散度的 L2 范数（相对于速度范数）
                    div_L2 = np.linalg.norm(div_u)
                    metrics['divergence_L2'] = float(div_L2)
                    
                    u_L2 = np.linalg.norm(u)
                    if u_L2 > 1e-14:
                        metrics['divergence_relative'] = float(div_L2 / u_L2)
                    
                    # 全局质量通量误差（积分）
                    # ∫_Ω ∇·u dx 应该为 0（或等于边界通量）
                    mass_flux = np.sum(div_u)
                    metrics['mass_flux_integral'] = float(mass_flux)
                    
                    # 相对质量守恒误差
                    # 对于封闭域：|∫ ∇·u| / (∫ |u|)
                    total_velocity_mag = np.sum(np.abs(u))
                    if total_velocity_mag > 1e-14:
                        mass_error = np.abs(mass_flux) / total_velocity_mag
                        metrics['mass_conservation_error'] = float(mass_error)
                    
                    # 速度梯度范数（剪切率）
                    grad_u_norm = self._compute_velocity_gradient_norm(u)
                    metrics['velocity_gradient_L2'] = float(grad_u_norm)
            
            # 2. 读取压力场
            agent_p_file = self.agent_output_dir / 'p.npy'
            if agent_p_file.exists():
                p = np.load(agent_p_file)
                
                p_L2 = np.linalg.norm(p)
                metrics['pressure_L2'] = float(p_L2)
                
                # 压力均值（零空间检查）
                p_mean = np.mean(p)
                p_std = np.std(p)
                
                metrics['pressure_mean'] = float(p_mean)
                metrics['pressure_std'] = float(p_std)
                
                # 检查是否处理了压力常数不定性
                # 通常应固定一个点的压力或强制零均值
                if np.abs(p_mean) > 0.01 * p_std:  # 均值显著非零
                    metrics['pressure_mean_enforced'] = False
                else:
                    metrics['pressure_mean_enforced'] = True
            
            # 3. inf-sup 稳定性指标（简化检查）
            # 真正的 inf-sup 常数需要特征值计算，这里用启发式指标
            if agent_u_file.exists() and agent_p_file.exists():
                u = np.load(agent_u_file)
                p = np.load(agent_p_file)
                div_u = self._compute_divergence_fd(u)
                
                # 压力梯度范数
                grad_p = self._compute_pressure_gradient_norm(p)
                
                # 启发式 inf-sup 指标：||∇p|| / ||u||
                u_norm = np.linalg.norm(u)
                if u_norm > 1e-14:
                    inf_sup_indicator = grad_p / u_norm
                    metrics['inf_sup_indicator'] = float(inf_sup_indicator)
            
            # 4. 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics
    
    def _compute_divergence_fd(self, u: np.ndarray) -> np.ndarray:
        """使用有限差分计算散度 ∇·u"""
        if u.ndim == 3:  # 2D: (nx, ny, 2)
            nx, ny, _ = u.shape
            h = 1.0 / nx
            
            # ∂u_x/∂x + ∂u_y/∂y
            du_x = np.gradient(u[:, :, 0], h, axis=0)
            du_y = np.gradient(u[:, :, 1], h, axis=1)
            div_u = du_x + du_y
            
            return div_u
        else:
            return np.zeros_like(u[:, :, 0])
    
    def _compute_velocity_gradient_norm(self, u: np.ndarray) -> float:
        """计算速度梯度范数 ||∇u||_L2"""
        try:
            if u.ndim == 3:  # 2D: (nx, ny, 2)
                nx, ny, _ = u.shape
                h = 1.0 / nx
                
                # ∂u_x/∂x, ∂u_x/∂y, ∂u_y/∂x, ∂u_y/∂y
                du_x_dx = np.gradient(u[:, :, 0], h, axis=0)
                du_x_dy = np.gradient(u[:, :, 0], h, axis=1)
                du_y_dx = np.gradient(u[:, :, 1], h, axis=0)
                du_y_dy = np.gradient(u[:, :, 1], h, axis=1)
                
                # Frobenius 范数：sqrt(sum of all gradients squared)
                grad_norm_sq = du_x_dx**2 + du_x_dy**2 + du_y_dx**2 + du_y_dy**2
                return np.sqrt(np.sum(grad_norm_sq))
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_pressure_gradient_norm(self, p: np.ndarray) -> float:
        """计算压力梯度范数 ||∇p||_L2"""
        try:
            if p.ndim == 2:  # 2D: (nx, ny)
                nx, ny = p.shape
                h = 1.0 / nx
                
                # ∂p/∂x, ∂p/∂y
                dp_dx = np.gradient(p, h, axis=0)
                dp_dy = np.gradient(p, h, axis=1)
                
                grad_norm_sq = dp_dx**2 + dp_dy**2
                return np.sqrt(np.sum(grad_norm_sq))
            else:
                return 0.0
        except:
            return 0.0


# HelmholtzCaseTest 已移除 - 应使用 EllipticCaseTest + structure: symmetric_indefinite 标签


class HyperbolicCaseTest(BaseCaseTest):
    """双曲型一阶（对流/守恒律）测试"""
    
    def get_pde_type(self) -> str:
        return 'hyperbolic'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        双曲方程参数推荐：
        - 上风格式 / DG / flux limiter
        - 时间步需满足 CFL 条件
        """
        # 从 config 读取对流速度和 CFL
        pde_config = self.config.get('oracle_config', {}).get('pde', {})
        cfl_target = pde_config.get('cfl', 0.5)  # 默认 CFL = 0.5
        
        if mode == 'fix_accuracy':
            resolution = 128
            h = 1.0 / resolution
            dt = cfl_target * h  # CFL 条件
            return {
                'resolution': resolution,
                'degree': 1,
                'dt': dt
            }
        elif mode == 'fix_time':
            resolution = 64
            h = 1.0 / resolution
            dt = cfl_target * h
            return {
                'resolution': resolution,
                'degree': 1,
                'dt': dt
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算双曲方程专用指标：
        - overshoot/undershoot: 非物理震荡（违反最大值原理）
        - tv_growth: TV(u_final) / TV(u_initial) - TVD 格式应 ≤ 1
        - mass_conservation_error: 守恒律检查
        - shock_resolution: 激波分辨率
        - cfl_number: CFL 数
        """
        metrics = {}
        
        try:
            # 读取解
            agent_u_file = self.agent_output_dir / 'u.npy'
            oracle_u_file = self.oracle_output_dir / 'u.npy'
            
            if agent_u_file.exists() and oracle_u_file.exists():
                u_agent = np.load(agent_u_file)
                u_oracle = np.load(oracle_u_file)
                
                # 1. 超调/欠调（相对于 oracle 的范围）
                u_max_ref = np.max(u_oracle)
                u_min_ref = np.min(u_oracle)
                
                overshoot = max(0.0, np.max(u_agent) - u_max_ref)
                undershoot = max(0.0, u_min_ref - np.min(u_agent))
                
                metrics['overshoot'] = float(overshoot)
                metrics['undershoot'] = float(undershoot)
                
                # 相对震荡强度
                solution_range = u_max_ref - u_min_ref
                if solution_range > 1e-14:
                    metrics['overshoot_relative'] = float(overshoot / solution_range)
                    metrics['undershoot_relative'] = float(undershoot / solution_range)
                
                # 2. 总变差（TV）
                tv_agent = self._compute_total_variation(u_agent)
                metrics['total_variation'] = float(tv_agent)
                
                # TV 增长率（TVD 检查）
                agent_u0_file = self.agent_output_dir / 'u_initial.npy'
                if agent_u0_file.exists():
                    u0 = np.load(agent_u0_file)
                    tv_initial = self._compute_total_variation(u0)
                    
                    if tv_initial > 1e-14:
                        tv_growth = tv_agent / tv_initial
                        metrics['tv_growth_ratio'] = float(tv_growth)
                        
                        if tv_growth > 1.01:  # 允许 1% 数值误差
                            metrics['tvd_violated'] = True
                        else:
                            metrics['tvd_violated'] = False
                    
                    # 3. 质量守恒
                    mass_initial = np.sum(u0)
                    mass_final = np.sum(u_agent)
                    
                    if np.abs(mass_initial) > 1e-14:
                        mass_error = np.abs(mass_final - mass_initial) / np.abs(mass_initial)
                        metrics['mass_conservation_error'] = float(mass_error)
                
                # 4. 激波分辨率
                shock_width = self._compute_shock_width(u_agent)
                metrics['shock_width_points'] = float(shock_width)
                
                # 相对激波宽度（相对于网格尺寸）
                resolution = result.get('test_params', {}).get('resolution', 0)
                if resolution > 0 and shock_width > 0:
                    h = 1.0 / resolution
                    shock_width_physical = shock_width * h
                    metrics['shock_width_physical'] = float(shock_width_physical)
            
            # 5. CFL 数
            resolution = result.get('test_params', {}).get('resolution', 0)
            dt = result.get('test_params', {}).get('dt', 0)
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            advection_speed = pde_config.get('advection_speed', 1.0)
            
            if resolution > 0 and dt > 0:
                h = 1.0 / resolution
                cfl = advection_speed * dt / h
                metrics['cfl_number'] = float(cfl)
                
                if cfl > 1.0:
                    metrics['cfl_warning'] = f"CFL={cfl:.2f} > 1.0 (可能不稳定)"
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics
    
    def _compute_total_variation(self, u: np.ndarray) -> float:
        """计算总变差 TV(u)"""
        if u.ndim == 1:
            return np.sum(np.abs(np.diff(u)))
        elif u.ndim == 2:
            tv_x = np.sum(np.abs(np.diff(u, axis=0)))
            tv_y = np.sum(np.abs(np.diff(u, axis=1)))
            return tv_x + tv_y
        else:
            return 0.0

    def _compute_shock_width(self, u: np.ndarray) -> float:
        """
        估算激波宽度（占用的网格点数）
        算法：找到梯度最大的位置，计算从 10% 到 90% 跳变所需的距离
        """
        try:
            if u.ndim != 1:
                return 0.0  # 目前只支持 1D 激波宽度计算
            
            # 计算梯度绝对值
            grad = np.abs(np.gradient(u))
            max_grad_idx = np.argmax(grad)
            max_grad = grad[max_grad_idx]
            
            if max_grad < 1e-6:
                return 0.0  # 没有明显的梯度/激波
            
            # 以最大梯度点为中心，向左右搜索
            # 定义激波的上下界：取解的 min 和 max
            u_min, u_max = np.min(u), np.max(u)
            jump = u_max - u_min
            if jump < 1e-6:
                return 0.0
                
            # 简单的半高宽估计 FWHM 或 10-90% Rise Distance
            # 这里使用梯度分布的宽度：有多少个点的梯度 > max_grad * 0.1
            width_points = np.sum(grad > max_grad * 0.1)
            return float(width_points)
            
        except:
            return 0.0




# ============================================================================
# 新增的8个标准 PDE 类型测试类
# ============================================================================

class MixedTypeCaseTest(BaseCaseTest):
    """
    混合型方程测试（如对流扩散方程，随参数可在椭圆/双曲间切换）
    
    主要针对对流扩散方程：-ε∆u + b·∇u = f
    - 当 ε >> ||b|| 时表现为椭圆型（扩散主导）
    - 当 ε << ||b|| 时表现为双曲型（对流主导）
    """
    
    def get_pde_type(self) -> str:
        return 'mixed_type'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        对流扩散参数推荐：
        - 需要稳定化（SUPG/Upwind）
        - 高 Péclet 数需要更高分辨率或稳定化
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 128,
                'degree': 2
            }
        elif mode == 'fix_time':
            return {
                'resolution': 64,
                'degree': 1
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算对流扩散专用指标：
        - peclet_number: Péclet 数 Pe = ||b||L/ε
        - overshoot/undershoot: 非物理震荡指标
        - boundary_layer_resolution: 边界层分辨率
        - stabilization_quality: 稳定化效果评估
        """
        metrics = {}
        
        try:
            # 读取 Péclet 数
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            peclet = pde_config.get('peclet', None)
            if peclet is not None:
                metrics['peclet_number'] = float(peclet)
            
            # 读取解
            agent_u_file = self.agent_output_dir / 'u.npy'
            oracle_u_file = self.oracle_output_dir / 'u.npy'
            
            if agent_u_file.exists() and oracle_u_file.exists():
                u_agent = np.load(agent_u_file)
                u_oracle = np.load(oracle_u_file)
                
                # 1. 超调/欠调指标
                u_max_ref = np.max(u_oracle)
                u_min_ref = np.min(u_oracle)
                
                overshoot = np.max([0.0, np.max(u_agent) - u_max_ref])
                undershoot = np.max([0.0, u_min_ref - np.min(u_agent)])
                
                metrics['overshoot'] = float(overshoot)
                metrics['undershoot'] = float(undershoot)
                
                solution_range = u_max_ref - u_min_ref
                if solution_range > 1e-14:
                    metrics['overshoot_relative'] = float(overshoot / solution_range)
                    metrics['undershoot_relative'] = float(undershoot / solution_range)
                
                # 2. 总变差（检测 Gibbs 震荡）
                tv_agent = self._compute_total_variation(u_agent)
                tv_oracle = self._compute_total_variation(u_oracle)
                
                metrics['total_variation'] = float(tv_agent)
                if tv_oracle > 1e-14:
                    metrics['tv_ratio'] = float(tv_agent / tv_oracle)
                
                # 3. 边界层误差
                if peclet is not None and peclet > 1:
                    bl_error = self._compute_boundary_layer_error(u_agent, u_oracle, peclet)
                    if bl_error is not None:
                        metrics['boundary_layer_error'] = float(bl_error)
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics
    
    def _compute_total_variation(self, u: np.ndarray) -> float:
        """计算总变差 TV(u)"""
        if u.ndim == 1:
            return float(np.sum(np.abs(np.diff(u))))
        elif u.ndim == 2:
            tv_x = np.sum(np.abs(np.diff(u, axis=0)))
            tv_y = np.sum(np.abs(np.diff(u, axis=1)))
            return float(tv_x + tv_y)
        else:
            return 0.0
    
    def _compute_boundary_layer_error(self, u_agent: np.ndarray, u_oracle: np.ndarray, peclet: float) -> Optional[float]:
        """计算边界层区域的误差"""
        try:
            if u_agent.ndim == 1:
                nx = len(u_agent)
                epsilon = 1.0 / (peclet + 1e-10)
                bl_thickness = 3 * epsilon
                bl_points = int(bl_thickness * nx)
                bl_points = max(bl_points, 5)
                bl_points = min(bl_points, nx // 4)
                
                err_left = np.linalg.norm(u_agent[:bl_points] - u_oracle[:bl_points])
                err_right = np.linalg.norm(u_agent[-bl_points:] - u_oracle[-bl_points:])
                
                return max(err_left, err_right)
            else:
                return None
        except:
            return None


class DispersiveCaseTest(BaseCaseTest):
    """
    色散型方程测试（Schrödinger, KdV 等）
    
    核心特征：
    - 相速度 ≠ 群速度
    - 需要保持相位关系
    - 通常需要复数运算或特殊时间积分
    """
    
    def get_pde_type(self) -> str:
        return 'dispersive'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        色散方程参数推荐：
        - 高阶空间离散（至少 P2 或 spectral）
        - 小时间步长以捕捉高频模式
        - 对称时间积分格式（Crank-Nicolson, Strang splitting）
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 256,  # 需要高分辨率
                'degree': 2,
                'dt': 0.0001  # 小时间步长
            }
        elif mode == 'fix_time':
            return {
                'resolution': 128,
                'degree': 1,
                'dt': 0.001
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算色散方程专用指标：
        - phase_velocity_error: 相速度误差
        - group_velocity_error: 群速度误差
        - dispersion_error: 色散关系误差
        - mass_conservation: 质量守恒（Schrödinger: ∫|ψ|²dx）
        - energy_conservation: 能量守恒
        """
        metrics = {}
        
        try:
            # 读取解
            agent_u_file = self.agent_output_dir / 'u.npy'
            oracle_u_file = self.oracle_output_dir / 'u.npy'
            
            if agent_u_file.exists() and oracle_u_file.exists():
                u_agent = np.load(agent_u_file)
                u_oracle = np.load(oracle_u_file)
                
                # 1. 质量守恒（L2范数）
                mass_agent = np.linalg.norm(u_agent)
                mass_oracle = np.linalg.norm(u_oracle)
                
                metrics['mass_agent'] = float(mass_agent)
                metrics['mass_oracle'] = float(mass_oracle)
                
                if mass_oracle > 1e-14:
                    mass_error = np.abs(mass_agent - mass_oracle) / mass_oracle
                    metrics['mass_conservation_error'] = float(mass_error)
                
                # 2. 相位误差（波峰位置）
                phase_error = self._compute_phase_error(u_agent, u_oracle)
                metrics['phase_error'] = float(phase_error)
                
                # 3. 频谱比较（如果解是周期的）
                if u_agent.ndim == 1:
                    spectrum_error = self._compute_spectrum_error(u_agent, u_oracle)
                    if spectrum_error is not None:
                        metrics['spectrum_error'] = float(spectrum_error)
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics
    
    def _compute_phase_error(self, u_agent: np.ndarray, u_oracle: np.ndarray) -> float:
        """计算相位误差"""
        try:
            idx_agent = np.argmax(np.abs(u_agent))
            idx_oracle = np.argmax(np.abs(u_oracle))
            
            if u_agent.ndim == 1:
                return np.abs(idx_agent - idx_oracle) / u_agent.shape[0]
            elif u_agent.ndim == 2:
                row_a, col_a = np.unravel_index(idx_agent, u_agent.shape)
                row_o, col_o = np.unravel_index(idx_oracle, u_oracle.shape)
                return np.sqrt((row_a - row_o)**2 + (col_a - col_o)**2) / u_agent.shape[0]
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_spectrum_error(self, u_agent: np.ndarray, u_oracle: np.ndarray) -> Optional[float]:
        """计算频谱误差"""
        try:
            if u_agent.ndim != 1:
                return None
            
            # FFT
            fft_agent = np.fft.fft(u_agent)
            fft_oracle = np.fft.fft(u_oracle)
            
            # L2 误差在频域
            spectrum_error = np.linalg.norm(fft_agent - fft_oracle) / np.linalg.norm(fft_oracle)
            return spectrum_error
        except:
            return None


class ReactionDiffusionCaseTest(BaseCaseTest):
    """
    反应扩散方程测试（Allen-Cahn, Fisher-KPP, Gray-Scott 等）
    
    核心挑战：
    - 非线性反应项的刚性
    - 模式形成（pattern formation）
    - 行波解的传播速度
    - 能量定律
    """
    
    def get_pde_type(self) -> str:
        return 'reaction_diffusion'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        反应扩散参数推荐：
        - IMEX 时间积分（隐式扩散 + 显式反应）
        - 自适应时间步长
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 128,
                'degree': 2,
                'dt': 0.001
            }
        elif mode == 'fix_time':
            return {
                'resolution': 64,
                'degree': 1,
                'dt': 0.01
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算反应扩散专用指标：
        - energy_decay: 自由能单调递减（Allen-Cahn）
        - mass_conservation: 质量守恒（某些模型）
        - front_propagation_speed: 行波传播速度
        - pattern_quality: 模式形成质量
        - nonlinear_solver_efficiency: 非线性迭代效率
        """
        metrics = {}
        
        try:
            # 读取 meta.json
            meta_file = self.agent_output_dir / 'meta.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                
                # 1. 非线性迭代
                if 'nonlinear_solver' in meta:
                    ns = meta['nonlinear_solver']
                    if isinstance(ns, dict) and 'iterations' in ns:
                        iters = ns['iterations']
                        if isinstance(iters, list):
                            metrics['newton_iterations_mean'] = float(np.mean(iters))
                            metrics['newton_iterations_max'] = int(np.max(iters))
                
                # 2. 能量演化
                if 'energy_history' in meta:
                    energy = np.array(meta['energy_history'])
                    
                    energy_diffs = np.diff(energy)
                    n_violations = np.sum(energy_diffs > 1e-10)
                    
                    metrics['energy_monotone'] = bool(n_violations == 0)
                    metrics['energy_violations'] = int(n_violations)
                    
                    if len(energy) > 1 and np.abs(energy[0]) > 1e-14:
                        decay_ratio = (energy[0] - energy[-1]) / np.abs(energy[0])
                        metrics['energy_decay_ratio'] = float(decay_ratio)
            
            # 3. 质量守恒
            u0_file = self.agent_output_dir / 'u_initial.npy'
            u_final_file = self.agent_output_dir / 'u.npy'
            
            if u0_file.exists() and u_final_file.exists():
                u0 = np.load(u0_file)
                u_final = np.load(u_final_file)
                
                mass0 = np.sum(u0)
                mass_final = np.sum(u_final)
                
                if np.abs(mass0) > 1e-14:
                    mass_error = np.abs(mass_final - mass0) / np.abs(mass0)
                    metrics['mass_conservation_error'] = float(mass_error)
            
            # 4. 行波传播速度（如果有初始和最终状态）
            if u0_file.exists() and u_final_file.exists():
                u0 = np.load(u0_file)
                u_final = np.load(u_final_file)
                
                front_speed = self._estimate_front_speed(u0, u_final, result)
                if front_speed is not None:
                    metrics['front_propagation_speed'] = float(front_speed)
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics
    
    def _estimate_front_speed(self, u0: np.ndarray, u_final: np.ndarray, result: Dict) -> Optional[float]:
        """估算行波传播速度"""
        try:
            if u0.ndim != 1:
                return None
            
            # 找到半高点的位置
            threshold = 0.5 * (np.max(u0) + np.min(u0))
            
            front_idx_0 = np.argmax(u0 > threshold)
            front_idx_final = np.argmax(u_final > threshold)
            
            dx = 1.0 / len(u0)
            distance = (front_idx_final - front_idx_0) * dx
            
            # 总时间
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            if 'time' in pde_config:
                t_end = pde_config['time'].get('t_end', 1.0)
                speed = distance / t_end
                return speed
            else:
                return None
        except:
            return None


class CompressibleFlowCaseTest(BaseCaseTest):
    """
    可压缩流测试（Euler 方程，可压缩 Navier-Stokes）
    
    核心挑战：
    - 激波捕捉
    - 熵条件
    - 密度正性保持
    - 高 Mach 数稳定性
    """
    
    def get_pde_type(self) -> str:
        return 'compressible_flow'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        可压缩流参数推荐：
        - 高分辨率激波捕捉格式（WENO, TVD）
        - CFL 条件
        - Riemann solver
        """
        pde_config = self.config.get('oracle_config', {}).get('pde', {})
        cfl = pde_config.get('cfl', 0.5)
        
        if mode == 'fix_accuracy':
            resolution = 256
            h = 1.0 / resolution
            dt = cfl * h
            return {
                'resolution': resolution,
                'degree': 2,
                'dt': dt
            }
        elif mode == 'fix_time':
            resolution = 128
            h = 1.0 / resolution
            dt = cfl * h
            return {
                'resolution': resolution,
                'degree': 1,
                'dt': dt
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算可压缩流专用指标：
        - shock_resolution: 激波分辨率
        - density_positivity: 密度正性保持
        - entropy_production: 熵产生（应非负）
        - mass/momentum/energy_conservation: 守恒律
        - mach_number: Mach 数
        """
        metrics = {}
        
        try:
            # 读取密度场
            agent_rho_file = self.agent_output_dir / 'rho.npy'
            oracle_rho_file = self.oracle_output_dir / 'rho.npy'
            
            if agent_rho_file.exists():
                rho_agent = np.load(agent_rho_file)
                
                # 1. 密度正性检查
                rho_min = np.min(rho_agent)
                metrics['density_min'] = float(rho_min)
                metrics['density_positive'] = bool(rho_min > -1e-10)
                
                if rho_min < 0:
                    metrics['density_positivity_violation'] = float(np.abs(rho_min))
                
                # 2. 激波分辨率
                if rho_agent.ndim == 1:
                    shock_width = self._compute_shock_width(rho_agent)
                    metrics['shock_width'] = float(shock_width)
            
            # 3. 守恒律检查
            mass_error = self._check_mass_conservation()
            if mass_error is not None:
                metrics['mass_conservation_error'] = float(mass_error)
            
            # 4. Mach 数
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            mach = pde_config.get('mach', None)
            if mach is not None:
                metrics['mach_number'] = float(mach)
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics
    
    def _compute_shock_width(self, rho: np.ndarray) -> float:
        """估算激波宽度"""
        try:
            grad = np.abs(np.gradient(rho))
            max_grad = np.max(grad)
            
            if max_grad < 1e-6:
                return 0.0
            
            width_points = np.sum(grad > max_grad * 0.1)
            return float(width_points)
        except:
            return 0.0
    
    def _check_mass_conservation(self) -> Optional[float]:
        """检查质量守恒"""
        try:
            rho0_file = self.agent_output_dir / 'rho_initial.npy'
            rho_final_file = self.agent_output_dir / 'rho.npy'
            
            if rho0_file.exists() and rho_final_file.exists():
                rho0 = np.load(rho0_file)
                rho_final = np.load(rho_final_file)
                
                mass0 = np.sum(rho0)
                mass_final = np.sum(rho_final)
                
                if np.abs(mass0) > 1e-14:
                    return np.abs(mass_final - mass0) / np.abs(mass0)
            
            return None
        except:
            return None


class KineticCaseTest(BaseCaseTest):
    """
    动理学方程测试（Vlasov, Boltzmann, Fokker-Planck）
    
    核心挑战：
    - 高维相空间离散
    - 质量/动量/能量守恒
    - 熵增原理
    - 计算代价极高
    """
    
    def get_pde_type(self) -> str:
        return 'kinetic'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        动理学方程参数推荐：
        - 相空间网格（x, v）
        - 半拉格朗日或谱方法
        """
        if mode == 'fix_accuracy':
            return {
                'resolution_x': 64,  # 物理空间
                'resolution_v': 64,  # 速度空间
                'degree': 2,
                'dt': 0.01
            }
        elif mode == 'fix_time':
            return {
                'resolution_x': 32,
                'resolution_v': 32,
                'degree': 1,
                'dt': 0.05
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算动理学方程专用指标：
        - mass_conservation: ∫f dxdv = const
        - momentum_conservation: ∫v·f dxdv = const
        - energy_conservation: ∫v²·f dxdv = const
        - entropy_production: 熵增（H定理）
        - computational_cost: 计算成本（DOF 很大）
        """
        metrics = {}
        
        try:
            # 读取分布函数 f(x, v)
            agent_f_file = self.agent_output_dir / 'f.npy'
            
            if agent_f_file.exists():
                f_agent = np.load(agent_f_file)
                
                # 假设 f 的形状为 (nx, nv)
                if f_agent.ndim == 2:
                    nx, nv = f_agent.shape
                    
                    # 计算宏观量（简化：假设 v ∈ [-v_max, v_max]）
                    v_max = 5.0  # 从配置读取
                    dv = 2 * v_max / nv
                    v_grid = np.linspace(-v_max, v_max, nv)
                    
                    # 密度：ρ(x) = ∫f dv
                    rho = np.sum(f_agent, axis=1) * dv
                    metrics['total_mass'] = float(np.sum(rho))
                    
                    # 动量：m(x) = ∫v·f dv
                    momentum = np.sum(f_agent * v_grid[None, :], axis=1) * dv
                    metrics['total_momentum'] = float(np.sum(momentum))
                    
                    # 能量：E = ∫v²·f dv
                    energy = np.sum(f_agent * (v_grid[None, :]**2), axis=1) * dv
                    metrics['total_energy'] = float(np.sum(energy))
            
            # 检查守恒性（需要初始状态）
            f0_file = self.agent_output_dir / 'f_initial.npy'
            if f0_file.exists() and agent_f_file.exists():
                f0 = np.load(f0_file)
                f_final = np.load(agent_f_file)
                
                if f0.ndim == 2:
                    nx, nv = f0.shape
                    v_max = 5.0
                    dv = 2 * v_max / nv
                    
                    mass0 = np.sum(f0) * dv
                    mass_final = np.sum(f_final) * dv
                    
                    if np.abs(mass0) > 1e-14:
                        mass_error = np.abs(mass_final - mass0) / np.abs(mass0)
                        metrics['mass_conservation_error'] = float(mass_error)
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics


class FractionalCaseTest(BaseCaseTest):
    """
    分数阶 PDE 测试（分数 Laplacian, Caputo 导数）
    
    核心挑战：
    - 非局部算子的计算
    - 稠密矩阵
    - 边界条件处理
    - 收敛阶与分数阶的关系
    """
    
    def get_pde_type(self) -> str:
        return 'fractional'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        分数阶方程参数推荐：
        - 需要特殊的求积公式
        - 计算成本高（非局部）
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 128,
                'degree': 2,
                'dt': 0.001  # 如果有时间演化
            }
        elif mode == 'fix_time':
            return {
                'resolution': 64,
                'degree': 1,
                'dt': 0.01
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算分数阶方程专用指标：
        - fractional_order_alpha: 分数阶参数 α
        - convergence_rate: 收敛阶（与 α 相关）
        - computational_cost_ratio: 相对标准方法的计算成本
        - matrix_sparsity: 矩阵稀疏性（通常很低）
        """
        metrics = {}
        
        try:
            # 读取分数阶参数
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            alpha = pde_config.get('fractional_alpha', None)
            if alpha is not None:
                metrics['fractional_order_alpha'] = float(alpha)
            
            # 计算 DOF（分数阶通常需要全局自由度）
            resolution = result.get('test_params', {}).get('resolution', 0)
            degree = result.get('test_params', {}).get('degree', 1)
            dof = (resolution * degree) ** 2
            metrics['dof'] = int(dof)
            
            # 计算效率（分数阶方法通常慢）
            runtime = result.get('runtime_sec', 0)
            if runtime > 0:
                efficiency = dof / runtime
                metrics['efficiency_dof_per_sec'] = float(efficiency)
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
                
                # 检查矩阵填充度
                if 'matrix_nnz' in solver_info and 'dof' in metrics:
                    nnz = solver_info['matrix_nnz']
                    dof = metrics['dof']
                    sparsity = 1.0 - nnz / (dof ** 2)
                    metrics['matrix_sparsity'] = float(sparsity)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics


class StochasticCaseTest(BaseCaseTest):
    """
    随机 PDE 测试（SPDE: 带随机强迫或系数的 PDE）
    
    核心挑战：
    - 样本统计（均值、方差、pdf）
    - Monte Carlo 方法
    - 随机 Galerkin
    - 不确定性量化
    """
    
    def get_pde_type(self) -> str:
        return 'stochastic'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        SPDE 参数推荐：
        - 需要多次实现（MC样本数）
        - 或使用 Polynomial Chaos
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 64,
                'degree': 2,
                'n_samples': 100  # Monte Carlo 样本数
            }
        elif mode == 'fix_time':
            return {
                'resolution': 32,
                'degree': 1,
                'n_samples': 50
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算随机PDE专用指标：
        - mean_solution: 解的均值
        - variance: 解的方差
        - confidence_interval: 置信区间
        - mc_convergence_rate: MC 收敛率（1/√N）
        - moments_accuracy: 高阶矩精度
        """
        metrics = {}
        
        try:
            # 读取多个样本（假设存储为 u_sample_0.npy, u_sample_1.npy, ...）
            samples = []
            i = 0
            while True:
                sample_file = self.agent_output_dir / f'u_sample_{i}.npy'
                if not sample_file.exists():
                    break
                u_sample = np.load(sample_file)
                samples.append(u_sample)
                i += 1
            
            if len(samples) > 0:
                samples = np.array(samples)  # shape: (n_samples, ...)
                
                # 1. 均值和方差
                u_mean = np.mean(samples, axis=0)
                u_var = np.var(samples, axis=0)
                
                metrics['n_samples'] = len(samples)
                metrics['mean_L2_norm'] = float(np.linalg.norm(u_mean))
                metrics['mean_variance'] = float(np.mean(u_var))
                metrics['max_variance'] = float(np.max(u_var))
                
                # 2. 置信区间宽度（95% CI）
                u_std = np.std(samples, axis=0)
                ci_width = 1.96 * u_std / np.sqrt(len(samples))
                metrics['mean_ci_width'] = float(np.mean(ci_width))
                
                # 3. 与 oracle 均值比较（如果存在）
                oracle_mean_file = self.oracle_output_dir / 'u_mean.npy'
                if oracle_mean_file.exists():
                    u_mean_oracle = np.load(oracle_mean_file)
                    mean_error = np.linalg.norm(u_mean - u_mean_oracle) / np.linalg.norm(u_mean_oracle)
                    metrics['mean_solution_error'] = float(mean_error)
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics


class MultiphysicsCaseTest(BaseCaseTest):
    """
    多物理耦合测试（流固耦合、热-力耦合、电磁-热耦合等）
    
    核心挑战：
    - 多场耦合
    - 块预条件
    - 时间尺度分离
    - 界面条件
    """
    
    def get_pde_type(self) -> str:
        return 'multiphysics'
    
    def _get_test_params(self, mode: str) -> Dict[str, Any]:
        """
        多物理参数推荐：
        - 分区/单域方法
        - 块预条件器
        - 可能需要不同网格
        """
        if mode == 'fix_accuracy':
            return {
                'resolution': 64,
                'degree': 2,
                'dt': 0.001
            }
        elif mode == 'fix_time':
            return {
                'resolution': 32,
                'degree': 1,
                'dt': 0.01
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_specialized_metrics(self, result: Dict) -> Dict:
        """
        计算多物理耦合专用指标：
        - coupling_iterations: 耦合迭代次数
        - field_conservation: 各场守恒性
        - interface_continuity: 界面连续性
        - load_balance: 负载平衡（不同物理场）
        """
        metrics = {}
        
        try:
            # 读取多个场
            fields = ['u', 'p', 'T', 'rho']  # 可能的场变量
            available_fields = []
            
            for field_name in fields:
                field_file = self.agent_output_dir / f'{field_name}.npy'
                if field_file.exists():
                    available_fields.append(field_name)
            
            metrics['available_fields'] = available_fields
            metrics['n_fields'] = len(available_fields)
            
            # 读取耦合迭代信息
            meta_file = self.agent_output_dir / 'meta.json'
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                
                if 'coupling_iterations' in meta:
                    iters = meta['coupling_iterations']
                    if isinstance(iters, list):
                        metrics['coupling_iterations_mean'] = float(np.mean(iters))
                        metrics['coupling_iterations_max'] = int(np.max(iters))
                    else:
                        metrics['coupling_iterations'] = iters
            
            # 检查界面连续性（如果有界面数据）
            interface_file = self.agent_output_dir / 'interface_jump.npy'
            if interface_file.exists():
                jump = np.load(interface_file)
                metrics['interface_jump_L2'] = float(np.linalg.norm(jump))
                metrics['interface_jump_max'] = float(np.max(np.abs(jump)))
            
            # 读取求解器信息
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute specialized metrics: {str(e)}"
        
        return metrics
