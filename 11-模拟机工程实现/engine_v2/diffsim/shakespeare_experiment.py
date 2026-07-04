"""
shakespeare_experiment.py — 莎士比亚实验：九机制 vs 猴子打印机

核心问题：
  为什么地球在46亿年内打印出了莎士比亚全集，
  而不是需要无限时间的猴子打印机？

实验设计：
  三种条件，同一个目标——从随机初始状态生成目标结构。
  
  条件A：纯随机（猴子打印机）
    - 每步随机翻转1个比特，无记忆，无积累，无方向
  
  条件B：最小变易（只有A4）
    - 每步只翻转1个比特，但有汉明距离梯度引导
    - 有方向性，但无聚簇、无守恒、无锁定
  
  条件C：完整九机制（差异引擎）
    - 聚簇→层级→守恒→完备→变易→破缺→循环→锁定→自指
    - 有方向，有积累，有锁定，有层级压缩

测量指标：
  1. 到达目标的步数（效率）
  2. 信息积累曲线（累积匹配度随时间的变化）
  3. 结构复杂度曲线（活跃聚簇数随时间的变化）
  4. 搜索空间压缩率（有效搜索维度 vs 原始维度）

理论预测：
  - 条件A：几乎不可能到达目标（指数级时间）
  - 条件B：可以到达，但效率低（多项式时间，但系数大）
  - 条件C：高效到达（多项式时间，系数小，且有超线性加速）

作者: OpenClaw AI
日期: 2026-07-04
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import json
import os
import sys

# 确保可以导入 engine_v2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M
from diffsim.metrics import jaccard_flux


# ============================================================
# 目标结构定义：不是莎士比亚全集，而是有结构的比特模式
# ============================================================

@dataclass
class TargetStructure:
    """目标结构：一个有内部关联的比特模式。
    
    不是随机比特串，而是有"语法"的结构：
    - 多个局部聚簇（类似"单词"）
    - 聚簇之间的特定间距（类似"语法"）
    - 整体模式的对称性破缺（类似"意义"）
    """
    name: str
    bits: np.ndarray          # 目标比特串
    clusters: List[List[int]] # 聚簇位置（"单词"）
    description: str


def make_shakespeare_target(N: int = 64, seed: int = 42) -> TargetStructure:
    """构造一个有结构的目标模式，代表"莎士比亚"。
    
    结构特征：
    - 5-8个局部聚簇，每个聚簇3-6个连续1
    - 聚簇之间有特定间距（语法结构）
    - 整体不对称（不是简单的周期模式）
    """
    rng = np.random.RandomState(seed)
    bits = np.zeros(N, dtype=int)
    clusters = []
    
    # 选择聚簇位置：不均匀分布（模拟"语法"）
    n_clusters = rng.randint(5, 9)
    # 用黄金比例间距分配聚簇位置
    phi = (1 + np.sqrt(5)) / 2
    positions = []
    pos = rng.randint(0, N // n_clusters)
    for i in range(n_clusters):
        positions.append(pos % N)
        pos += int(N / n_clusters * phi) % N
    
    # 每个聚簇设置不同大小（模拟"词汇多样性"）
    for i, p in enumerate(positions):
        size = rng.randint(3, 7)
        cluster = [(p + j) % N for j in range(size)]
        for idx in cluster:
            bits[idx] = 1
        clusters.append(cluster)
    
    return TargetStructure(
        name="Shakespeare",
        bits=bits,
        clusters=clusters,
        description=f"{n_clusters} clusters, {np.sum(bits)} ones in {N} bits"
    )


def make_simple_target(N: int = 64, density: float = 0.3, seed: int = 42) -> TargetStructure:
    """简单目标：随机但固定密度的1。"""
    rng = np.random.RandomState(seed)
    bits = (rng.random(N) < density).astype(int)
    return TargetStructure(
        name="Simple",
        bits=bits,
        clusters=[list(np.where(bits == 1)[0])],
        description=f"{np.sum(bits)} ones in {N} bits"
    )


# ============================================================
# 条件A：纯随机（猴子打印机）
# ============================================================

def run_monkey_typewriter(
    target: TargetStructure,
    max_steps: int = 100_000,
    seed: int = 42,
) -> Dict:
    """猴子打印机：每步随机翻转1个比特，无记忆，无积累。"""
    rng = np.random.RandomState(seed)
    N = len(target.bits)
    state = (rng.random(N) < 0.5).astype(int)
    
    history = {
        'method': 'monkey',
        'steps': [],
        'match_curve': [],
        'complexity_curve': [],
        'hamming_curve': [],
    }
    
    best_match = np.sum(state == target.bits) / N
    best_state = state.copy()
    
    for step in range(max_steps):
        # 纯随机翻转
        pos = rng.randint(0, N)
        state[pos] = 1 - state[pos]
        
        # 计算匹配度
        match = np.sum(state == target.bits) / N
        hamming = np.sum(state != target.bits)
        
        # 记录（每100步采样一次）
        if step % 100 == 0:
            history['steps'].append(step)
            history['match_curve'].append(match)
            history['hamming_curve'].append(hamming)
            # 复杂度：活跃聚簇数（连续1的段数）
            complexity = count_clusters(state)
            history['complexity_curve'].append(complexity)
        
        if match > best_match:
            best_match = match
            best_state = state.copy()
        
        # 完美匹配
        if match >= 1.0:
            history['converged'] = True
            history['converged_at'] = step
            break
    else:
        history['converged'] = False
        history['converged_at'] = None
    
    history['best_match'] = best_match
    history['final_state'] = best_state.tolist()
    return history


# ============================================================
# 条件B：最小变易（只有A4，有方向引导）
# ============================================================

def run_minimal_variation(
    target: TargetStructure,
    max_steps: int = 100_000,
    seed: int = 42,
) -> Dict:
    """最小变易：每步只翻转1个比特，但有汉明距离梯度引导。
    
    策略：总是翻转最能减少汉明距离的那个比特。
    这是有方向的搜索，但没有聚簇、守恒、锁定。
    """
    rng = np.random.RandomState(seed)
    N = len(target.bits)
    state = (rng.random(N) < 0.5).astype(int)
    
    history = {
        'method': 'minimal_variation',
        'steps': [],
        'match_curve': [],
        'complexity_curve': [],
        'hamming_curve': [],
    }
    
    for step in range(max_steps):
        # 计算每个位置翻转后的汉明距离
        hamming_now = np.sum(state != target.bits)
        
        # 找到最佳翻转位置
        best_pos = -1
        best_hamming = hamming_now
        candidates = []
        
        for pos in range(N):
            # 翻转这个位置
            new_hamming = hamming_now
            if state[pos] == target.bits[pos]:
                new_hamming += 1  # 翻转后变差
            else:
                new_hamming -= 1  # 翻转后变好
            
            if new_hamming < best_hamming:
                best_hamming = new_hamming
                candidates = [pos]
            elif new_hamming == best_hamming:
                candidates.append(pos)
        
        if candidates:
            # 在最优候选中随机选一个
            pos = rng.choice(candidates)
        else:
            # 所有翻转都会变差，随机选一个（探索）
            pos = rng.randint(0, N)
        
        state[pos] = 1 - state[pos]
        
        # 计算匹配度
        match = np.sum(state == target.bits) / N
        hamming = np.sum(state != target.bits)
        
        # 记录
        if step % 100 == 0:
            history['steps'].append(step)
            history['match_curve'].append(match)
            history['hamming_curve'].append(hamming)
            complexity = count_clusters(state)
            history['complexity_curve'].append(complexity)
        
        if match >= 1.0:
            history['converged'] = True
            history['converged_at'] = step
            break
    else:
        history['converged'] = False
        history['converged_at'] = None
    
    history['best_match'] = match
    history['final_state'] = state.tolist()
    return history


# ============================================================
# 条件C：完整九机制（差异引擎）
# ============================================================

def run_difference_engine(
    target: TargetStructure,
    max_steps: int = 100_000,
    N: int = 64,
    seed: int = 42,
    n_colors: int = 4,
) -> Dict:
    """差异引擎：完整九机制驱动。
    
    使用 engine_v2 的 DifferenceField + 九机制。
    在每步中，九机制共同作用于差异场。
    """
    rng = np.random.RandomState(seed)
    
    # 初始化差异场
    state = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=state)
    
    # 为同色位设置基础绑定
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    
    # 创建 Layer 对象
    from diffsim.world_v2 import Layer, Params
    params = Params(
        max_steps=max_steps,
        bind_inc=0.18,
        bind_threshold=1.0,
        min_org_size=3,
        seal_fraction=0.6,
        lock_inc=0.12,
        lock_threshold=0.6,
    )
    layer = Layer(field, params)
    
    history = {
        'method': 'difference_engine',
        'steps': [],
        'match_curve': [],
        'complexity_curve': [],
        'hamming_curve': [],
        'cluster_count_curve': [],
        'binding_energy_curve': [],
    }
    
    best_match = np.sum(field.state == target.bits) / N
    
    for step in range(max_steps):
        layer.step = step
        
        # 执行九机制（m1-m8，m9在递归世界中调用）
        M.m1_clustering(layer)      # 聚簇
        M.m2_hierarchy(layer)       # 层级
        M.m3_conservation(layer)    # 守恒
        M.m4_innate_completeness(layer)  # 先天完备性
        M.m5_minimal_variation(layer)    # 最小变易
        M.m6_breaking(layer)        # 破缺
        M.m7_cycle(layer)           # 循环
        M.m8_locking(layer)         # 锁定
        
        # 计算匹配度
        match = np.sum(field.state == target.bits) / N
        hamming = np.sum(field.state != target.bits)
        
        # 记录
        if step % 100 == 0:
            history['steps'].append(step)
            history['match_curve'].append(match)
            history['hamming_curve'].append(hamming)
            complexity = count_clusters(field.state)
            history['complexity_curve'].append(complexity)
            # 聚簇数量
            n_clusters = len(layer.tentative_orgs) if layer.tentative_orgs else 0
            history['cluster_count_curve'].append(n_clusters)
            # 绑定能量
            binding_energy = np.sum(field.binding) / 2
            history['binding_energy_curve'].append(binding_energy)
        
        if match > best_match:
            best_match = match
        
        if match >= 1.0:
            history['converged'] = True
            history['converged_at'] = step
            break
    else:
        history['converged'] = False
        history['converged_at'] = None
    
    history['best_match'] = best_match
    history['final_state'] = field.state.tolist()
    return history


# ============================================================
# 辅助函数
# ============================================================

def count_clusters(state: np.ndarray) -> int:
    """计算活跃比特中的聚簇数（连续1的段数）。"""
    if len(state) == 0:
        return 0
    # 环形处理
    extended = np.concatenate([state, state[:1]])
    transitions = np.sum(np.abs(np.diff(extended)))
    return int(transitions // 2)


def information_content(state: np.ndarray) -> float:
    """计算状态的信息内容（Shannon熵）。"""
    if len(state) == 0:
        return 0.0
    p1 = np.mean(state)
    p0 = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0.0
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))


# ============================================================
# 主实验
# ============================================================

def run_experiment(
    N: int = 64,
    max_steps: int = 50_000,
    n_trials: int = 5,
    seed: int = 42,
) -> Dict:
    """运行完整实验：三种条件对比。"""
    
    print("=" * 70)
    print("莎士比亚实验：九机制 vs 猴子打印机")
    print("=" * 70)
    print(f"N={N}, max_steps={max_steps}, trials={n_trials}")
    print()
    
    results = {
        'config': {'N': N, 'max_steps': max_steps, 'n_trials': n_trials},
        'conditions': {},
    }
    
    # 构造目标
    target = make_shakespeare_target(N=N, seed=seed)
    print(f"目标结构: {target.name} — {target.description}")
    print(f"目标比特: {target.bits.tolist()}")
    print(f"目标聚簇数: {len(target.clusters)}")
    print()
    
    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        
        # 条件A：猴子打印机
        print(f"Trial {trial+1}/{n_trials} — 条件A: 猴子打印机...")
        t0 = time.time()
        monkey = run_monkey_typewriter(target, max_steps=max_steps, seed=trial_seed)
        t_monkey = time.time() - t0
        print(f"  收敛: {monkey['converged']}, 最佳匹配: {monkey['best_match']:.4f}, 耗时: {t_monkey:.2f}s")
        
        # 条件B：最小变易
        print(f"Trial {trial+1}/{n_trials} — 条件B: 最小变易...")
        t0 = time.time()
        minimal = run_minimal_variation(target, max_steps=max_steps, seed=trial_seed)
        t_minimal = time.time() - t0
        print(f"  收敛: {minimal['converged']}, 最佳匹配: {minimal['best_match']:.4f}, 耗时: {t_minimal:.2f}s")
        
        # 条件C：差异引擎
        print(f"Trial {trial+1}/{n_trials} — 条件C: 差异引擎...")
        t0 = time.time()
        engine = run_difference_engine(target, max_steps=max_steps, N=N, seed=trial_seed)
        t_engine = time.time() - t0
        print(f"  收敛: {engine['converged']}, 最佳匹配: {engine['best_match']:.4f}, 耗时: {t_engine:.2f}s")
        print()
        
        # 存储结果
        key = f"trial_{trial}"
        results['conditions'][key] = {
            'monkey': {
                'converged': monkey['converged'],
                'converged_at': monkey['converged_at'],
                'best_match': monkey['best_match'],
                'match_curve': monkey['match_curve'],
                'complexity_curve': monkey['complexity_curve'],
            },
            'minimal_variation': {
                'converged': minimal['converged'],
                'converged_at': minimal['converged_at'],
                'best_match': minimal['best_match'],
                'match_curve': minimal['match_curve'],
                'complexity_curve': minimal['complexity_curve'],
            },
            'difference_engine': {
                'converged': engine['converged'],
                'converged_at': engine['converged_at'],
                'best_match': engine['best_match'],
                'match_curve': engine['match_curve'],
                'complexity_curve': engine['complexity_curve'],
                'cluster_count_curve': engine.get('cluster_count_curve', []),
                'binding_energy_curve': engine.get('binding_energy_curve', []),
            },
        }
    
    # 汇总统计
    print("=" * 70)
    print("汇总统计")
    print("=" * 70)
    
    for method in ['monkey', 'minimal_variation', 'difference_engine']:
        matches = [results['conditions'][f'trial_{t}'][method]['best_match'] 
                   for t in range(n_trials)]
        converges = [results['conditions'][f'trial_{t}'][method]['converged'] 
                     for t in range(n_trials)]
        print(f"\n{method}:")
        print(f"  平均最佳匹配: {np.mean(matches):.4f} ± {np.std(matches):.4f}")
        print(f"  收敛率: {sum(converges)}/{n_trials}")
    
    return results


def main():
    """主入口。"""
    import argparse
    parser = argparse.ArgumentParser(description='莎士比亚实验')
    parser.add_argument('--N', type=int, default=64, help='比特数')
    parser.add_argument('--steps', type=int, default=50000, help='最大步数')
    parser.add_argument('--trials', type=int, default=3, help='试验次数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default=None, help='输出文件')
    args = parser.parse_args()
    
    results = run_experiment(
        N=args.N,
        max_steps=args.steps,
        n_trials=args.trials,
        seed=args.seed,
    )
    
    if args.output:
        # 保存结果（只保存可序列化的部分）
        save_results = {
            'config': results['config'],
            'conditions': {}
        }
        for key, cond in results['conditions'].items():
            save_results['conditions'][key] = {}
            for method, data in cond.items():
                save_results['conditions'][key][method] = {
                    k: v for k, v in data.items()
                    if not isinstance(v, np.ndarray)
                }
        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\n结果已保存到 {args.output}")


if __name__ == '__main__':
    main()
