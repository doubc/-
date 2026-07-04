"""
shakespeare_experiment_v2.py — 莎士比亚实验 V2：生成 vs 搜索

核心修正：
  V1 的错误是把差异引擎也当成搜索算法——让它去匹配一个外部目标。
  但差异引擎不是搜索器，它是生成器。
  
  正确的实验设计：
  - 猴子打印机：纯随机比特，无结构（噪声基线）
  - 最小变易+梯度：有方向搜索，能找到目标但不生成新结构
  - 差异引擎：九机制驱动，自发生成层级结构
  
  比较的不是"谁能找到莎士比亚"，而是：
  1. 谁能生成最复杂的内部结构？
  2. 谁的信息积累率最高？
  3. 谁能自发涌现出层级组织？
  4. 在有目标引导时，谁最快到达？

理论预测：
  - 猴子：无结构，信息量恒定在~1 bit/位
  - 梯度搜索：快速到达目标，但无内部结构
  - 差异引擎：生成丰富的层级结构，信息量随时间增长
  
  "打印莎士比亚"不是找到一个比特串，
  而是生成一个能产生莎士比亚的结构。
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M
from diffsim.metrics import jaccard_flux


# ============================================================
# 测量工具
# ============================================================

def shannon_entropy(state: np.ndarray) -> float:
    """Shannon 熵 (bit)。"""
    if len(state) == 0:
        return 0.0
    p1 = np.mean(state)
    p0 = 1 - p1
    if p0 < 1e-10 or p1 < 1e-10:
        return 0.0
    return float(-(p0 * np.log2(p0) + p1 * np.log2(p1)))


def count_clusters(state: np.ndarray) -> int:
    """活跃比特中的聚簇数（连续1的段数，环形）。"""
    if len(state) == 0:
        return 0
    s = np.asarray(state, dtype=int)
    extended = np.concatenate([s, s[:1]])
    transitions = int(np.sum(np.abs(np.diff(extended))))
    return transitions // 2


def structural_complexity(state: np.ndarray) -> float:
    """结构复杂度 = 聚簇数 × 聚簇大小的标准差。
    
    高复杂度 = 很多大小不一的聚簇（有层次的结构）
    低复杂度 = 要么全是0/1，要么均匀分布
    """
    n_clusters = count_clusters(state)
    if n_clusters == 0:
        return 0.0
    
    # 找到每个聚簇的大小
    s = np.asarray(state, dtype=int)
    sizes = []
    in_cluster = False
    size = 0
    for i in range(len(s) * 2):  # 环形遍历两圈
        idx = i % len(s)
        if s[idx] == 1:
            if not in_cluster:
                in_cluster = True
                size = 1
            else:
                size += 1
        else:
            if in_cluster:
                sizes.append(size)
                in_cluster = False
                size = 0
        if i >= len(s) and not in_cluster:
            break
    if in_cluster:
        sizes.append(size)
    
    if len(sizes) == 0:
        return 0.0
    
    # 复杂度 = 聚簇数 × 大小变异系数
    mean_size = np.mean(sizes)
    if mean_size < 1e-10:
        return 0.0
    cv = np.std(sizes) / mean_size if len(sizes) > 1 else 0.0
    return float(n_clusters * (1 + cv))


def mutual_information(state1: np.ndarray, state2: np.ndarray) -> float:
    """两个状态之间的互信息。"""
    if len(state1) != len(state2) or len(state1) == 0:
        return 0.0
    # 联合分布
    joint = np.zeros((2, 2))
    for a, b in zip(state1, state2):
        joint[int(a), int(b)] += 1
    joint /= len(state1)
    
    # 边际分布
    p1 = joint.sum(axis=1)
    p2 = joint.sum(axis=0)
    
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 1e-10 and p1[i] > 1e-10 and p2[j] > 1e-10:
                mi += joint[i, j] * np.log2(joint[i, j] / (p1[i] * p2[j]))
    return float(mi)


# ============================================================
# 条件A：纯随机（猴子打印机）
# ============================================================

def run_monkey(N: int, max_steps: int, seed: int) -> Dict:
    """纯随机比特翻转，无记忆无积累。"""
    rng = np.random.RandomState(seed)
    state = (rng.random(N) < 0.5).astype(int)
    initial_state = state.copy()
    
    log = {'entropy': [], 'clusters': [], 'complexity': [], 'steps': []}
    
    for step in range(max_steps):
        # 随机翻转
        pos = rng.randint(0, N)
        state[pos] = 1 - state[pos]
        
        if step % 100 == 0:
            log['steps'].append(step)
            log['entropy'].append(shannon_entropy(state))
            log['clusters'].append(count_clusters(state))
            log['complexity'].append(structural_complexity(state))
    
    return {
        'method': 'monkey',
        'final_state': state,
        'initial_state': initial_state,
        'log': log,
        'info_with_initial': mutual_information(initial_state, state),
    }


# ============================================================
# 条件B：有目标引导的梯度搜索
# ============================================================

def run_gradient_search(N: int, max_steps: int, seed: int, 
                        target: np.ndarray) -> Dict:
    """梯度搜索：每步选最能靠近目标的翻转。"""
    rng = np.random.RandomState(seed)
    state = (rng.random(N) < 0.5).astype(int)
    initial_state = state.copy()
    
    log = {'entropy': [], 'clusters': [], 'complexity': [], 'steps': [],
           'match': []}
    
    for step in range(max_steps):
        hamming = np.sum(state != target)
        
        # 找最佳翻转
        best_pos = -1
        best_hamming = hamming
        candidates = []
        
        for pos in range(N):
            new_h = hamming + (1 if state[pos] == target[pos] else -1)
            if new_h < best_hamming:
                best_hamming = new_h
                candidates = [pos]
            elif new_h == best_hamming:
                candidates.append(pos)
        
        if candidates:
            pos = rng.choice(candidates)
        else:
            pos = rng.randint(0, N)
        
        state[pos] = 1 - state[pos]
        
        if step % 100 == 0:
            log['steps'].append(step)
            log['entropy'].append(shannon_entropy(state))
            log['clusters'].append(count_clusters(state))
            log['complexity'].append(structural_complexity(state))
            log['match'].append(np.sum(state == target) / N)
    
    return {
        'method': 'gradient_search',
        'final_state': state,
        'initial_state': initial_state,
        'log': log,
        'info_with_initial': mutual_information(initial_state, state),
        'final_match': float(np.sum(state == target) / N),
    }


# ============================================================
# 条件C：差异引擎（九机制生成）
# ============================================================

def run_engine(N: int, max_steps: int, seed: int) -> Dict:
    """差异引擎：九机制驱动的结构生成。"""
    rng = np.random.RandomState(seed)
    initial_active = np.where(rng.random(N) < 0.5)[0].tolist()
    
    field = DifferenceField(N=N, active=initial_active, rng=rng)
    
    # 初始化绑定
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    
    from diffsim.world_v2 import Layer, Params
    params = Params(max_steps=max_steps)
    layer = Layer(field, params)
    
    initial_state = field.state.copy()
    
    log = {'entropy': [], 'clusters': [], 'complexity': [], 'steps': [],
           'n_orgs': [], 'binding_energy': [], 'lock_level': []}
    
    for step in range(max_steps):
        layer.step = step
        
        M.m1_clustering(layer)
        M.m2_hierarchy(layer)
        M.m3_conservation(layer)
        M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer)
        M.m6_breaking(layer)
        M.m7_cycle(layer)
        M.m8_locking(layer)
        
        if step % 100 == 0:
            log['steps'].append(step)
            log['entropy'].append(shannon_entropy(field.state))
            log['clusters'].append(count_clusters(field.state))
            log['complexity'].append(structural_complexity(field.state))
            log['n_orgs'].append(len(layer.tentative_orgs) if layer.tentative_orgs else 0)
            log['binding_energy'].append(float(np.sum(field.binding) / 2))
            log['lock_level'].append(float(np.mean(field.lock_level)))
    
    return {
        'method': 'difference_engine',
        'final_state': field.state,
        'initial_state': initial_state,
        'log': log,
        'info_with_initial': mutual_information(initial_state, field.state),
    }


# ============================================================
# 条件D：差异引擎 + 目标引导
# ============================================================

def run_engine_guided(N: int, max_steps: int, seed: int,
                      target: np.ndarray) -> Dict:
    """差异引擎 + 目标引导：九机制 + 梯度偏置。
    
    这是真正的"地球打印莎士比亚"条件：
    有内部结构积累（九机制），也有外部选择压力（目标）。
    """
    rng = np.random.RandomState(seed)
    initial_active = np.where(rng.random(N) < 0.5)[0].tolist()
    
    field = DifferenceField(N=N, active=initial_active, rng=rng)
    
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    
    from diffsim.world_v2 import Layer, Params
    params = Params(max_steps=max_steps)
    layer = Layer(field, params)
    
    initial_state = field.state.copy()
    
    log = {'entropy': [], 'clusters': [], 'complexity': [], 'steps': [],
           'match': [], 'n_orgs': [], 'binding_energy': []}
    
    for step in range(max_steps):
        layer.step = step
        
        M.m1_clustering(layer)
        M.m2_hierarchy(layer)
        M.m3_conservation(layer)
        M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer)
        M.m6_breaking(layer)
        M.m7_cycle(layer)
        M.m8_locking(layer)
        
        # 目标引导偏置：每50步，对最偏离目标的位置施加微弱推力
        if step % 50 == 0 and step > 0:
            mismatches = np.where(field.state != target)[0]
            if len(mismatches) > 0:
                # 选一个最"自由"的（未锁定的） mismatched 位
                free_mismatches = [b for b in mismatches 
                                   if b not in field.sealed_bits]
                if free_mismatches:
                    # 优先翻转有同色邻居支持的位置
                    best = max(free_mismatches, 
                              key=lambda b: np.sum(field.binding[b, :]))
                    field.state[best] = 1 - field.state[best]
        
        if step % 100 == 0:
            log['steps'].append(step)
            log['entropy'].append(shannon_entropy(field.state))
            log['clusters'].append(count_clusters(field.state))
            log['complexity'].append(structural_complexity(field.state))
            log['match'].append(float(np.sum(field.state == target) / N))
            log['n_orgs'].append(len(layer.tentative_orgs) if layer.tentative_orgs else 0)
            log['binding_energy'].append(float(np.sum(field.binding) / 2))
    
    return {
        'method': 'engine_guided',
        'final_state': field.state,
        'initial_state': initial_state,
        'log': log,
        'info_with_initial': mutual_information(initial_state, field.state),
        'final_match': float(np.sum(field.state == target) / N),
    }


# ============================================================
# 主实验
# ============================================================

def make_target(N: int, seed: int = 42) -> np.ndarray:
    """构造有结构的目标模式。"""
    rng = np.random.RandomState(seed)
    bits = np.zeros(N, dtype=int)
    n_clusters = rng.randint(4, 8)
    phi = (1 + np.sqrt(5)) / 2
    pos = rng.randint(0, N // n_clusters)
    for _ in range(n_clusters):
        size = rng.randint(3, 6)
        for j in range(size):
            bits[(pos + j) % N] = 1
        pos = (pos + int(N / n_clusters * phi)) % N
    return bits


def run_full_experiment(N: int = 48, max_steps: int = 30000,
                        n_trials: int = 3, seed: int = 42) -> Dict:
    """完整四条件对比实验。"""
    
    print("=" * 70)
    print("莎士比亚实验 V2：生成 vs 搜索")
    print("=" * 70)
    print(f"N={N}, max_steps={max_steps}, trials={n_trials}")
    print()
    
    target = make_target(N, seed=seed)
    print(f"目标: {np.sum(target)} ones, {count_clusters(target)} clusters")
    print()
    
    all_results = {}
    
    for trial in range(n_trials):
        s = seed + trial * 1000
        print(f"--- Trial {trial+1}/{n_trials} (seed={s}) ---")
        
        t0 = time.time()
        monkey = run_monkey(N, max_steps, s)
        print(f"  猴子:     entropy={monkey['log']['entropy'][-1]:.3f}, "
              f"clusters={monkey['log']['clusters'][-1]}, "
              f"complexity={monkey['log']['complexity'][-1]:.2f} "
              f"({time.time()-t0:.1f}s)")
        
        t0 = time.time()
        gradient = run_gradient_search(N, max_steps, s, target)
        print(f"  梯度搜索: match={gradient['final_match']:.3f}, "
              f"complexity={gradient['log']['complexity'][-1]:.2f} "
              f"({time.time()-t0:.1f}s)")
        
        t0 = time.time()
        engine = run_engine(N, max_steps, s)
        print(f"  差异引擎: entropy={engine['log']['entropy'][-1]:.3f}, "
              f"clusters={engine['log']['clusters'][-1]}, "
              f"complexity={engine['log']['complexity'][-1]:.2f}, "
              f"orgs={engine['log']['n_orgs'][-1]}, "
              f"binding={engine['log']['binding_energy'][-1]:.1f} "
              f"({time.time()-t0:.1f}s)")
        
        t0 = time.time()
        guided = run_engine_guided(N, max_steps, s, target)
        print(f"  引擎+目标: match={guided['final_match']:.3f}, "
              f"complexity={guided['log']['complexity'][-1]:.2f}, "
              f"orgs={guided['log']['n_orgs'][-1]} "
              f"({time.time()-t0:.1f}s)")
        print()
        
        all_results[f"trial_{trial}"] = {
            'monkey': monkey,
            'gradient': gradient,
            'engine': engine,
            'guided': guided,
        }
    
    # 汇总
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    
    for method in ['monkey', 'gradient', 'engine', 'guided']:
        complexities = [all_results[f'trial_{t}'][method]['log']['complexity'][-1]
                       for t in range(n_trials)]
        print(f"\n{method}:")
        print(f"  最终复杂度: {np.mean(complexities):.2f} ± {np.std(complexities):.2f}")
        
        if method in ['gradient', 'guided']:
            matches = [all_results[f'trial_{t}'][method]['final_match']
                      for t in range(n_trials)]
            print(f"  最终匹配度: {np.mean(matches):.3f} ± {np.std(matches):.3f}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=48)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_full_experiment(N=args.N, max_steps=args.steps,
                       n_trials=args.trials, seed=args.seed)
