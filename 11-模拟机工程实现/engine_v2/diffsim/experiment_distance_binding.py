"""
experiment_distance_binding.py — 距离依赖绑定 → 原子结构

核心改动：绑定初始化从均匀改为距离依赖。
  当前：binding[i,j] = 0.1（所有同色位，均匀）
  改后：binding[i,j] = 1.0 / d_H(i,j)（随汉明距离衰减）

理论依据：WorldBase 定理 G — Φ∝-1/d_H

预测：
  1. 壳层结构出现（不同距离的绑定形成离散分组）
  2. 势场是 1/r（绑定随距离衰减）
  3. D_eff≈3 更稳定
  4. 能量量子化出现（壳层间能量差离散）
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M
from diffsim.m5_tension import m5_tension_driven
from diffsim.world_v2 import Layer, Params


def hamming_distance(a, b):
    return bin(a ^ b).count('1')


def init_distance_binding(field, strength=1.0):
    """距离依赖的绑定初始化。
    
    binding[i,j] = strength / d_H(i,j)
    
    这实现了 WorldBase 定理 G：Φ∝-1/d_H
    """
    N = field.N
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                d = hamming_distance(i, j)
                if d > 0:
                    field.binding[i, j] = field.binding[j, i] = strength / d
                else:
                    field.binding[i, j] = field.binding[j, i] = strength


def measure_shells(field):
    """测量壳层结构。"""
    active = np.where(field.state == 1)[0]
    if len(active) < 3:
        return 0, []
    sub = field.binding[np.ix_(active, active)]
    try:
        eigenvalues = np.linalg.eigvalsh(sub)
        eigenvalues = np.sort(eigenvalues)[::-1]
        gaps = np.diff(eigenvalues)
        if len(gaps) > 0:
            threshold = np.mean(gaps) + 2 * np.std(gaps)
            n_shells = int(np.sum(gaps > threshold)) + 1
        else:
            n_shells = 1
        return n_shells, eigenvalues.tolist()[:10]
    except:
        return 0, []


def measure_potential(field):
    """测量势场形状。"""
    active = np.where(field.state == 1)[0]
    if len(active) < 3:
        return {}
    distance_binding = {}
    for i in range(min(len(active), 80)):
        for j in range(i+1, min(len(active), 80)):
            d = hamming_distance(active[i], active[j])
            b = field.binding[active[i], active[j]]
            if d not in distance_binding:
                distance_binding[d] = []
            distance_binding[d].append(b)
    return {d: float(np.mean(blist)) for d, blist in sorted(distance_binding.items())}


def run_engine(N, seed, steps, distance_binding=True, strength=1.0):
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    
    if distance_binding:
        init_distance_binding(field, strength=strength)
    else:
        for i in range(N):
            for j in range(i+1, N):
                if field.color[i] == field.color[j]:
                    field.binding[i, j] = field.binding[j, i] = 0.1
    
    layer = Layer(field, Params(max_steps=steps))
    for step in range(steps):
        layer.step = step
        M.m1_clustering(layer); M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        m5_tension_driven(layer, tension_threshold=0.2, max_tension_flips=2)
        M.m6_breaking(layer); M.m7_cycle(layer); M.m8_locking(layer)
    return field


# ============================================================
# 实验一：均匀绑定 vs 距离依赖绑定
# ============================================================
print()
print("=" * 70)
print("  距离依赖绑定 → 原子结构")
print("=" * 70)
print()

N = 96; steps = 2000; seed = 42

print("  实验一：均匀 vs 距离依赖（N=96）")
print()

# 均匀绑定
field_uniform = run_engine(N, seed, steps, distance_binding=False)
n_shells_u, eigen_u = measure_shells(field_uniform)
pot_u = measure_potential(field_uniform)

# 距离依赖绑定
field_distance = run_engine(N, seed, steps, distance_binding=True, strength=1.0)
n_shells_d, eigen_d = measure_shells(field_distance)
pot_d = measure_potential(field_distance)

print(f"  {'指标':>12s} | {'均匀绑定':>12s} | {'距离依赖':>12s}")
print("  " + "-" * 42)
print(f"  {'binding':>12s} | {np.sum(field_uniform.binding)/2:12.1f} | {np.sum(field_distance.binding)/2:12.1f}")
print(f"  {'lock':>12s} | {np.mean(field_uniform.lock_level):12.3f} | {np.mean(field_distance.lock_level):12.3f}")
print(f"  {'壳层':>12s} | {n_shells_u:12d} | {n_shells_d:12d}")
print(f"  {'密封':>12s} | {len(field_uniform.sealed_bits):12d} | {len(field_distance.sealed_bits):12d}")
print(f"  {'活跃':>12s} | {int(field_uniform.state.sum()):12d} | {int(field_distance.state.sum()):12d}")

print()
print("  势场对比（距离依赖绑定）：")
print(f"  {'距离':>6s} | {'绑定':>10s} | {'1/d':>8s} | {'比值':>8s}")
print("  " + "-" * 35)
for d, b in sorted(pot_d.items())[:12]:
    inv_d = 1.0 / d if d > 0 else 0
    ratio = b / inv_d if inv_d > 0 else 0
    print(f"  {d:6d} | {b:10.4f} | {inv_d:8.4f} | {ratio:8.3f}")

print()
print("  势场对比（均匀绑定）：")
print(f"  {'距离':>6s} | {'绑定':>10s}")
print("  " + "-" * 20)
for d, b in sorted(pot_u.items())[:12]:
    print(f"  {d:6d} | {b:10.4f}")


# ============================================================
# 实验二：不同强度的距离依赖绑定
# ============================================================
print()
print("=" * 70)
print("  实验二：不同强度的距离依赖绑定")
print("=" * 70)
print()

print(f"  {'强度':>6s} | {'binding':>10s} | {'壳层':>4s} | {'密封':>4s} | {'活跃':>4s}")
print("  " + "-" * 38)

for strength in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    field = run_engine(N, seed, steps, distance_binding=True, strength=strength)
    n_shells, _ = measure_shells(field)
    print(f"  {strength:6.1f} | {np.sum(field.binding)/2:10.1f} | {n_shells:4d} | "
          f"{len(field.sealed_bits):4d} | {int(field.state.sum()):4d}")


# ============================================================
# 实验三：不同 N 的距离依赖绑定
# ============================================================
print()
print("=" * 70)
print("  实验三：不同 N 的距离依赖绑定")
print("=" * 70)
print()

print(f"  {'N':>5s} | {'binding':>10s} | {'壳层':>4s} | {'密封':>4s} | {'活跃':>4s}")
print("  " + "-" * 38)

for N_val in [48, 96, 192, 384]:
    field = run_engine(N_val, seed, steps, distance_binding=True, strength=1.0)
    n_shells, _ = measure_shells(field)
    print(f"  {N_val:5d} | {np.sum(field.binding)/2:10.1f} | {n_shells:4d} | "
          f"{len(field.sealed_bits):4d} | {int(field.state.sum()):4d}")

print()
print("  如果壳层数随 N 增加而增加，")
print("  说明距离依赖绑定能够产生壳层结构。")
print()
print("  差异即世界，语法即原子。")
