"""
experiment_distance_clustering.py — 距离依赖的聚簇

核心改动：不仅绑定初始化是距离依赖的，聚簇（m1）也是距离依赖的。
  当前 m1：同色位绑定 += 0.18（均匀）
  改后 m1：同色位绑定 += 0.18 / d_H(i,j)（距离衰减）

这实现了物理中的"近距相互作用更强"：
  - 近距离位之间的绑定增长快（强相互作用）
  - 远距离位之间的绑定增长慢（弱相互作用）
  - 壳层结构自然涌现
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


def m1_distance_dependent(layer, decay_power=1.0):
    """距离依赖的聚簇。
    
    同色位绑定 += bind_inc / d_H(i,j)^decay_power
    
    decay_power=0: 均匀（当前行为）
    decay_power=1: 1/r 衰减
    decay_power=2: 1/r^2 衰减
    """
    f = layer.field
    act = np.where(f.state == 1)[0]
    if len(act) < 2:
        return
    
    for i_idx in range(len(act)):
        for j_idx in range(i_idx + 1, len(act)):
            i = act[i_idx]
            j = act[j_idx]
            d = hamming_distance(i, j)
            if d == 0:
                continue
            
            # 距离衰减
            decay = 1.0 / (d ** decay_power) if decay_power > 0 else 1.0
            
            # 同色优先
            same_color = f.color[i] == f.color[j]
            inc = layer.p.bind_inc * decay
            if not same_color:
                inc *= 0.15
            
            f.binding[i, j] += inc
            f.binding[j, i] += inc
    
    np.clip(f.binding, 0.0, layer.p.bind_cap, out=f.binding)


def measure_shells(field):
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


N = 96; steps = 2000; seed = 42

print()
print("=" * 70)
print("  距离依赖的聚簇 → 原子结构")
print("=" * 70)
print()

print(f"  {'衰减指数':>8s} | {'binding':>10s} | {'壳层':>4s} | {'密封':>4s} | {'活跃':>4s}")
print("  " + "-" * 40)

for decay in [0.0, 0.5, 1.0, 1.5, 2.0]:
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    # 距离依赖初始化
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                d = hamming_distance(i, j)
                if d > 0:
                    field.binding[i, j] = field.binding[j, i] = 1.0 / (d ** max(decay, 0.5))
    
    layer = Layer(field, Params(max_steps=steps))
    for step in range(steps):
        layer.step = step
        m1_distance_dependent(layer, decay_power=decay)
        M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        m5_tension_driven(layer, tension_threshold=0.2, max_tension_flips=2)
        M.m6_breaking(layer); M.m7_cycle(layer); M.m8_locking(layer)
    
    n_shells, _ = measure_shells(field)
    print(f"  {decay:8.1f} | {np.sum(field.binding)/2:10.1f} | {n_shells:4d} | "
          f"{len(field.sealed_bits):4d} | {int(field.state.sum()):4d}")

# 打印衰减=1.0 时的势场
print()
print("  势场形状（衰减指数=1.0）：")
rng = np.random.RandomState(seed)
active = np.where(rng.random(N) < 0.5)[0].tolist()
field = DifferenceField(N=N, active=active, rng=rng)
for i in range(N):
    for j in range(i+1, N):
        if field.color[i] == field.color[j]:
            d = hamming_distance(i, j)
            if d > 0:
                field.binding[i, j] = field.binding[j, i] = 1.0 / d

layer = Layer(field, Params(max_steps=steps))
for step in range(steps):
    layer.step = step
    m1_distance_dependent(layer, decay_power=1.0)
    M.m2_hierarchy(layer)
    M.m3_conservation(layer); M.m4_innate_completeness(layer)
    m5_tension_driven(layer, tension_threshold=0.2, max_tension_flips=2)
    M.m6_breaking(layer); M.m7_cycle(layer); M.m8_locking(layer)

pot = measure_potential(field)
print(f"  {'距离':>6s} | {'绑定':>10s} | {'1/d':>8s} | {'比值':>8s}")
print("  " + "-" * 35)
for d, b in sorted(pot.items())[:15]:
    inv_d = 1.0 / d if d > 0 else 0
    ratio = b / inv_d if inv_d > 0 else 0
    print(f"  {d:6d} | {b:10.4f} | {inv_d:8.4f} | {ratio:8.3f}")

# 壳层结构的特征值分布
n_shells, top_eigen = measure_shells(field)
print()
print(f"  壳层数: {n_shells}")
print(f"  前10个特征值: {[f'{e:.2f}' for e in top_eigen]}")
print()

if n_shells > 1:
    print("  ✅ 壳层结构出现！")
    print("  距离依赖的聚簇产生了离散的能级分组。")
else:
    print("  壳层结构未出现。需要调整参数。")

print()
print("  差异即世界，语法即原子。")
