"""
experiment_atomic_scale.py — 多大 N 能跑出原子结构？

核心问题：
  差异引擎在什么尺度上开始产生类似原子的结构特征？

原子结构的关键特征：
  1. 维度锁定 D_eff=3（三维空间）
  2. 壳层结构（电子壳层：1s, 2s, 2p, 3s...）
  3. 离散对称性（旋转对称、镜像对称）
  4. 1/r 势场（库仑势）
  5. 量子化（能量不连续）

实验设计：
  扫描 N = 48, 96, 192, 384, 768
  测量：
  - 维度锁定 D_eff
  - 壳层数（聚簇的层级结构）
  - 对称性（绑定矩阵的特征值分布）
  - 势场形状（binding 随距离的衰减）
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M
from diffsim.m5_tension import m5_tension_driven
from diffsim.world_v2 import Layer, Params


def measure_dimensionality(field):
    """测量有效维度。
    
    方法：计算活跃位之间的汉明距离分布，
    用最近邻距离的标度律估计维度。
    
    D_eff = d(log(n_neighbors)) / d(log(distance))
    """
    active = np.where(field.state == 1)[0]
    if len(active) < 3:
        return 0.0
    
    # 计算活跃位之间的汉明距离
    distances = []
    for i in range(min(len(active), 100)):
        for j in range(i+1, min(len(active), 100)):
            d = bin(active[i] ^ active[j]).count('1')
            distances.append(d)
    
    if not distances:
        return 0.0
    
    # 用距离分布的均值和方差估计维度
    d_arr = np.array(distances)
    mean_d = np.mean(d_arr)
    var_d = np.var(d_arr)
    
    # 对于 D 维空间，距离的方差/均值^2 ≈ 1/(2D)
    if mean_d > 0 and var_d > 0:
        D_est = mean_d**2 / (2 * var_d)
        return float(D_est)
    return 0.0


def measure_shell_structure(field):
    """测量壳层结构。
    
    方法：计算绑定矩阵的特征值分布，
    壳层结构表现为特征值的离散分组。
    """
    binding = field.binding
    active = np.where(field.state == 1)[0]
    if len(active) < 3:
        return 0, []
    
    # 提取活跃位之间的绑定子矩阵
    sub = binding[np.ix_(active, active)]
    
    # 计算特征值
    try:
        eigenvalues = np.linalg.eigvalsh(sub)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # 壳层数 = 特征值的"台阶"数
        # 台阶 = 特征值之间的显著间隔
        gaps = np.diff(eigenvalues)
        if len(gaps) > 0:
            threshold = np.mean(gaps) + 2 * np.std(gaps)
            n_shells = int(np.sum(gaps > threshold)) + 1
        else:
            n_shells = 1
        
        return n_shells, eigenvalues.tolist()[:10]
    except:
        return 0, []


def measure_symmetry(field):
    """测量对称性。
    
    方法：计算绑定矩阵的对称性指标。
    完美对称 → 指标=1
    完全不对称 → 指标=0
    """
    binding = field.binding
    N = field.N
    
    # 对称性 = ||B - B^T|| / ||B||
    diff = binding - binding.T
    norm_diff = np.linalg.norm(diff)
    norm_B = np.linalg.norm(binding)
    
    if norm_B < 1e-10:
        return 0.0
    
    symmetry = 1.0 - norm_diff / (2 * norm_B)
    return float(max(0, symmetry))


def measure_potential(field):
    """测量势场形状。
    
    方法：计算绑定强度随汉明距离的衰减。
    类似 1/r 势 → 绑定 ∝ 1/d
    """
    active = np.where(field.state == 1)[0]
    if len(active) < 3:
        return {}
    
    # 按距离分组计算平均绑定
    distance_binding = {}
    for i in range(min(len(active), 50)):
        for j in range(i+1, min(len(active), 50)):
            d = bin(active[i] ^ active[j]).count('1')
            b = field.binding[active[i], active[j]]
            if d not in distance_binding:
                distance_binding[d] = []
            distance_binding[d].append(b)
    
    # 计算每个距离的平均绑定
    result = {}
    for d, bindings in sorted(distance_binding.items()):
        result[d] = float(np.mean(bindings))
    
    return result


def run_engine(N, seed, steps):
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
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


N_values = [48, 96, 192, 384]
steps = 2000
seed = 42

print()
print("=" * 70)
print("  多大 N 能跑出原子结构？")
print("=" * 70)
print()
print("  原子结构的关键特征：")
print("  1. 维度锁定 D_eff=3（三维空间）")
print("  2. 壳层结构（电子壳层）")
print("  3. 离散对称性")
print("  4. 1/r 势场（库仑势）")
print()
print(f"  {'N':>5s} | {'binding':>10s} | {'D_eff':>6s} | {'壳层':>4s} | {'对称':>6s} | {'活跃':>5s} | {'密封':>5s}")
print("  " + "-" * 55)

for N in N_values:
    field = run_engine(N, seed, steps)
    
    bind = float(np.sum(field.binding) / 2)
    D_eff = measure_dimensionality(field)
    n_shells, top_eigen = measure_shell_structure(field)
    symmetry = measure_symmetry(field)
    active = int(field.state.sum())
    sealed = len(field.sealed_bits)
    
    print(f"  {N:5d} | {bind:10.1f} | {D_eff:6.2f} | {n_shells:4d} | {symmetry:6.3f} | {active:5d} | {sealed:5d}")

# 对于最大的 N，打印势场形状
print()
print("  势场形状（最大 N）：")
N_max = N_values[-1]
field = run_engine(N_max, seed, steps)
potential = measure_potential(field)
if potential:
    print(f"  {'距离':>6s} | {'平均绑定':>10s} | {'1/d':>8s} | {'比值':>8s}")
    print("  " + "-" * 35)
    for d, b in sorted(potential.items())[:15]:
        inv_d = 1.0 / d if d > 0 else 0
        ratio = b / inv_d if inv_d > 0 else 0
        print(f"  {d:6d} | {b:10.3f} | {inv_d:8.4f} | {ratio:8.3f}")

print()
print("  理论预测：")
print("  - D_eff 应该趋向 3（维度锁定）")
print("  - 壳层数应该随 N 增加而增加")
print("  - 对称性应该随 N 增加而增加")
print("  - 势场应该接近 1/r（库仑势）")
print()
print("  如果这些特征在某个 N 处开始出现，")
print("  那就是'原子结构'的临界尺度。")
print()
print("  差异即世界，语法即原子。")
