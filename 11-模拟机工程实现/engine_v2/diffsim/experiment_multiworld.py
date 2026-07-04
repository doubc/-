"""
experiment_multiworld.py — 多世界实验：原子结构是否唯一？

核心问题：
  1. 不同种子是否产生相同的原子结构？
  2. 不同参数（衰减指数、绑定强度）是否产生不同的结构？
  3. 是否存在"其他样式"的世界——不是原子，而是完全不同的组织方式？

实验设计：
  条件A：同参数不同种子 → 测试结构的随机性
  条件B：同种子不同参数 → 测试结构对参数的敏感性
  条件C：大尺度（N=512）→ 看能否出现更复杂的分子

测量：
  - 壳层数分布
  - 壳层中心数分布
  - 中心间距离分布
  - 绑定矩阵的特征值分布
  - 结构的"相似度"——不同世界之间的结构有多像？
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


def m1_multiplicative(layer, alpha=0.05, decay_power=2.0):
    f = layer.field
    act = np.where(f.state == 1)[0]
    if len(act) < 2: return
    for i_idx in range(len(act)):
        for j_idx in range(i_idx + 1, len(act)):
            i, j = act[i_idx], act[j_idx]
            d = hamming_distance(i, j)
            if d == 0: continue
            same = f.color[i] == f.color[j]
            factor = 1.0 + (alpha if same else alpha * 0.15) / (d ** decay_power)
            f.binding[i, j] *= factor
            f.binding[j, i] *= factor
    np.clip(f.binding, 0.0, 1e6, out=f.binding)


def analyze_world(field, label=""):
    """分析一个世界的结构特征。"""
    active = np.where(field.state == 1)[0]
    
    # 壳层中心
    binding_strength = []
    for b in active:
        strength = float(np.sum(field.binding[b, :]))
        binding_strength.append(strength)
    
    if not binding_strength:
        return None
    
    # 绑定矩阵特征值
    if len(active) >= 3:
        sub = field.binding[np.ix_(active, active)]
        try:
            eigenvalues = np.sort(np.linalg.eigvalsh(sub))[::-1]
        except:
            eigenvalues = np.array([0])
    else:
        eigenvalues = np.array([0])
    
    # 中心间距离
    sorted_indices = np.argsort(binding_strength)[::-1]
    top_centers = active[sorted_indices[:5]]
    center_distances = []
    for i in range(len(top_centers)):
        for j in range(i+1, len(top_centers)):
            d = hamming_distance(top_centers[i], top_centers[j])
            center_distances.append(d)
    
    return {
        'label': label,
        'n_active': len(active),
        'n_sealed': len(field.sealed_bits),
        'binding_total': float(np.sum(field.binding) / 2),
        'binding_mean': float(np.mean(binding_strength)),
        'binding_std': float(np.std(binding_strength)),
        'eigenvalues': eigenvalues.tolist()[:10],
        'center_distances': sorted(center_distances),
        'top5_strengths': sorted(binding_strength, reverse=True)[:5],
    }


def run_world(N, seed, steps, alpha=0.05, decay_power=2.0):
    """运行一个世界。"""
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                d = hamming_distance(i, j)
                if d > 0:
                    field.binding[i, j] = field.binding[j, i] = 1.0 / (d ** decay_power)
    layer = Layer(field, Params(max_steps=steps))
    for step in range(steps):
        layer.step = step
        m1_multiplicative(layer, alpha=alpha, decay_power=decay_power)
        M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        m5_tension_driven(layer, tension_threshold=0.2, max_tension_flips=2)
        M.m6_breaking(layer); M.m7_cycle(layer); M.m8_locking(layer)
    return field


# ============================================================
# 条件A：同参数不同种子 → 结构是否唯一？
# ============================================================
print()
print("=" * 70)
print("  条件A：同参数不同种子 → 原子结构是否唯一？")
print("=" * 70)
print()

N = 128; steps = 2000; n_worlds = 20
worlds_A = []

for s in range(n_worlds):
    seed = s * 100 + 42
    field = run_world(N, seed, steps)
    info = analyze_world(field, f"seed={seed}")
    worlds_A.append(info)

print(f"  {'世界':>6s} | {'活跃':>4s} | {'密封':>4s} | {'总绑定':>10s} | {'中心强度':>10s} | {'中心距离':>10s}")
print("  " + "-" * 60)
for w in worlds_A:
    print(f"  {w['label']:>6s} | {w['n_active']:4d} | {w['n_sealed']:4d} | "
          f"{w['binding_total']:10.0f} | {w['top5_strengths'][0]:10.0f} | "
          f"{str(w['center_distances'][:3]):>10s}")

# 计算结构相似度
print()
print("  结构相似度分析：")
binding_totals = [w['binding_total'] for w in worlds_A]
center_counts = [len(w['center_distances']) for w in worlds_A]
print(f"  总绑定: {np.mean(binding_totals):.0f} ± {np.std(binding_totals):.0f} (变异系数 {np.std(binding_totals)/np.mean(binding_totals)*100:.1f}%)")
print(f"  中心距离数: {np.mean(center_counts):.1f} ± {np.std(center_counts):.1f}")

if np.std(binding_totals) / np.mean(binding_totals) < 0.2:
    print("  → 结构高度一致：不同种子产生相似的结构。原子结构是'必然'的。")
else:
    print("  → 结构差异显著：不同种子产生不同的结构。存在'多种可能的世界'。")


# ============================================================
# 条件B：同种子不同参数 → 不同物理定律的世界
# ============================================================
print()
print()
print("=" * 70)
print("  条件B：不同参数 → 不同'物理定律'的世界")
print("=" * 70)
print()

print(f"  {'参数':>12s} | {'活跃':>4s} | {'密封':>4s} | {'总绑定':>10s} | {'中心强度':>10s}")
print("  " + "-" * 50)

params_to_test = [
    ("α=0.01,d=2", 0.01, 2.0),
    ("α=0.05,d=2", 0.05, 2.0),
    ("α=0.10,d=2", 0.10, 2.0),
    ("α=0.05,d=1", 0.05, 1.0),
    ("α=0.05,d=3", 0.05, 3.0),
    ("α=0.05,d=4", 0.05, 4.0),
]

for label, alpha, dp in params_to_test:
    field = run_world(128, 42, 2000, alpha=alpha, decay_power=dp)
    info = analyze_world(field, label)
    print(f"  {label:>12s} | {info['n_active']:4d} | {info['n_sealed']:4d} | "
          f"{info['binding_total']:10.0f} | {info['top5_strengths'][0]:10.0f}")

print()
print("  如果不同参数产生不同的结构，")
print("  说明'物理定律'（参数）决定了世界的结构。")
print("  不同的参数 = 不同的世界 = 不同的原子。")


# ============================================================
# 条件C：大尺度 N=512 → 更复杂的分子
# ============================================================
print()
print()
print("=" * 70)
print("  条件C：大尺度 N=512 → 分子几何")
print("=" * 70)
print()

field_c = run_world(512, 42, 3000)
info_c = analyze_world(field_c, "N=512")

active_c = np.where(field_c.state == 1)[0]
binding_strength_c = [(b, float(np.sum(field_c.binding[b, :]))) for b in active_c]
binding_strength_c.sort(key=lambda x: x[1], reverse=True)

print(f"  活跃位: {info_c['n_active']}")
print(f"  密封位: {info_c['n_sealed']}")
print()
print("  前 8 个壳层中心（'原子'）:")
print(f"  {'排名':>4s} | {'bit':>4s} | {'强度':>10s} | {'壳层':>4s}")
print("  " + "-" * 30)

for rank, (bit, strength) in enumerate(binding_strength_c[:8]):
    # 计算该中心的壳层数
    shells = {}
    for b in active_c:
        d = hamming_distance(bit, b)
        if d <= 20:
            if d not in shells:
                shells[d] = []
            shells[d].append(b)
    n_shells = len(shells)
    print(f"  {rank:4d} | {bit:4d} | {strength:10.0f} | {n_shells:4d}")

# 中心间距离矩阵
print()
print("  中心间距离矩阵（前 6 个中心）:")
top6 = [bit for bit, _ in binding_strength_c[:6]]
print(f"  {'':>4s}", end="")
for bit in top6:
    print(f" {bit:>4d}", end="")
print()
for i, bi in enumerate(top6):
    print(f"  {bi:4d}", end="")
    for j, bj in enumerate(top6):
        d = hamming_distance(bi, bj)
        print(f" {d:4d}", end="")
    print()

# 中心间绑定强度
print()
print("  中心间绑定强度（前 6 个中心）:")
print(f"  {'':>4s}", end="")
for bit in top6:
    print(f" {bit:>8d}", end="")
print()
for i, bi in enumerate(top6):
    print(f"  {bi:4d}", end="")
    for j, bj in enumerate(top6):
        b = float(field_c.binding[bi, bj])
        print(f" {b:8.0f}", end="")
    print()

print()
print("  如果某些中心对的绑定显著高于其他对，")
print("  那就是'化学键'——分子结构的标志。")
print()
print("  差异即世界，语法即多元。")
