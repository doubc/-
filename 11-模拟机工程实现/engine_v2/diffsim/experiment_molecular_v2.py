"""
experiment_molecular_v2.py — 分子结构 v2：乘法更新 + 更强衰减

核心改动：
  1. m1 从加法改为乘法：binding *= (1 + alpha/d)
     → 距离依赖被保留，不会被覆盖
  2. 衰减从 1/d 改为 1/d²
     → 近距离绑定更突出，远距离更弱
  3. N 推到 512
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


def m1_multiplicative_distance(layer, alpha=0.05, decay_power=2.0):
    """乘法更新的距离依赖聚簇。
    
    binding[i,j] *= (1 + alpha / d^decay_power)
    
    这样距离依赖会被保留：
    - 近距离：乘数大，绑定增长快
    - 远距离：乘数小，绑定增长慢
    """
    f = layer.field
    act = np.where(f.state == 1)[0]
    if len(act) < 2:
        return
    
    for i_idx in range(len(act)):
        for j_idx in range(i_idx + 1, len(act)):
            i, j = act[i_idx], act[j_idx]
            d = hamming_distance(i, j)
            if d == 0:
                continue
            
            # 乘法因子
            factor = 1.0 + alpha / (d ** decay_power)
            
            # 同色优先
            if f.color[i] != f.color[j]:
                factor = 1.0 + (alpha * 0.15) / (d ** decay_power)
            
            f.binding[i, j] *= factor
            f.binding[j, i] *= factor
    
    # 防止数值溢出
    np.clip(f.binding, 0.0, 1e6, out=f.binding)


def find_shell_centers(field, n_top=5):
    """找到绑定强度最高的位作为壳层中心。"""
    active = np.where(field.state == 1)[0]
    if len(active) == 0:
        return []
    
    binding_strength = []
    for b in active:
        strength = float(np.sum(field.binding[b, :]))
        binding_strength.append((b, strength))
    
    binding_strength.sort(key=lambda x: x[1], reverse=True)
    return binding_strength[:n_top]


def analyze_center(field, center_bit, max_radius=30):
    """分析某个中心的壳层结构。"""
    active = np.where(field.state == 1)[0]
    
    # 按距离分组
    shells = {}
    for b in active:
        d = hamming_distance(center_bit, b)
        if d <= max_radius:
            if d not in shells:
                shells[d] = []
            shells[d].append(b)
    
    # 每个壳层的平均绑定
    shell_info = []
    for d in sorted(shells.keys()):
        bits = shells[d]
        avg_bind = np.mean([field.binding[center_bit, b] for b in bits])
        shell_info.append((d, len(bits), avg_bind))
    
    return shell_info


N_values = [128, 256, 512]
steps = 3000
seed = 42

print()
print("=" * 70)
print("  分子结构 v2：乘法更新 + 1/d² 衰减")
print("=" * 70)
print()

for N in N_values:
    print(f"  === N={N} ===")
    
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    
    # 距离依赖初始化：1/d²
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                d = hamming_distance(i, j)
                if d > 0:
                    field.binding[i, j] = field.binding[j, i] = 1.0 / (d ** 2)
    
    layer = Layer(field, Params(max_steps=steps))
    for step in range(steps):
        layer.step = step
        m1_multiplicative_distance(layer, alpha=0.05, decay_power=2.0)
        M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        m5_tension_driven(layer, tension_threshold=0.2, max_tension_flips=2)
        M.m6_breaking(layer); M.m7_cycle(layer); M.m8_locking(layer)
    
    # 找壳层中心
    centers = find_shell_centers(field, n_top=5)
    
    print(f"  活跃位: {int(field.state.sum())}")
    print(f"  密封位: {len(field.sealed_bits)}")
    print()
    
    for ci, (bit, strength) in enumerate(centers):
        shell_info = analyze_center(field, bit, max_radius=20)
        print(f"  中心{ci}: bit={bit}, 总绑定={strength:.1f}")
        print(f"    {'距离':>4s} | {'位数':>4s} | {'平均绑定':>10s} | {'绑定×d²':>10s}")
        print("    " + "-" * 35)
        for d, count, avg_bind in shell_info[:8]:
            print(f"    {d:4d} | {count:4d} | {avg_bind:10.4f} | {avg_bind * d**2:10.4f}")
        print()
    
    # 中心间距离
    print(f"  中心间距离:")
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            d = hamming_distance(centers[i][0], centers[j][0])
            b = float(field.binding[centers[i][0], centers[j][0]])
            print(f"    {centers[i][0]}-{centers[j][0]}: 距离={d}, 绑定={b:.4f}")
    print()

print("  如果绑定×d² 在不同距离上保持大致相同，")
print("  说明势场是 1/r²（而非 1/r）。")
print("  如果绑定×d² 随距离增加，说明势场衰减比 1/r² 更快。")
print()
print("  差异即世界，语法即分子。")
