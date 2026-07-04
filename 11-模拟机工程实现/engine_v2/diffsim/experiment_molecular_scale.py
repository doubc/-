"""
experiment_molecular_scale.py — 大尺度实验：能否跑出分子结构？

目标：N=300~500，观察能否出现多原子相互作用的特征。

分子结构的关键特征：
  1. 多个独立的壳层中心（多个原子）
  2. 壳层之间的相互作用（化学键）
  3. 键长的离散化（量子化距离）
  4. 能量最小化（稳定结构）
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
    f = layer.field
    act = np.where(f.state == 1)[0]
    if len(act) < 2: return
    for i_idx in range(len(act)):
        for j_idx in range(i_idx + 1, len(act)):
            i, j = act[i_idx], act[j_idx]
            d = hamming_distance(i, j)
            if d == 0: continue
            decay = 1.0 / (d ** decay_power) if decay_power > 0 else 1.0
            same = f.color[i] == f.color[j]
            inc = layer.p.bind_inc * decay * (1.0 if same else 0.15)
            f.binding[i, j] += inc
            f.binding[j, i] += inc
    np.clip(f.binding, 0.0, layer.p.bind_cap, out=f.binding)


def find_shell_centers(field, threshold=100):
    """找到壳层中心——绑定强度最高的位。"""
    active = np.where(field.state == 1)[0]
    if len(active) == 0:
        return []
    
    # 每个活跃位的总绑定强度
    binding_strength = []
    for b in active:
        strength = float(np.sum(field.binding[b, :]))
        binding_strength.append((b, strength))
    
    # 按绑定强度排序
    binding_strength.sort(key=lambda x: x[1], reverse=True)
    
    # 找到"中心"——绑定强度显著高于平均值的位
    strengths = [s for _, s in binding_strength]
    mean_s = np.mean(strengths)
    std_s = np.std(strengths)
    
    centers = []
    for b, s in binding_strength:
        if s > mean_s + 2 * std_s:
            centers.append((b, s))
    
    return centers


def find_shell_structure_per_center(field, center_bit, radius=20):
    """找到某个中心附近的壳层结构。"""
    active = np.where(field.state == 1)[0]
    
    # 找到中心附近的活跃位
    nearby = []
    for b in active:
        d = hamming_distance(center_bit, b)
        if d <= radius:
            nearby.append((b, d))
    
    if len(nearby) < 2:
        return 0, []
    
    # 按距离分组
    distance_groups = {}
    for b, d in nearby:
        if d not in distance_groups:
            distance_groups[d] = []
        distance_groups[d].append(b)
    
    # 壳层 = 不同距离的组
    shells = []
    for d in sorted(distance_groups.keys()):
        bits = distance_groups[d]
        avg_binding = np.mean([field.binding[center_bit, b] for b in bits])
        shells.append((d, len(bits), avg_binding))
    
    return len(shells), shells


def measure_molecular_bonds(field, centers):
    """测量中心之间的"键"。"""
    bonds = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            ci, si = centers[i]
            cj, sj = centers[j]
            d = hamming_distance(ci, cj)
            bond_strength = float(field.binding[ci, cj])
            bonds.append({
                'center1': ci,
                'center2': cj,
                'distance': d,
                'strength': bond_strength,
                'strength1': si,
                'strength2': sj,
            })
    return bonds


N_values = [128, 256, 384]
steps = 3000
seed = 42

print()
print("=" * 70)
print("  大尺度实验：能否跑出分子结构？")
print("=" * 70)
print()

for N in N_values:
    print(f"  === N={N} ===")
    
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    
    # 距离依赖初始化
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
    
    # 找壳层中心
    centers = find_shell_centers(field)
    
    print(f"  活跃位: {int(field.state.sum())}")
    print(f"  密封位: {len(field.sealed_bits)}")
    print(f"  壳层中心数: {len(centers)}")
    
    if centers:
        print(f"  中心详情:")
        for ci, (bit, strength) in enumerate(centers[:5]):
            n_shells, shell_info = find_shell_structure_per_center(field, bit, radius=30)
            print(f"    中心{ci}: bit={bit}, 强度={strength:.1f}, 壳层={n_shells}")
            if shell_info:
                for d, count, avg_bind in shell_info[:4]:
                    print(f"      距离{d}: {count}个位, 平均绑定={avg_bind:.3f}")
        
        # 测量"键"
        bonds = measure_molecular_bonds(field, centers)
        if bonds:
            print(f"  中心间'键':")
            for bond in bonds[:5]:
                print(f"    {bond['center1']}-{bond['center2']}: "
                      f"距离={bond['distance']}, 强度={bond['strength']:.3f}")
    
    print()

print("  理论预测：")
print("  - N=128: 应该看到 1-2 个壳层中心")
print("  - N=256: 应该看到 2-3 个壳层中心，可能有'键'")
print("  - N=384: 应该看到更多中心，'键'更明显")
print()
print("  如果多个中心之间有'键'连接，")
print("  那就是分子结构的雏形。")
print()
print("  差异即世界，语法即分子。")
