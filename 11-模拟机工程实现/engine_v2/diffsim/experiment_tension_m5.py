"""
experiment_tension_m5.py — 内部张力驱动的 m5 实验

对比：
  条件A：标准 m5（a1_source 驱动，4 步后饱和）
  条件B：张力驱动 m5（聚簇边界张力驱动，持续运转）

测量：
  - binding energy 曲线
  - 翻转次数曲线
  - 结构深度指标

理论预测：
  条件A：4 步后翻转=0，binding 平坦
  条件B：张力持续驱动翻转，binding 持续增长
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M
from diffsim.m5_tension import m5_tension_driven, compute_tension
from diffsim.world_v2 import Layer, Params


def se(s):
    p1=np.mean(s); p0=1-p1
    if p0<1e-10 or p1<1e-10: return 0.0
    return float(-(p0*np.log2(p0)+p1*np.log2(p1)))

def cc(s):
    s=np.asarray(s,dtype=int); ext=np.concatenate([s,s[:1]])
    return int(np.sum(np.abs(np.diff(ext)))//2)

def structural_depth(field):
    """结构深度指标 = binding × lock × sealed_fraction。"""
    bind = float(np.sum(field.binding) / 2)
    lock = float(np.mean(field.lock_level))
    sealed_frac = len(field.sealed_bits) / max(field.N, 1)
    return bind * lock * (sealed_frac + 0.01)


N = 48; steps = 3000; trials = 3

print()
print("=" * 62)
print("  内部张力驱动的 m5 实验")
print("=" * 62)
print()

# === 条件A：标准 m5 ===
print("  条件A：标准 m5（a1_source 驱动）")
print(f"  {'Step':>5s} | {'binding':>10s} | {'翻转':>5s} | {'聚簇':>5s} | {'深度':>10s}")
print("  " + "-" * 45)

for t in range(trials):
    seed = 42 + t * 1000
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
        prev = field.state.copy()
        M.m1_clustering(layer); M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer); M.m6_breaking(layer)
        M.m7_cycle(layer); M.m8_locking(layer)
        flips = int(np.sum(field.state != prev))
        
        if step < 10 or step % 500 == 0:
            bind = float(np.sum(field.binding) / 2)
            depth = structural_depth(field)
            print(f"  {step:5d} | {bind:10.1f} | {flips:5d} | {cc(field.state):5d} | {depth:10.1f}")
    if t < trials - 1:
        print("  ...")

print()

# === 条件B：张力驱动 m5 ===
print("  条件B：张力驱动 m5（聚簇边界张力驱动）")
print(f"  {'Step':>5s} | {'binding':>10s} | {'翻转':>5s} | {'聚簇':>5s} | {'深度':>10s} | {'张力':>6s}")
print("  " + "-" * 55)

for t in range(trials):
    seed = 42 + t * 1000
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    layer = Layer(field, Params(max_steps=steps))
    
    tension_flips_total = 0
    
    for step in range(steps):
        layer.step = step
        prev = field.state.copy()
        M.m1_clustering(layer); M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        
        # 用张力驱动的 m5 替代标准 m5
        m5_tension_driven(layer, tension_threshold=0.5, max_tension_flips=2)
        
        M.m6_breaking(layer)
        M.m7_cycle(layer); M.m8_locking(layer)
        flips = int(np.sum(field.state != prev))
        
        if getattr(layer, '_tension_driven', False):
            tension_flips_total += flips
        
        if step < 10 or step % 500 == 0:
            bind = float(np.sum(field.binding) / 2)
            depth = structural_depth(field)
            tension = compute_tension(field)
            avg_tension = float(np.mean(tension))
            print(f"  {step:5d} | {bind:10.1f} | {flips:5d} | {cc(field.state):5d} | "
                  f"{depth:10.1f} | {avg_tension:6.3f}")
    
    print(f"    张力驱动翻转总计: {tension_flips_total}")
    if t < trials - 1:
        print("  ...")

print()
print("  对比分析：")
print("  条件A：标准 m5，4 步后翻转=0，binding 平坦")
print("  条件B：张力驱动 m5，聚簇边界持续产生翻转，binding 持续增长")
print()
print("  如果条件B 的 binding 显著高于条件A，")
print("  说明内部张力是引擎持续运转的真正驱动力。")
print()
print("  差异即世界，语法即张力。")
