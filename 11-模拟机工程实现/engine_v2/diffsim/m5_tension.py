"""
m5_tension.py — 内部张力驱动的最小变易

问题：当前 m5 在 a1_source 耗尽后不再工作（budget=0）。
修正：引入内部张力驱动——聚簇边界产生的张力可以触发新的翻转。

理论依据：
  差异论命题 2.2："聚簇形成的结构，不只是差异的结果，
  它还会反过来重塑差异本身。"
  
  聚簇不是终点。聚簇之间的边界会产生张力：
  - 高绑定位与低绑定位之间的边界
  - 活跃区域与密封区域的交界
  - 不同组织之间的界面
  
  张力积累到阈值时，触发新的翻转。

实现：
  tension(i) = |Σ(binding[i,j] * state[j]) for j in neighbors(i)|
  
  当 tension(i) > threshold 时，位 i 被翻转。
  这模拟了"聚簇边界不稳定"的物理直觉。
"""

import numpy as np


def compute_tension(field):
    """计算每个位的内部张力。
    
    张力 = 绑定加权的邻居状态差异。
    高张力 = 该位与邻居的绑定很强，但状态不同。
    """
    N = field.N
    tension = np.zeros(N)
    
    for i in range(N):
        if i in field.sealed_bits:
            continue
        # 绑定加权的邻居状态和
        weighted_sum = np.sum(field.binding[i, :] * field.state)
        # 张力 = |加权和 - 当前状态 × 总绑定|
        total_binding = np.sum(field.binding[i, :])
        if total_binding > 0:
            # 如果 state[i]=1，weighted_sum 应该高（与邻居一致）
            # 如果 state[i]=0，weighted_sum 应该低
            # 张力 = 不一致的程度
            expected = field.state[i] * total_binding
            tension[i] = abs(weighted_sum - expected)
    
    return tension


def m5_tension_driven(layer, tension_threshold=0.5, max_tension_flips=2):
    """张力驱动的最小变易。
    
    在标准 m5 的基础上，增加张力驱动的翻转：
    - 标准阶段：a1_source → budget → 翻转（主动阶段）
    - 张力阶段：聚簇边界 → 张力 → 张力超阈值 → 翻转（被动阶段）
    """
    f = layer.field
    if f.sealed:
        return
    rng = f.rng
    
    # === 标准阶段：a1_source 驱动 ===
    inject_cands = [b for (typ, b) in f.candidates if typ == 'inject']
    absorb_cands = [b for (typ, b) in f.candidates if typ == 'absorb']
    budget = layer._conservation_budget
    moves = 0
    
    n_inject = min(len(inject_cands), budget['max_inject'])
    if n_inject > 0:
        chosen = rng.choice(inject_cands, size=n_inject, replace=False).tolist()
        for b in chosen:
            f.state[b] = 1
            moves += 1
    
    n_absorb = min(len(absorb_cands), budget['max_absorb'])
    if n_absorb > 0 and absorb_cands:
        chosen = rng.choice(absorb_cands, size=n_absorb, replace=False).tolist()
        for b in chosen:
            f.state[b] = 0
            moves += 1
    
    # === 张力阶段：聚簇边界驱动 ===
    # 只在主动阶段无翻转时触发（被动阶段）
    if moves == 0:
        tension = compute_tension(f)
        
        # 找到高张力位
        high_tension_bits = []
        for i in range(f.N):
            if i in f.sealed_bits:
                continue
            if not f.admissible(i):
                continue
            if tension[i] > tension_threshold:
                high_tension_bits.append(i)
        
        if high_tension_bits:
            # 选择张力最高的位翻转
            high_tension_bits.sort(key=lambda b: tension[b], reverse=True)
            n_flip = min(len(high_tension_bits), max_tension_flips)
            
            for b in high_tension_bits[:n_flip]:
                f.state[b] = 1 - f.state[b]
                moves += 1
    
    layer.moves_this_step = moves
    layer._tension_driven = moves > 0 and budget['max_inject'] == 0
