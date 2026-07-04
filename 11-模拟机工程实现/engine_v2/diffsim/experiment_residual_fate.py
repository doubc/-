"""
experiment_residual_fate.py — 余差命运实验

核心修正：
  不是所有余差都需要被递归。
  真实的宇宙中，大部分差异是混沌的——它们什么都不做。
  九机制只在局部生效，形成结构岛。

实验设计：
  将余差分为三类：
    1. 递归余差：被投入下一轮九机制的差异
    2. 噪声余差：什么都不做，就待在那里的差异
    3. 消散余差：被环境吸收的差异
  
  改变递归比例（0%~100%），观察结构积累的变化。
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M
from diffsim.world_v2 import Layer, Params


def run_with_residual_fate(N, seed, steps, recursion_fraction=0.5):
    """
    运行引擎，余差按指定比例分为递归和噪声。
    
    recursion_fraction: 余差中被递归的比例
      0.0 = 所有余差都是噪声（什么都不做）
      0.5 = 一半递归，一半噪声
      1.0 = 所有余差都被递归（原始方案）
    """
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    
    layer = Layer(field, Params(max_steps=steps))
    
    total_recycled = 0
    total_noise = 0
    
    for step in range(steps):
        layer.step = step
        M.m1_clustering(layer); M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer); M.m6_breaking(layer)
        M.m7_cycle(layer); M.m8_locking(layer)
        
        # 每 100 步处理余差命运
        if step > 0 and step % 100 == 0:
            active_bits = set(np.where(field.state == 1)[0])
            sealed_bits = field.sealed_bits
            residuals = active_bits - sealed_bits
            
            if residuals:
                residual_list = list(residuals)
                rng_local = np.random.RandomState(seed + step)
                rng_local.shuffle(residual_list)
                
                n_recycle = int(len(residual_list) * recursion_fraction)
                recycled = set(residual_list[:n_recycle])
                noise = set(residual_list[n_recycle:])
                
                total_recycled += len(recycled)
                total_noise += len(noise)
                
                # 递归余差：重新注入为 a1_source
                if recycled:
                    field.a1_source = recycled
                
                # 噪声余差：什么都不做，就待在那里
    
    return {
        'binding': float(np.sum(field.binding) / 2),
        'lock': float(np.mean(field.lock_level)),
        'sealed': len(field.sealed_bits),
        'total_recycled': total_recycled,
        'total_noise': total_noise,
    }


N = 48; steps = 2000; trials = 3

print()
print("=" * 62)
print("  余差命运实验：递归 vs 噪声")
print("=" * 62)
print()
print("  核心修正：不是所有余差都需要被递归。")
print("  真实的宇宙中，大部分差异是混沌的——它们什么都不做。")
print("  九机制只在局部生效，形成结构岛。")
print()
print(f"  {'递归比例':>8s} | {'binding':>10s} | {'lock':>8s} | {'密封':>6s} | {'递归量':>8s} | {'噪声量':>8s}")
print("  " + "-" * 60)

for fraction in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    binds = []; locks = []; sealed = []
    recycled = []; noise = []
    
    for t in range(trials):
        seed = 42 + t * 1000
        result = run_with_residual_fate(N, seed, steps, fraction)
        binds.append(result['binding'])
        locks.append(result['lock'])
        sealed.append(result['sealed'])
        recycled.append(result['total_recycled'])
        noise.append(result['total_noise'])
    
    print(f"  {fraction:8.0%} | {np.mean(binds):10.1f} | {np.mean(locks):8.3f} | "
          f"{np.mean(sealed):6.1f} | {np.mean(recycled):8.1f} | {np.mean(noise):8.1f}")

print()
print("  分析：")
print("  - 递归比例 0%：所有余差都是噪声，无递归")
print("  - 递归比例 100%：所有余差都被递归（原始方案）")
print("  - 递归比例 10~50%：部分递归，部分噪声（最接近真实宇宙）")
print()
print("  预测：")
print("  - 递归比例 0% 时，binding 最低（无结构积累）")
print("  - 递归比例 100% 时，binding 最高（全部递归）")
print("  - 递归比例 10~50% 时，binding 居中（局部结构岛）")
print()
print("  理论意义：")
print("  真实的宇宙不需要被九机制完全组织。")
print("  九机制只在局部生效，形成结构岛。")
print("  大部分余差什么都不做——它们是混沌的背景。")
print("  结构岛在混沌海洋中自维持，这就是我们看到的世界。")
print()
print("  差异即世界，语法即局部。")
