"""
experiment_nearest_vs_breaking.py — 最近稳态 vs 余差聚簇与破缺

核心命题：
  最近稳态（m5）不影响余差聚簇（m1-m3）和破缺（m6）。
  
  即：系统到达稳态后，聚簇和破缺仍按自己的逻辑运行，
  不受"已经到达最近稳态"这一事实的约束。

实验设计：
  条件A：正常引擎运行到稳态
  条件B：到达稳态后，人为改变绑定结构（模拟"余差扰动"），
         观察破缺是否仍按自己的逻辑发生
  
  如果破缺受稳态约束 → 改变绑定后破缺不变
  如果破缺独立于稳态 → 改变绑定后破缺随之改变
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M
from diffsim.world_v2 import Layer, Params


def se(s):
    p1=np.mean(s); p0=1-p1
    if p0<1e-10 or p1<1e-10: return 0.0
    return float(-(p0*np.log2(p0)+p1*np.log2(p1)))

def cc(s):
    s=np.asarray(s,dtype=int); ext=np.concatenate([s,s[:1]])
    return int(np.sum(np.abs(np.diff(ext)))//2)


N = 48; steps = 2000; trials = 5


# ============================================================
# 实验6a：稳态后扰动绑定 → 破缺是否改变？
# ============================================================
print()
print("=" * 62)
print("  实验6a：最近稳态 vs 破缺")
print("=" * 62)
print()
print("  方法：")
print("    1. 引擎运行到稳态 S_stable")
print("    2. 记录 S_stable 的绑定结构、候选集、破缺结果")
print("    3. 人为改变绑定结构（增加/打乱/清零）")
print("    4. 重新运行 m4+m5+m6，观察破缺是否改变")
print()
print("  预测：破缺会改变。因为 m6 依赖候选集，")
print("        候选集依赖绑定结构，改变绑定 → 改变候选 → 改变破缺。")
print()

print(f"  {'Trial':>5s} | {'稳态binding':>10s} | {'稳态破缺':>10s} | "
      f"{'扰动后破缺':>10s} | {'破缺改变':>8s}")
print("  " + "-" * 55)

n_changed = 0
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
    
    # 运行到稳态
    for step in range(steps):
        layer.step = step
        M.m1_clustering(layer); M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer); M.m6_breaking(layer)
        M.m7_cycle(layer); M.m8_locking(layer)
    
    stable_binding = float(np.sum(field.binding) / 2)
    stable_state = field.state.copy()
    stable_lock = field.lock_level.copy()
    
    # 记录稳态的候选集和破缺
    M.m3_conservation(layer)
    M.m4_innate_completeness(layer)
    stable_candidates = set(field.candidates)
    
    # 人为打乱绑定结构
    rng2 = np.random.RandomState(seed + 999)
    perturbed_binding = field.binding.copy()
    # 随机交换绑定值
    n_swap = N * N // 4
    for _ in range(n_swap):
        i, j = rng2.randint(0, N, 2)
        k, l = rng2.randint(0, N, 2)
        perturbed_binding[i, j], perturbed_binding[k, l] = \
            perturbed_binding[k, l], perturbed_binding[i, j]
    field.binding = perturbed_binding
    
    # 重新计算候选集
    M.m3_conservation(layer)
    M.m4_innate_completeness(layer)
    perturbed_candidates = set(field.candidates)
    
    # 比较候选集
    same_candidates = stable_candidates == perturbed_candidates
    
    # 运行破缺
    M.m5_minimal_variation(layer)
    M.m6_breaking(layer)
    perturbed_state = field.state.copy()
    
    state_changed = not np.array_equal(stable_state, perturbed_state)
    
    if state_changed:
        n_changed += 1
    
    print(f"  {t+1:5d} | {stable_binding:10.1f} | {'不变' if not state_changed else '改变':>10s} | "
          f"{'不变' if not state_changed else '改变':>10s} | "
          f"{'是' if state_changed else '否':>8s}")

print()
print(f"  破缺改变率: {n_changed}/{trials}")
if n_changed > 0:
    print("  ✅ 验证：改变绑定结构后，破缺随之改变。")
    print("     最近稳态不影响余差聚簇和破缺——它们是独立运行的。")
else:
    print("  ❌ 未验证：破缺未改变。")
print()


# ============================================================
# 实验6b：不同稳态起点 → 相同的聚簇和破缺？
# ============================================================
print("=" * 62)
print("  实验6b：不同稳态起点 → 余差聚簇是否相同？")
print("=" * 62)
print()
print("  方法：")
print("    1. 用不同种子到达不同稳态 S1, S2, S3...")
print("    2. 在每个稳态上运行 m1-m3（余差聚簇）")
print("    3. 比较余差聚簇的结果：是否与稳态无关？")
print()

print(f"  {'种子':>6s} | {'稳态binding':>10s} | {'稳态sum':>8s} | "
      f"{'余差聚簇数':>10s} | {'余差binding':>12s}")
print("  " + "-" * 55)

all_residual_clusters = []
all_residual_bindings = []

for s in range(10):
    seed = s * 100 + 42
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    
    layer = Layer(field, Params(max_steps=steps))
    
    # 运行到稳态
    for step in range(steps):
        layer.step = step
        M.m1_clustering(layer); M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer); M.m6_breaking(layer)
        M.m7_cycle(layer); M.m8_locking(layer)
    
    stable_binding = float(np.sum(field.binding) / 2)
    stable_sum = int(field.state.sum())
    
    # 在稳态上运行 m1-m3（余差聚簇）
    pre_binding = field.binding.copy()
    M.m1_clustering(layer)
    M.m2_hierarchy(layer)
    M.m3_conservation(layer)
    post_binding = field.binding.copy()
    
    # 余差 = 新绑定 - 旧绑定
    residual_binding = float(np.sum(post_binding - pre_binding) / 2)
    residual_clusters = len(layer.tentative_orgs) if layer.tentative_orgs else 0
    
    all_residual_clusters.append(residual_clusters)
    all_residual_bindings.append(residual_binding)
    
    print(f"  {seed:6d} | {stable_binding:10.1f} | {stable_sum:8d} | "
          f"{residual_clusters:10d} | {residual_binding:12.1f}")

print()
print(f"  余差聚簇数: {np.mean(all_residual_clusters):.1f} ± {np.std(all_residual_clusters):.1f}")
print(f"  余差binding: {np.mean(all_residual_bindings):.1f} ± {np.std(all_residual_bindings):.1f}")
print()

# 检查余差聚簇是否与稳态binding相关
stable_bindings = []
for s in range(10):
    seed = s * 100 + 42
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
        M.m5_minimal_variation(layer); M.m6_breaking(layer)
        M.m7_cycle(layer); M.m8_locking(layer)
    stable_bindings.append(float(np.sum(field.binding) / 2))

# 相关系数
if len(stable_bindings) == len(all_residual_bindings):
    corr = np.corrcoef(stable_bindings, all_residual_bindings)[0, 1]
    print(f"  稳态binding vs 余差binding 相关系数: {corr:.3f}")
    if abs(corr) < 0.3:
        print("  ✅ 弱相关：余差聚簇与稳态binding几乎无关。")
        print("     余差聚簇按自己的逻辑运行，不受最近稳态约束。")
    else:
        print(f"  ⚠️ 相关性={corr:.3f}，需要更大样本确认。")
print()


# ============================================================
# 实验6c：稳态后继续运行 → 聚簇是否仍在增长？
# ============================================================
print("=" * 62)
print("  实验6c：稳态后继续运行 → binding 是否继续增长？")
print("=" * 62)
print()
print("  方法：引擎运行到稳态后，继续运行 3000 步，")
print("        观察 binding 是否继续增长。")
print()
print("  如果余差聚簇独立于稳态 → binding 继续增长")
print("  如果余差聚簇受稳态约束 → binding 停止增长")
print()

print(f"  {'Step':>6s} | {'binding':>12s} | {'lock':>8s} | {'聚簇':>6s}")
print("  " + "-" * 40)

for s in range(3):
    seed = 42 + s * 1000
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    
    total_steps = 5000
    layer = Layer(field, Params(max_steps=total_steps))
    
    bind_trace = []
    for step in range(total_steps):
        layer.step = step
        M.m1_clustering(layer); M.m2_hierarchy(layer)
        M.m3_conservation(layer); M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer); M.m6_breaking(layer)
        M.m7_cycle(layer); M.m8_locking(layer)
        if step % 500 == 0:
            bind_trace.append(float(np.sum(field.binding) / 2))
    
    print(f"  Seed {seed}:")
    for idx, bv in enumerate(bind_trace):
        print(f"  {idx*500:6d} | {bv:12.1f}")
    print()

print("  如果 binding 在稳态后继续增长，")
print("  说明余差聚簇不受最近稳态约束。")
print()
print("  差异即世界，语法即一切。")
