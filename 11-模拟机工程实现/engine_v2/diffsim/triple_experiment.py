"""
triple_experiment.py — 三个核心思想实验

实验1：时间之箭 — 结构积累的不可逆性
实验2：小步革命 — 最小变易是变化的总法则
实验3：热寂悖论 — 复杂性为何能在熵增中增长
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffsim.core import DifferenceField
from diffsim import mechanisms as M


# ============================================================
# 共用工具
# ============================================================

def shannon_entropy(state):
    p1 = np.mean(state)
    p0 = 1 - p1
    if p0 < 1e-10 or p1 < 1e-10:
        return 0.0
    return float(-(p0 * np.log2(p0) + p1 * np.log2(p1)))

def count_clusters(state):
    s = np.asarray(state, dtype=int)
    ext = np.concatenate([s, s[:1]])
    return int(np.sum(np.abs(np.diff(ext))) // 2)

def structural_complexity(state):
    nc = count_clusters(state)
    if nc == 0:
        return 0.0
    s = np.asarray(state, dtype=int)
    sizes = []
    in_c = False; sz = 0
    for i in range(len(s) * 2):
        idx = i % len(s)
        if s[idx] == 1:
            if not in_c: in_c = True; sz = 1
            else: sz += 1
        else:
            if in_c: sizes.append(sz); in_c = False; sz = 0
        if i >= len(s) and not in_c: break
    if in_c: sizes.append(sz)
    if not sizes: return 0.0
    cv = np.std(sizes) / max(np.mean(sizes), 1e-10) if len(sizes) > 1 else 0.0
    return float(nc * (1 + cv))


def make_engine(N, seed, max_steps):
    """创建差异引擎并运行到稳态。"""
    rng = np.random.RandomState(seed)
    active = np.where(rng.random(N) < 0.5)[0].tolist()
    field = DifferenceField(N=N, active=active, rng=rng)
    for i in range(N):
        for j in range(i+1, N):
            if field.color[i] == field.color[j]:
                field.binding[i, j] = field.binding[j, i] = 0.1
    from diffsim.world_v2 import Layer, Params
    layer = Layer(field, Params(max_steps=max_steps))
    
    snapshots = []
    for step in range(max_steps):
        layer.step = step
        M.m1_clustering(layer)
        M.m2_hierarchy(layer)
        M.m3_conservation(layer)
        M.m4_innate_completeness(layer)
        M.m5_minimal_variation(layer)
        M.m6_breaking(layer)
        M.m7_cycle(layer)
        M.m8_locking(layer)
        if step % 200 == 0:
            snapshots.append({
                'step': step,
                'state': field.state.copy(),
                'binding': float(np.sum(field.binding) / 2),
                'lock': float(np.mean(field.lock_level)),
                'entropy': shannon_entropy(field.state),
                'complexity': structural_complexity(field.state),
                'clusters': count_clusters(field.state),
                'orgs': len(layer.tentative_orgs) if layer.tentative_orgs else 0,
            })
    return field, layer, snapshots


# ============================================================
# 实验1：时间之箭
# ============================================================

def experiment_arrow_of_time(N=48, max_steps=10000, n_trials=5):
    """
    验证：结构积累是否不可逆。
    
    方法：运行引擎到稳态，然后强制反转所有翻转，
    观察系统能否回到初始状态。
    
    预测：不能。锁定的结构会抵抗反转。
    """
    print("=" * 70)
    print("  实验1：时间之箭")
    print("=" * 70)
    print(f"  N={N}, steps={max_steps}, trials={n_trials}")
    print()
    
    results = []
    
    for trial in range(n_trials):
        seed = 42 + trial * 1000
        rng = np.random.RandomState(seed)
        
        # 正向运行
        active = np.where(rng.random(N) < 0.5)[0].tolist()
        field = DifferenceField(N=N, active=active, rng=rng)
        initial_state = field.state.copy()
        for i in range(N):
            for j in range(i+1, N):
                if field.color[i] == field.color[j]:
                    field.binding[i, j] = field.binding[j, i] = 0.1
        
        from diffsim.world_v2 import Layer, Params
        layer = Layer(field, Params(max_steps=max_steps))
        
        # 记录每步翻转
        flip_history = []
        for step in range(max_steps):
            layer.step = step
            prev_state = field.state.copy()
            M.m1_clustering(layer)
            M.m2_hierarchy(layer)
            M.m3_conservation(layer)
            M.m4_innate_completeness(layer)
            M.m5_minimal_variation(layer)
            M.m6_breaking(layer)
            M.m7_cycle(layer)
            M.m8_locking(layer)
            # 记录哪些位翻转了
            flips = np.where(field.state != prev_state)[0]
            flip_history.append(flips)
        
        forward_binding = float(np.sum(field.binding) / 2)
        forward_lock = float(np.mean(field.lock_level))
        forward_state = field.state.copy()
        
        # 反向运行：强制反转所有翻转
        reverse_binding_trace = [forward_binding]
        reverse_lock_trace = [forward_lock]
        
        for step in range(max_steps - 1, -1, -1):
            flips = flip_history[step]
            for bit in flips:
                field.state[bit] = 1 - field.state[bit]
            # 重新计算绑定和锁定（简化：只追踪状态）
            reverse_binding_trace.append(float(np.sum(field.binding) / 2))
            reverse_lock_trace.append(float(np.mean(field.lock_level)))
        
        final_state = field.state
        state_recovery = float(np.mean(final_state == initial_state))
        binding_recovery = reverse_binding_trace[-1]
        lock_recovery = reverse_lock_trace[-1]
        
        results.append({
            'forward_binding': forward_binding,
            'forward_lock': forward_lock,
            'state_recovery': state_recovery,
            'binding_at_reverse_end': binding_recovery,
            'lock_at_reverse_end': lock_recovery,
        })
        
        print(f"  Trial {trial+1}: forward_binding={forward_binding:.1f}, "
              f"forward_lock={forward_lock:.3f}, "
              f"state_recovery={state_recovery:.3f}")
    
    print()
    print("  汇总:")
    avg_recovery = np.mean([r['state_recovery'] for r in results])
    avg_fwd_bind = np.mean([r['forward_binding'] for r in results])
    avg_fwd_lock = np.mean([r['forward_lock'] for r in results])
    print(f"    正向 binding: {avg_fwd_bind:.1f}")
    print(f"    正向 lock: {avg_fwd_lock:.3f}")
    print(f"    反转后状态恢复率: {avg_recovery:.3f}")
    print()
    
    if avg_recovery < 0.95:
        print("  ✅ 验证：时间之箭不可逆。反转无法恢复初始状态。")
    else:
        print("  ❌ 未验证：反转后状态恢复率过高。")
    print()
    
    return results


# ============================================================
# 实验2：小步革命
# ============================================================

def experiment_small_steps(N=48, max_steps=10000, n_trials=5):
    """
    验证：最小变易是变化的总法则。
    
    方法：用不同步长运行引擎，比较结构积累。
    
    预测：小步 > 中步 > 大步 > 巨步。
    """
    print("=" * 70)
    print("  实验2：小步革命")
    print("=" * 70)
    print(f"  N={N}, steps={max_steps}, trials={n_trials}")
    print()
    
    step_sizes = [1, 3, 6, 12, 24]  # 每步翻转的比特数
    all_results = {}
    
    for step_size in step_sizes:
        bindings = []
        locks = []
        complexities = []
        orgs_list = []
        
        for trial in range(n_trials):
            seed = 42 + trial * 1000
            rng = np.random.RandomState(seed)
            active = np.where(rng.random(N) < 0.5)[0].tolist()
            field = DifferenceField(N=N, active=active, rng=rng)
            for i in range(N):
                for j in range(i+1, N):
                    if field.color[i] == field.color[j]:
                        field.binding[i, j] = field.binding[j, i] = 0.1
            
            from diffsim.world_v2 import Layer, Params
            params = Params(max_steps=max_steps)
            layer = Layer(field, params)
            
            for step in range(max_steps):
                layer.step = step
                # 标准九机制
                M.m1_clustering(layer)
                M.m2_hierarchy(layer)
                M.m3_conservation(layer)
                M.m4_innate_completeness(layer)
                M.m5_minimal_variation(layer)
                M.m6_breaking(layer)
                M.m7_cycle(layer)
                M.m8_locking(layer)
                
                # 如果步长大于1，额外翻转 (step_size - 1) 个随机位
                if step_size > 1:
                    active_bits = np.where(field.state == 1)[0]
                    inactive_bits = np.where(field.state == 0)[0]
                    for _ in range(step_size - 1):
                        # 随机选择翻转方向（保持大致平衡）
                        if rng.random() < 0.5 and len(active_bits) > 0:
                            bit = rng.choice(active_bits)
                        elif len(inactive_bits) > 0:
                            bit = rng.choice(inactive_bits)
                        else:
                            bit = rng.randint(0, N)
                        field.state[bit] = 1 - field.state[bit]
            
            bindings.append(float(np.sum(field.binding) / 2))
            locks.append(float(np.mean(field.lock_level)))
            complexities.append(structural_complexity(field.state))
            orgs_list.append(len(layer.tentative_orgs) if layer.tentative_orgs else 0)
        
        all_results[step_size] = {
            'binding': np.mean(bindings),
            'lock': np.mean(locks),
            'complexity': np.mean(complexities),
            'orgs': np.mean(orgs_list),
        }
    
    print(f"  {'步长':>6s} | {'binding':>10s} | {'lock':>8s} | {'complexity':>10s} | {'orgs':>6s}")
    print("  " + "-" * 55)
    for ss in step_sizes:
        r = all_results[ss]
        print(f"  {ss:6d} | {r['binding']:10.1f} | {r['lock']:8.3f} | "
              f"{r['complexity']:10.2f} | {r['orgs']:6.1f}")
    
    print()
    
    # 验证：小步是否显著优于大步
    small_bind = all_results[1]['binding']
    large_bind = all_results[12]['binding']
    ratio = small_bind / max(large_bind, 0.001)
    
    if ratio > 2.0:
        print(f"  ✅ 验证：小步 binding 是大步的 {ratio:.1f} 倍。最小变易是变化的总法则。")
    else:
        print(f"  ⚠️ 部分验证：小步/大步 binding 比值 = {ratio:.1f}。")
    print()
    
    return all_results


# ============================================================
# 实验3：热寂悖论
# ============================================================

def experiment_heat_death(N=48, max_steps=20000, n_trials=5):
    """
    验证：复杂性为何能在熵增中增长。
    
    方法：同时追踪 Shannon 熵和结构复杂度。
    
    预测：熵保持大致不变，复杂度持续增长。
    """
    print("=" * 70)
    print("  实验3：热寂悖论")
    print("=" * 70)
    print(f"  N={N}, steps={max_steps}, trials={n_trials}")
    print()
    
    all_entropy_curves = []
    all_complexity_curves = []
    all_binding_curves = []
    
    for trial in range(n_trials):
        seed = 42 + trial * 1000
        rng = np.random.RandomState(seed)
        active = np.where(rng.random(N) < 0.5)[0].tolist()
        field = DifferenceField(N=N, active=active, rng=rng)
        for i in range(N):
            for j in range(i+1, N):
                if field.color[i] == field.color[j]:
                    field.binding[i, j] = field.binding[j, i] = 0.1
        
        from diffsim.world_v2 import Layer, Params
        layer = Layer(field, Params(max_steps=max_steps))
        
        entropy_curve = []
        complexity_curve = []
        binding_curve = []
        
        for step in range(max_steps):
            layer.step = step
            M.m1_clustering(layer)
            M.m2_hierarchy(layer)
            M.m3_conservation(layer)
            M.m4_innate_completeness(layer)
            M.m5_minimal_variation(layer)
            M.m6_breaking(layer)
            M.m7_cycle(layer)
            M.m8_locking(layer)
            
            if step % 500 == 0:
                entropy_curve.append(shannon_entropy(field.state))
                complexity_curve.append(structural_complexity(field.state))
                binding_curve.append(float(np.sum(field.binding) / 2))
        
        all_entropy_curves.append(entropy_curve)
        all_complexity_curves.append(complexity_curve)
        all_binding_curves.append(binding_curve)
    
    # 打印时间序列
    checkpoints = list(range(0, len(all_entropy_curves[0])))
    step_values = [i * 500 for i in checkpoints]
    
    print(f"  {'Step':>8s} | {'熵(mean±std)':>18s} | {'复杂度(mean±std)':>18s} | {'binding':>10s}")
    print("  " + "-" * 65)
    
    for idx in checkpoints[::4]:  # 每隔4个采样点打印
        ent_vals = [c[idx] for c in all_entropy_curves]
        cplx_vals = [c[idx] for c in all_complexity_curves]
        bind_vals = [c[idx] for c in all_binding_curves]
        print(f"  {step_values[idx]:8d} | "
              f"{np.mean(ent_vals):7.3f} ± {np.std(ent_vals):.3f}  | "
              f"{np.mean(cplx_vals):7.2f} ± {np.std(cplx_vals):.2f}  | "
              f"{np.mean(bind_vals):10.1f}")
    
    print()
    
    # 计算趋势
    ent_start = np.mean([c[0] for c in all_entropy_curves])
    ent_end = np.mean([c[-1] for c in all_entropy_curves])
    cplx_start = np.mean([c[0] for c in all_complexity_curves])
    cplx_end = np.mean([c[-1] for c in all_complexity_curves])
    bind_start = np.mean([c[0] for c in all_binding_curves])
    bind_end = np.mean([c[-1] for c in all_binding_curves])
    
    print(f"  熵:   {ent_start:.3f} → {ent_end:.3f} (变化 {ent_end - ent_start:+.3f})")
    print(f"  复杂度: {cplx_start:.2f} → {cplx_end:.2f} (变化 {cplx_end - cplx_start:+.2f})")
    print(f"  binding: {bind_start:.1f} → {bind_end:.1f} (变化 {bind_end - bind_start:+.1f})")
    print()
    
    ent_change = abs(ent_end - ent_start)
    cplx_change = cplx_end - cplx_start
    
    if ent_change < 0.05 and cplx_change > 1.0:
        print("  ✅ 验证：熵基本不变，复杂度持续增长。复杂性不需要对抗熵增。")
    elif cplx_change > 0:
        print("  ⚠️ 部分验证：复杂度有增长，但熵也有变化。")
    else:
        print("  ❌ 未验证：复杂度未见增长。")
    print()
    
    return {
        'entropy_start': ent_start, 'entropy_end': ent_end,
        'complexity_start': cplx_start, 'complexity_end': cplx_end,
        'binding_start': bind_start, 'binding_end': bind_end,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  差异论核心思想实验：三个命题的模拟机验证                        ║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("  与莎士比亚实验同级别的三个可验证命题：")
    print("  1. 时间之箭 — 结构积累的不可逆性")
    print("  2. 小步革命 — 最小变易是变化的总法则")
    print("  3. 热寂悖论 — 复杂性为何能在熵增中增长")
    print()
    
    t0 = time.time()
    
    r1 = experiment_arrow_of_time(N=48, max_steps=10000, n_trials=5)
    r2 = experiment_small_steps(N=48, max_steps=10000, n_trials=5)
    r3 = experiment_heat_death(N=48, max_steps=20000, n_trials=5)
    
    elapsed = time.time() - t0
    
    print("=" * 70)
    print(f"  全部完成。总耗时: {elapsed:.1f}s")
    print("=" * 70)
    print()
    print("  差异即世界，语法即一切。")


if __name__ == '__main__':
    main()
