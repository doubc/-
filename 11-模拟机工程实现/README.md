# 差异论模拟机工程实现

> 在九公理约束下构造有限离散世界，观察稳定结构能否递归生成高层单元。

## 一句话

**给差异论公理装上轮子，让它自己跑起来。**

这不是数值模拟，不是机器学习训练，不是可视化演示。这是一个**公理约束下的离散世界实验机**——从 `{0,1}^N` 上的比特翻转出发，让九条公理自己决定哪些结构能存活、哪些能涌现、哪些能封装成更高层级的单元。

---

## 为什么需要模拟机

差异论的理论推导（WorldBase 形式化框架）已经证明：从 10 条公理可以推出 Einstein 场方程、规范结构、量子力学核心框架。但这条推导链有一个瓶颈——**从离散到连续的数学极限**（定理 CL）在纯解析层面暂时走不通。

模拟机的思路很简单：既然纸上推不动，就让语法自己跑。

- 02-worldbase 形式化框架问的是"公理能推出什么"——纯数学推导
- 11-模拟机工程问的是"公理约束下什么会自发涌现"——计算验证

如果模拟机中涌现出了维度锁定、引力势、规范结构、层级封装——那不是"9 步归纳碰巧对应公理"，而是同一套语法在计算中验证了自己。

---

## 理论底座：四层架构

```
WorldBase（差异学会存在）
    ↓ 10条公理约束 {0,1}^N
差异即世界（差异学会生成）
    ↓ 9机制生成链：聚簇→层级→守恒→完备→变易→破缺→循环→锁定→自指
象界（差异学会显现）
    ↓ 8章生成链：边界→自维持→记忆→复制→筛选→功能→前主体态
差异论（差异进入历史）
    ↓ 制度·叙事·身份·锁定
```

模拟机当前覆盖范围：**WorldBase + 差异即世界**（第一阶段），象界机制（Phase 2-16）。
**当前状态**: Phase 17 — engine_v2 自指闭环深度验证。

> **重要更正 (2026-06-09)**: Phase 16 的「死秩序不可打破」归档结论已被推翻。
> 根因是 A9 自指动作缺失。engine_v2 补回后，L2 涌现率 0%→95%。
> 详见 `engine_v2/README.md` 和 `docs/纠偏_A9自指缺失与过早归档.md`。

---

## 九公理

| 公理 | 名称 | 类别 | 模拟机中的角色 |
|------|------|------|--------------|
| A1 | 差异源 | 观测 | 持续 +1 外部注入（比特 0→1） |
| A1' | 横向涌现 | — | 绑定强度加权配对 → 聚类 |
| A2 | 离散编码 | 约束 | 状态空间 {0,1}^N |
| A3 | 局域性 | 约束 | 汉明距离=1 的邻域 |
| A4 | 最小变易 | 约束 | 每次只翻 1 个比特 |
| A5 | 守恒 | 约束 | 注入/吸收量平衡 |
| A6 | 流向耦合 | 观测 | DAG 不可逆（禁止逆向翻转） |
| A7 | 稳定闭合 | 约束 | 循环状态检测 |
| A8 | 差异汇 | 观测 | 持续 -1 外部吸收（比特 1→0） |
| A9 | 升维触发 | 触发 | 自由度封口 → 层级封装 |

核心演化流程：

```
A1(源,+1) → A6(流向) → A3(局域) → A4(最小变易)
    → A7(稳定结构形成)
    → A5(守恒残差检测)
    → A9(升维触发)
    → 粗粒化封装
    → 新层继续运行
```

---

## 工程架构

### 目录结构

```
11-模拟机工程实现/
├── engine_v2/              ← ★ 新主引擎（自指闭环版，2026-06-09 起）
│   ├── diffsim/
│   │   ├── core.py         ← DifferenceField：差异场
│   │   ├── mechanisms.py   ← 九个机制作为显式齿轮 (m1-m9)
│   │   ├── world.py        ← World：能量+熵+开放系统集成的独立世界
│   │   ├── world_v2.py     ← ★ Layer + RecursiveWorld：九机制驱动 + 递归封装
│   │   ├── worldbase_core.py ← 中截面、变易算符、su(3)/su(2) 代数
│   │   ├── worldbase_formulas.py ← WorldBase 实验公式推导
│   │   ├── energy.py / energy_v2.py ← 能量预算管理
│   │   ├── entropy.py      ← Shannon 熵 + 自由能追踪
│   │   ├── environment_energy.py ← 开放系统能量注入 + 熵排出
│   │   ├── metrics.py      ← Jaccard flux 活/死秩序指标
│   │   ├── fixed_point.py  ← 整体固定点检测器（自指同构判据）
│   │   ├── axiom_checker.py← NumPy 版公理约束检查器
│   │   ├── xiang_mechanisms.py ← 象界显现链 (m10-m16)
│   │   └── tests/          ← 单元测试
│   ├── experiments/        ← engine_v2 专用实验
│   ├── archive/            ← 历史探索归档
│   └── README.md
├── engine/                 ← Phase 1-16 历史引擎（保留参考）
├── acl/                    ← 公理约束语言（Axiomatic Constraint Language）
│   ├── axioms_v2.py        ← ★ 九公理硬性约束检查器（当前主版本）
│   ├── axioms_strict.py    ← 严格化九公理
│   └── axioms_v3.py        ← 实验性（引入排除历史）
├── layers/                 ← 层级世界规格
│   ├── hamming_layer.py    ← ★ {0,1}^N 二值状态空间层
│   ├── three_dim_hamming.py← 三维汉明格点
│   ├── L0_binary_lattice.py← L0 二元格点
│   ├── L1_abstract_layer.py← L1 抽象层
│   ├── coarse_grain.py     ← 粗粒化映射
│   └── layer_base.py       ← 层级基类
├── xiangjie/               ← 象界显现链
│   └── chain.py            ← ★ 八章门槛检测器（864行）
├── validators/             ← 验证器
│   └── structure_validator.py ← 五标准结构验证
├── models/                 ← 学习模型
│   └── local_conv_model.py ← 3×3 卷积局部模型
├── experiments/            ← 实验
│   ├── exp_0_baseline.py   ← 基线实验
│   ├── exp_1_baseline.py   ← 基线实验 v2
│   ├── exp_2_source_sink.py ← 源/汇平衡
│   ├── exp_3_minimal_variation.py ← 最小变易
│   ├── exp_4_conservation.py ← 守恒验证
│   ├── exp_5_stability.py ← 稳定性
│   ├── exp_6_ascent_trigger.py ← 升维触发
│   ├── exp_7_integrated_pressure.py ← 综合压力
│   ├── exp_8_xiangjie_chain.py ← 象界显现链
│   ├── exp_9_base_map_event.py ← 底图事件
│   ├── exp_10_physics_validation.py ← 物理验证
│   ├── exp_11_gravitational_potential.py ← 引力势验证
│   ├── exp_12_three_dim_physics.py ← 三维物理
│   ├── exp_13_emergence.py ← 涌现
│   ├── exp_14_emergence_v2.py ← 涌现 v2
│   ├── exp_16_binding_clustering.py ← 绑定聚类
│   ├── exp_21_t20k_sealed.py ← A9 封口触发
│   ├── exp_31_spatial_detection.py ← 空间探测
│   ├── exp_32_full_spatial_detection.py ← 全量空间探测
│   └── exp_40_hierarchical.py ← ★ 层级涌现实验
├── tests/                  ← 测试（330+ 测试）
│   ├── test_encapsulation.py ← 封装引擎测试（29项）
│   ├── test_hierarchy.py     ← 层级管理器测试（14项）
│   ├── test_bias_field.py    ← 偏置场测试
│   ├── test_bias_propagation.py ← 偏置传播测试
│   ├── test_xiangjie.py      ← 象界显现链测试（19项）
│   └── ...（共 25+ 测试文件）
├── docs/                   ← 设计文档
│   ├── code-inventory.md   ← 代码功能认定
│   ├── global-architecture.md ← 全局架构
│   ├── sealed-whole-definition.md ← 封口后整体定义
│   └── M4-batch11-encapsulation.md ← 批次11记录
├── legacy/                 ← 旧版代码
└── run_experiment.py       ← 实验运行入口
```

### 工程坐标系

| 层 | 说明 | 状态 |
|----|------|------|
| T | Theory（理论来源） | — |
| ACL | Axiomatic Constraint Language（公理约束语言） | ✅ 核心完成 |
| L | Layer（层级世界规格） | ✅ 核心完成 |
| E | Engine（世界演化引擎） | ✅ 核心完成 |
| M | Model（学习模型） | 🔄 基础完成 |
| V | Validator（验证器） | ✅ 核心完成 |
| R | Recursion（稳定结构封装与递归） | ✅ 批次11完成 |
| P | Physics Modules（物理模块） | 🔄 探测器框架 |
| D | Documentation（文档与理论回写） | 🔄 进行中 |

---

## 里程碑

| 阶段 | 日期 | 状态 | 核心成果 |
|------|------|------|---------|
| M0 | 2026-04 | ✅ | 骨架代码、公理接口、层级基础 |
| M1 | 2026-04-29 | ✅ | 差异反应堆 + 公理约束训练 |
| M1.1 | 2026-04-29 | ✅ | 语义修复 + 测试 + 流量平衡 |
| M2 | 2026-05-06 | ✅ | 稳定结构验证器 + 粗粒化映射 |
| M3 | 2026-05-14 | ✅ | 象界显现链（xiangjie/chain.py，八章门槛检测器，19 测试） |
| M4 | 2026-05-17 | ✅ | 底图事件 + 差异分层 + 汉明几何 + 九公理严格化 + 引力势验证 |
| M4 批次5 | 2026-05-14 | ✅ | 三维汉明 + 中截面分析 + 一阶变易代数 |
| M4 批次6 | 2026-05-15 | ✅ | 涌现探测器框架（5 个统计量探测器） |
| M4 批次7 | 2026-05-15 | ✅ | 公理重设计（axioms_v2.py 硬性约束版） |
| M4 批次8 | 2026-05-15 | ✅ | A1' 绑定聚类机制 |
| M4 批次9 | 2026-05-16 | ✅ | A5/A6/A7/A9 公理行为分析 |
| M4 批次10 | 2026-05-16 | ✅ | A5/A7/A9 修复（源汇平衡 + 近似循环检测 + 自由度封口） |
| M4 批次10b | 2026-05-17 | ✅ | A9 封口触发（N=48, T=20000，75% 比特冻结） |
| M4 批次11 | 2026-05-19 | ✅ | 分层封装（封装引擎 + 层级管理器 + 跨层级演化器，43 测试） |
| M4 批次12 | 2026-05-24 | ✅ | 回流偏置场修复（propagate_bias_up 三 bug + 键统一） |
| Phase 1-16 | 2026-05-24 ~ 2026-06-09 | ✅ | 象界机制全量实现，170+实验，发现N0*≈30.5相变 |
| Phase 16 归档 | 2026-06-09 | ❌ 撤回 | 过早归档，A9自指缺失导致"死秩序"误判 |
| Phase 17 | 2026-06-09 ~ | 🔄 | engine_v2 自指闭环，L2涌现率95%，深度验证中 |
| 代码审计 | 2026-06-27 | ✅ | 全量审计 + 修复：fixed_point导入、world.py重构、能量阈值集成、冗余归档 |

**当前状态**：Phase 17 — engine_v2 自指闭环深度验证，代码审计通过。

---

## 代码审计 (2026-06-27)

对全量代码（374 Python 文件，143,834 行）进行完整审计，覆盖架构、理论一致性、安全性、可维护性。

### 审计结论

| 维度 | 评价 |
|------|------|
| 理论-代码一致性 | ✅ 九公理→九机制映射完整，七个物理预测全部通过 |
| 自指闭环正确性 | ✅ v2 引擎 m9 命名位机制正确，L2 涌现率 95% |
| 安全性 | ✅ 无硬编码密钥、无危险系统调用、无网络外发 |
| 代码质量 | ⚠️ 双引擎并存增加认知负担，v1 应逐步归档 |

### 已修复问题

1. **`fixed_point.py` 导入错误**：`Layer` 从 `world.py` 改为 `world_v2.py`
2. **`world.py` 不可用**：`step()` 调用未定义函数 → 改用已有私有方法；修复 `EnergyManager` / `EntropyTracker` 属性名不匹配
3. **能量阈值 TODO**：`adjusted_threshold` 集成到 `lock_threshold`，能量系统实际影响密封决策
4. **冗余文件清理**：`world_v2_fixed.py`、`world_v2_original_backup.py` 归档至 `archive/world_v2_variants/`

### 待改进

- engine/（v1）的 40+ 模块应逐步归档到 `legacy/`
- 缺少 `requirements.txt` 或 `pyproject.toml` 统一管理依赖
- `axiom_checker.py` 的 `get_A8_bias()` 依赖 scipy 但未声明

---

## 已涌现的结构

| 结构 | 机制 | 验证 |
|------|------|------|
| **聚类**（A1'） | 绑定强度加权配对 → 稳定聚类 | ✅ |
| **层级**（九机制） | 聚类内部核心-外围结构 | ✅ |
| **循环**（A7） | 925+ 循环状态 | ✅ |
| **自由度封口**（A9） | 75% 比特冻结，25% 保持活跃 | ✅ |
| **引力势**（WorldBase §3） | Φ∝-1/d_H 解析验证零误差 | ✅ |
| **维度锁定**（WorldBase §2） | D_eff=3 | 🔄 探测器已实现 |
| **规范场**（WorldBase §5） | su(3) 代数结构 | 🔄 探测器已实现 |
| **层级封装**（批次11） | 48→15→3→3 比特涌现 | ✅ |
| **回流偏置场**（批次12） | 双向跨层级耦合 | ✅ |
| **自指闭环**（engine_v2） | A9封装自身，L1 flux=0.2123，L2涌现率95% | ✅ |

---

## 关键实验结果

### 莎士比亚实验：结构积累的不可逆性 (2026-07-04)

**核心问题**：为什么地球在46亿年内打印出了莎士比亚全集，而不是需要无限时间的猴子打印机？

**实验设计**：N=48 离散差异空间，20000步，5次试验。对比猴子打印机（纯随机）vs 差异引擎（九机制）。

| 指标 | 猴子打印机 | 差异引擎 |
|------|-----------|----------|
| 绑定能量 | **0.0** | **1272.6** |
| 路径锁定 | **0.000** | **0.392** |
| 组织数 | **0** | **1** |

**结论**：猴子打印机的时间是可逆的（每步重新开始），差异引擎的时间是不可逆的（每步都在积累）。差异即时间，语法即方向，结构即记忆。

详见 `docs/shakespeare_experiment.md` 和 `engine_v2/diffsim/shakespeare_final.py`。

### exp_8：象界显现链
- 16×16 网格，稳定结构到达象界阶段 VII（功能门槛）
- 前主体态未达（符合预期——第一阶段不要求前主体态）

### exp_11：引力势验证
- N=6 解析验证：Φ(w)=-1/(6-w) **零误差 PASS**
- N=12 标度律：Φ×d_H=1.000000 **PASS**
- 引力动力学：粒子被吸引（距离 -17.6%）**PASS**

### exp_40：层级涌现
- N=48 → L0(48) → L1(15) → L2(3) → L3(3)
- 3 次封装事件，每层都有九机制指标
- 全部验收通过

---

## 运行

### 安装

```bash
cd 11-模拟机工程实现
pip install numpy scipy torch pytest  # engine_v2 仅需 numpy
```

> **依赖说明**：engine_v2（当前主引擎）仅依赖 numpy。engine/（历史引擎）和 tests/ 需要 torch。

### 运行实验（engine_v2，推荐）

```bash
cd engine_v2

# 递归封装实验
PYTHONPATH=. python3 -c "
from diffsim.world_v2 import RecursiveWorld, Params
rw = RecursiveWorld(N0=36, seed=42, params=Params(max_steps=50))
report = rw.run()
for li in report['layers']:
    print(f\"L{li['layer']}: N={li['n_total']}, steps={li['steps']}, flux={li['flux']:.4f}\")
"

# 理论预测检验（7项可证伪实验）
PYTHONPATH=. python3 diffsim/experiment_falsification.py

# 单元测试
PYTHONPATH=. python3 diffsim/tests/test_closure.py
```

### 运行实验（engine v1，历史）

```bash
# 需要 torch
python scripts/_archive_phase16/run_experiment_v1.py --exp exp_40_hierarchical
```

### 运行测试

```bash
# engine_v2 测试（仅需 numpy）
cd engine_v2 && PYTHONPATH=. python3 diffsim/tests/test_closure.py

# engine v1 测试（需要 torch）
pytest tests/ -v
# 330+ tests passed, 23 skipped
```

---

## 第二阶段路线图

当前 → 象界（前主体态）：

```
跨层级引力调制
    ↓ 高层级比特的质量分布 → 低层级源/汇权重
解封机制
    ↓ 基底状态变化超过阈值 → 封装比特解冻
回流通道
    ↓ 高层级叙事状态 → 重写低层级约束矩阵
横向耦合
    ↓ 同层不同 BiasField 相互作用
长期记忆
    ↓ PersistentBiasField / BiasMemory
六阈值检测器
    ↓ 界面调节度 + 自维持稳健性 + 保持深度 + 复制保真度 + 选择压力 + 功能分化
前主体态涌现
```

---

## 第一阶段禁止事项

- ❌ 不声称模拟现实物理
- ❌ 不做完整力/量子
- ❌ 不训练大模型
- ❌ 不用视觉图案替代验证器

---

## 关联项目

| 目录 | 关系 |
|------|------|
| `01-核心理论-差异论/` | 理论底座（差异论四书） |
| `02-worldbase形式化框架/` | 公理推导链（数学证明） |
| `03-worldbase象界框架/` | 象界形式化（化学验证场） |
| `10-期货价格结构检索系统/` | 差异即世界在市场中的显影 |
| `06-广义相对论的论证/` | 从 10 公理到 Einstein 场方程的完整推导 |
| `09-莫比乌斯数学体系探索/` | 加法与乘法同源的拓扑实现 |

---

## 测试覆盖

```
330 passed, 23 skipped (pytest, 2026-05-24)
```

核心测试文件：

| 文件 | 测试数 | 覆盖 |
|------|--------|------|
| test_encapsulation.py | 29 | 封装引擎（Union-Find + 多数表决 + 解封检测） |
| test_hierarchy.py | 14 | 层级管理器（LayerState + 多层状态空间） |
| test_xiangjie.py | 19 | 象界显现链（八章门槛检测器） |
| test_bias_field.py | 12 | 回流偏置场（BiasField + 双向传播） |
| test_bias_propagation.py | 11 | 偏置传播（向下/向上 + 衰减） |
| test_hamming_layer.py | 12 | 汉明格点层（单比特翻转 + DAG 约束） |
| test_three_dim_hamming.py | 25 | 三维汉明格点 |
| test_mid_surface_analyzer.py | 10 | 中截面分析器 |
| test_first_order_algebra.py | 8 | 一阶变易代数 |
| test_detectors.py | 9 | 涌现探测器 |
| test_spatial_detectors.py | 9 | 空间探测器 |
| test_axioms_v2.py | 11 | 九公理硬性约束 |
| ... | ... | ... |

---

*GitHub: [doubc/The_Theory_of_Difference](https://github.com/doubc/The_Theory_of_Difference)*
