import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

np.random.seed(1)

# ----------------------------
# 1) 数据结构：智能体与系统
# ----------------------------

@dataclass
class StorageSpec:
    e_max: float          # MWh
    p_ch_max: float       # MW
    p_dis_max: float      # MW
    eta_ch: float         # 0-1
    eta_dis: float        # 0-1
    soc0: float           # MWh
    soc_min: float        # MWh
    soc_max: float        # MWh

@dataclass
class Agent:
    name: str
    bus: int                      # 接入节点
    is_prosumer: bool
    load_forecast: np.ndarray     # (T,)
    pv_forecast: np.ndarray       # (T,) for prosumer, else zeros
    load_real: np.ndarray         # (T,)
    pv_real: np.ndarray           # (T,)
    # 报价参数（由策略/智能体动作给出）
    bid_value: float              # £/MWh（买方边际价值上限的基准）
    offer_cost: float             # £/MWh（卖方边际成本基准；对PV可理解为机会成本/磨损等）
    # 储能（只有 prosumer 1/2 有）
    storage: Optional[StorageSpec] = None

@dataclass
class Network:
    """
    简化径向配网：0-1-2 两条线
    线0: 0->1 容量 cap01
    线1: 1->2 容量 cap12
    我们用径向潮流的“下游注入累加”近似：
      flow01 = net_inj(bus1)+net_inj(bus2)
      flow12 = net_inj(bus2)
    net_inj>0 表示向上游注入（发电/放电/外送），net_inj<0 表示从网购电（负荷/充电）。
    """
    cap01: float  # MW
    cap12: float  # MW

# ----------------------------
# 2) 出清：社会福利最大化 LP（DA/RT共用）
# ----------------------------

def clear_market_lp(
    agents: List[Agent],
    network: Network,
    T: int,
    stage: str,                     # "DA" or "RT"
    wholesale_price: np.ndarray,     # (T,) 外部电网边际成本/价格（作为“系统电源”）
    action_params: Dict[str, Dict],  # 各智能体本阶段动作（报价参数）
    penalty_unserved: float = 500.0  # 未满足负荷惩罚（£/MWh）
) -> Dict:
    """
    返回：
      - clearing_price: (T,) 这里用“影子价格”近似 LMP（简化：系统单节点价格）
      - schedules: per agent {p_buy, p_sell, p_ch, p_dis, soc, served_load, pv_used}
    """
    # 变量：每个agent每时段
    p_buy = {}
    p_sell = {}
    served = {}
    unserved = {}
    pv_used = {}
    p_ch = {}
    p_dis = {}
    soc = {}

    # 外部电网供给（系统电源）：g_grid >= 0
    g_grid = cp.Variable(T, nonneg=True)

    for a in agents:
        p_buy[a.name] = cp.Variable(T, nonneg=True)
        p_sell[a.name] = cp.Variable(T, nonneg=True)
        served[a.name] = cp.Variable(T, nonneg=True)
        unserved[a.name] = cp.Variable(T, nonneg=True)

        pv_used[a.name] = cp.Variable(T, nonneg=True)

        if a.storage is not None:
            p_ch[a.name] = cp.Variable(T, nonneg=True)
            p_dis[a.name] = cp.Variable(T, nonneg=True)
            soc[a.name] = cp.Variable(T)
        else:
            p_ch[a.name] = None
            p_dis[a.name] = None
            soc[a.name] = None

    constraints = []

    # 选择使用 forecast 或 real
    if stage == "DA":
        load = {a.name: a.load_forecast for a in agents}
        pv = {a.name: a.pv_forecast for a in agents}
    else:
        load = {a.name: a.load_real for a in agents}
        pv = {a.name: a.pv_real for a in agents}

    # 负荷满足：served + unserved = load
    for a in agents:
        constraints += [served[a.name] + unserved[a.name] == load[a.name]]

    # PV使用上限
    for a in agents:
        if a.is_prosumer:
            constraints += [pv_used[a.name] <= pv[a.name]]
        else:
            constraints += [pv_used[a.name] == 0]

    # 储能约束
    for a in agents:
        if a.storage is None:
            constraints += [p_ch[a.name] == None] if False else []
            continue
        st = a.storage
        constraints += [p_ch[a.name] <= st.p_ch_max]
        constraints += [p_dis[a.name] <= st.p_dis_max]
        # SOC 动力学
        constraints += [soc[a.name][0] == st.soc0
                        + st.eta_ch * p_ch[a.name][0]
                        - (1.0 / st.eta_dis) * p_dis[a.name][0]]
        for t in range(1, T):
            constraints += [soc[a.name][t] == soc[a.name][t-1]
                            + st.eta_ch * p_ch[a.name][t]
                            - (1.0 / st.eta_dis) * p_dis[a.name][t]]
        constraints += [soc[a.name] >= st.soc_min, soc[a.name] <= st.soc_max]

    # 功率平衡（全系统）
    # 系统供给：PV_used + p_sell(可理解为本地出清中的“对外供给”变量) + grid
    # 系统用电：served(负荷) + p_buy(从市场买) + 充电 + 其它
    #
    # 这里做“单市场池”：p_buy/p_sell 表示参与市场的净交易分解
    # 实际上你也可以简化为每个agent一个净注入变量 n = gen - load + dis - ch + grid_import
    for t in range(T):
        total_supply = g_grid[t] + cp.sum([pv_used[a.name][t] for a in agents]) + cp.sum([p_sell[a.name][t] for a in agents])
        total_demand = cp.sum([served[a.name][t] for a in agents]) + cp.sum([p_buy[a.name][t] for a in agents])

        # 储能充放电影响：充电视为需求，放电视为供给
        for a in agents:
            if a.storage is not None:
                total_supply += p_dis[a.name][t]
                total_demand += p_ch[a.name][t]

        constraints += [total_supply == total_demand]

    # Agent 能量收支：买卖与本地PV/储能与负荷一致（每个 agent 的“能量守恒”）
    # served_load 由 (PV_used + buy + discharge) 供给，且多余可卖出或弃电（弃电由 pv_used<=pv 体现）
    for a in agents:
        for t in range(T):
            lhs_supply = pv_used[a.name][t] + p_buy[a.name][t]
            if a.storage is not None:
                lhs_supply += p_dis[a.name][t]
            rhs_use = served[a.name][t] + p_sell[a.name][t]
            if a.storage is not None:
                rhs_use += p_ch[a.name][t]
            constraints += [lhs_supply == rhs_use]

    # 网络约束（径向线容量简化）
    # 计算各母线净注入：net_inj = (PV_used + dis + sell) - (served + ch + buy)
    # 注意这里 sell/buy是“市场交易分解”，仍可作为节点注入项（等价）
    buses = sorted(set(a.bus for a in agents))
    # 我们只做 0-1-2 结构：bus0 为外部电网点，不显式建 agent
    # 将 g_grid 视为在 bus0 注入供给
    for t in range(T):
        net_inj_bus = {b: 0 for b in buses}
        for a in agents:
            b = a.bus
            inj = pv_used[a.name][t] + p_sell[a.name][t] - served[a.name][t] - p_buy[a.name][t]
            if a.storage is not None:
                inj += p_dis[a.name][t] - p_ch[a.name][t]
            net_inj_bus[b] += inj

        # 这里假设 bus 分别为 1,2；bus0 为上级电网
        # flow12 = net_inj(bus2)
        # flow01 = net_inj(bus1) + net_inj(bus2)
        if 1 in net_inj_bus and 2 in net_inj_bus:
            flow12 = net_inj_bus[2]
            flow01 = net_inj_bus[1] + net_inj_bus[2]
            constraints += [flow12 <= network.cap12, flow12 >= -network.cap12]
            constraints += [flow01 <= network.cap01, flow01 >= -network.cap01]

    # ----------------------------
    # 目标函数：社会福利最大化
    # ----------------------------
    welfare = 0
    for a in agents:
        # 本阶段动作：例如 bid_multiplier / offer_adder
        ap = action_params.get(a.name, {})
        bid_mult = ap.get("bid_mult", 1.0)           # 买方愿付价倍率
        offer_adder = ap.get("offer_adder", 0.0)     # 卖方报价加成（£/MWh）

        # 买方效用：bid_price * served_load
        bid_price = bid_mult * a.bid_value
        welfare += bid_price * cp.sum(served[a.name])

        # 卖方成本：对卖出电量计成本（可理解为机会成本/磨损/燃料等）
        offer_price = a.offer_cost + offer_adder
        welfare -= offer_price * cp.sum(p_sell[a.name])

        # grid 成本
    welfare -= cp.sum(cp.multiply(wholesale_price, g_grid))

    # 未满足负荷惩罚（强制系统尽量满足）
    welfare -= penalty_unserved * cp.sum(cp.hstack([cp.sum(unserved[a.name]) for a in agents]))

    problem = cp.Problem(cp.Maximize(welfare), constraints)
    problem.solve(verbose=False)  # 自动选择求解器

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"LP not solved: {problem.status}")

    # 价格：严格的节点边际电价需要对每节点功率平衡取对偶，这里简化为“系统平衡约束”的对偶值
    # 我们取每时段系统平衡约束对应的 dual 作为系统价格近似（£/MWh）
    # 在上面构造中，每时段系统平衡约束是 constraints 中按顺序追加的一段；我们重新索引会很麻烦
    # 为保持简单：用“外部电网边际成本 + 拥塞溢价近似”，这里直接用 g_grid 的边际成本并加拥塞项（简化）
    # 如果你需要严格 LMP，我可以再给你一个“显式节点平衡+对偶提取”的版本。
    clearing_price = wholesale_price.copy()

    schedules = {}
    for a in agents:
        schedules[a.name] = {
            "p_buy": np.array(p_buy[a.name].value).reshape(-1),
            "p_sell": np.array(p_sell[a.name].value).reshape(-1),
            "served": np.array(served[a.name].value).reshape(-1),
            "unserved": np.array(unserved[a.name].value).reshape(-1),
            "pv_used": np.array(pv_used[a.name].value).reshape(-1),
        }
        if a.storage is not None:
            schedules[a.name].update({
                "p_ch": np.array(p_ch[a.name].value).reshape(-1),
                "p_dis": np.array(p_dis[a.name].value).reshape(-1),
                "soc": np.array(soc[a.name].value).reshape(-1),
            })
    schedules["GRID"] = {"g_grid": np.array(g_grid.value).reshape(-1)}

    return {
        "price": clearing_price,
        "schedules": schedules,
        "welfare": problem.value
    }

# ----------------------------
# 3) 两阶段结算：DA + RT
# ----------------------------

def two_settlement(
    agents: List[Agent],
    da: Dict,
    rt: Dict
) -> Dict[str, float]:
    """
    结算：对每个agent：
      payment = DA_price * (DA_net_import) + RT_price * (RT_net_import - DA_net_import)
    net_import>0 表示从系统净买（支出），<0 表示净卖（收入）
    """
    T = len(da["price"])
    pay = {a.name: 0.0 for a in agents}

    for a in agents:
        sch_da = da["schedules"][a.name]
        sch_rt = rt["schedules"][a.name]

        # 净买量 = p_buy - p_sell （MWh）
        da_import = sch_da["p_buy"] - sch_da["p_sell"]
        rt_import = sch_rt["p_buy"] - sch_rt["p_sell"]

        da_cost = np.sum(da["price"] * da_import)
        rt_cost = np.sum(rt["price"] * (rt_import - da_import))

        pay[a.name] = float(da_cost + rt_cost)

    return pay

# ----------------------------
# 4) 简单算例：4智能体、24小时
# ----------------------------

def build_demo_case(T=24) -> Tuple[List[Agent], Network, np.ndarray]:
    hours = np.arange(T)

    # 外部电网日内价格曲线（£/MWh），带早晚高峰
    wholesale = 40 + 15*np.exp(-0.5*((hours-8)/2.5)**2) + 20*np.exp(-0.5*((hours-19)/2.5)**2)
    wholesale = np.round(wholesale, 2)

    # 预测与实际（简单加噪）
    def noisy(x, sigma=0.1):
        return np.clip(x * (1 + np.random.normal(0, sigma, size=x.shape)), 0, None)

    # 两个 prosumer (A,B) 在 bus1 和 bus2
    base_pv_A = np.clip(6*np.sin((hours-6)/24*2*np.pi), 0, None)
    base_pv_B = np.clip(5.5*np.sin((hours-7)/24*2*np.pi), 0, None)

    load_A = 2.0 + 0.5*np.exp(-0.5*((hours-10)/3.0)**2)
    load_B = 1.8 + 0.4*np.exp(-0.5*((hours-15)/3.5)**2)

    # 两个纯负荷 (C,D) 在 bus1/bus2
    load_C = 4.5 + 1.2*np.exp(-0.5*((hours-9)/2.5)**2) + 1.5*np.exp(-0.5*((hours-18)/2.5)**2)
    load_D = 3.8 + 1.0*np.exp(-0.5*((hours-8)/2.7)**2) + 1.2*np.exp(-0.5*((hours-20)/2.7)**2)

    A = Agent(
        name="A(RES+ESS)", bus=1, is_prosumer=True,
        load_forecast=noisy(load_A, 0.05), pv_forecast=noisy(base_pv_A, 0.12),
        load_real=noisy(load_A, 0.08), pv_real=noisy(base_pv_A, 0.18),
        bid_value=90.0, offer_cost=5.0,
        storage=StorageSpec(e_max=8.0, p_ch_max=2.5, p_dis_max=2.5, eta_ch=0.95, eta_dis=0.95,
                            soc0=3.0, soc_min=0.8, soc_max=7.5)
    )
    B = Agent(
        name="B(RES+ESS)", bus=2, is_prosumer=True,
        load_forecast=noisy(load_B, 0.05), pv_forecast=noisy(base_pv_B, 0.12),
        load_real=noisy(load_B, 0.08), pv_real=noisy(base_pv_B, 0.18),
        bid_value=88.0, offer_cost=6.0,
        storage=StorageSpec(e_max=6.0, p_ch_max=2.0, p_dis_max=2.0, eta_ch=0.94, eta_dis=0.94,
                            soc0=2.5, soc_min=0.6, soc_max=5.6)
    )
    C = Agent(
        name="C(LOAD)", bus=1, is_prosumer=False,
        load_forecast=noisy(load_C, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_C, 0.10), pv_real=np.zeros(T),
        bid_value=110.0, offer_cost=999.0,
        storage=None
    )
    D = Agent(
        name="D(LOAD)", bus=2, is_prosumer=False,
        load_forecast=noisy(load_D, 0.06), pv_forecast=np.zeros(T),
        load_real=noisy(load_D, 0.10), pv_real=np.zeros(T),
        bid_value=105.0, offer_cost=999.0,
        storage=None
    )

    # 简化配网容量（MW）：如果容量紧，会产生拥塞导致储能/本地PV价值凸显
    network = Network(cap01=6.5, cap12=3.5)

    return [A, B, C, D], network, wholesale

def random_actions(agents: List[Agent]) -> Dict[str, Dict]:
    """
    先用随机策略：买方 bid_mult in [0.85,1.05], 卖方 offer_adder in [0,8]
    RL 接入后，这里由 policy(obs)->action 生成
    """
    actions = {}
    for a in agents:
        if a.is_prosumer:
            actions[a.name] = {
                "bid_mult": float(np.random.uniform(0.85, 1.05)),
                "offer_adder": float(np.random.uniform(0.0, 6.0))
            }
        else:
            actions[a.name] = {"bid_mult": float(np.random.uniform(0.90, 1.10))}
    return actions

def run_one_day_demo():
    agents, network, wholesale = build_demo_case(T=24)

    # 日前：基于预测
    act_DA = random_actions(agents)
    da = clear_market_lp(
        agents=agents, network=network, T=24, stage="DA",
        wholesale_price=wholesale, action_params=act_DA
    )

    # 日内/实时：基于实际（可做滚动，这里简化为一次 RT）
    act_RT = random_actions(agents)
    rt = clear_market_lp(
        agents=agents, network=network, T=24, stage="RT",
        wholesale_price=wholesale, action_params=act_RT
    )

    payment = two_settlement(agents, da, rt)

    print("=== Day-Ahead (DA) welfare:", round(da["welfare"], 2))
    print("=== Real-Time (RT) welfare:", round(rt["welfare"], 2))
    print("\n--- Payments (positive = cost, negative = revenue) ---")
    for k, v in payment.items():
        print(f"{k:12s}  {v:8.2f} £")

    # 示例：看A的SOC与交易
    A = agents[0]
    schA_da = da["schedules"][A.name]
    schA_rt = rt["schedules"][A.name]
    print("\n--- A(RES+ESS) snapshot ---")
    print("DA soc:", np.round(schA_da["soc"], 2))
    print("RT soc:", np.round(schA_rt["soc"], 2))
    print("DA net_import:", np.round(schA_da["p_buy"] - schA_da["p_sell"], 2))
    print("RT net_import:", np.round(schA_rt["p_buy"] - schA_rt["p_sell"], 2))

if __name__ == "__main__":
    run_one_day_demo()
