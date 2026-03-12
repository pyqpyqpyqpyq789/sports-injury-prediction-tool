import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===================== 全局论文级可视化设置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 配色（匹配示例图表）
COLOR_MODEL = '#1f77b4'  # 模型曲线：深蓝色
COLOR_CI = '#aec7e8'  # 置信区间：浅蓝
COLOR_ALL = '#d62728'  # 全干预：红色
COLOR_NONE = '#2ca02c'  # 无干预：绿色
ALPHA_CI = 0.3


# ===================== 核心函数：计算净获益 =====================
def calculate_net_benefit(y_true, y_proba, thresholds):
    # 先强制输入维度
    y_true = np.atleast_1d(y_true)
    y_proba = np.atleast_1d(y_proba)
    thresholds = np.atleast_1d(thresholds)

    """计算模型净获益"""
    n_total = len(y_true)
    n_pos = np.sum(y_true == 1)
    net_benefit = np.zeros_like(thresholds)

    for i, t in enumerate(thresholds):
        if t <= 0 or t >= 1:
            net_benefit[i] = 0
            continue
        pred_pos = y_proba >= t
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        net_benefit[i] = (tp / n_total) - (fp / n_total) * (t / (1 - t))
    return net_benefit


def calculate_net_benefit_all(thresholds, p_pos):
    """计算全干预净获益"""
    return p_pos - (1 - p_pos) * (thresholds / (1 - thresholds))


def calculate_net_benefit_none(thresholds):
    """计算无干预净获益（恒为0）"""
    return np.zeros_like(thresholds)


# ===================== Bootstrap计算模型置信区间 =====================
# def bootstrap_dca(y_true, y_proba, n_boot=1000, seed=42):
#     np.random.seed(seed)
#     thresholds = np.linspace(0.01, 0.99, 100)
#     thresholds = np.atleast_1d(thresholds)
#     boot_nb = np.zeros((n_boot, len(thresholds)))
#
#     for i in tqdm(range(n_boot), desc="Bootstrapping DCA"):
#         idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
#         y_boot = y_true[idx]
#         y_proba_boot = y_proba[idx]
#         boot_nb[i, :] = calculate_net_benefit(y_boot, y_proba_boot, thresholds)
#
#     mean_nb = np.mean(boot_nb, axis=0)
#     lb_nb = np.percentile(boot_nb, 2.5, axis=0)
#     ub_nb = np.percentile(boot_nb, 97.5, axis=0)
#
#     # 计算全干预/无干预净获益
#     p_pos = np.sum(y_true == 1) / len(y_true)
#     print('p_pos', p_pos)
#     nb_all = calculate_net_benefit_all(thresholds, p_pos)
#     nb_none = calculate_net_benefit_none(thresholds)
#
#     return thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none


def bootstrap_dca(y_true, y_proba, n_boot=1000, seed=42):
    np.random.seed(seed)
    # 1. 生成thresholds并强制一维（保留你的代码，增加断言）
    thresholds = np.linspace(0.01, 0.99, 100)
    thresholds = np.atleast_1d(thresholds)
    assert thresholds.ndim == 1, f"thresholds必须是一维！当前维度：{thresholds.ndim}"
    n_thresholds = len(thresholds)

    # 2. 初始化boot_nb（显式指定维度，避免隐式错误）
    boot_nb = np.zeros((n_boot, n_thresholds), dtype=np.float64)

    for i in tqdm(range(n_boot), desc="Bootstrapping DCA"):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_boot = y_true[idx]
        y_proba_boot = y_proba[idx]

        # 3. 调用calculate_net_benefit后强制一维
        nb = calculate_net_benefit(y_boot, y_proba_boot, thresholds)
        nb = np.atleast_1d(nb)  # 关键！确保返回值是一维
        assert len(nb) == n_thresholds, f"净收益长度异常：{len(nb)} != {n_thresholds}"
        boot_nb[i, :] = nb  # 此时nb是一维，赋值无问题

    # 4. 计算统计量（确保结果一维）
    mean_nb = np.mean(boot_nb, axis=0)
    lb_nb = np.percentile(boot_nb, 2.5, axis=0)
    ub_nb = np.percentile(boot_nb, 97.5, axis=0)
    mean_nb, lb_nb, ub_nb = map(np.atleast_1d, [mean_nb, lb_nb, ub_nb])

    # 5. 处理p_pos和全干预/无干预净获益（强制返回值一维）
    p_pos = np.sum(y_true == 1) / len(y_true)
    print('p_pos', p_pos)
    nb_all = calculate_net_benefit_all(thresholds, p_pos)
    nb_none = calculate_net_benefit_none(thresholds)
    # 强制一维兜底
    nb_all = np.atleast_1d(nb_all)
    nb_none = np.atleast_1d(nb_none)

    # 最终校验所有返回值维度
    assert all(arr.ndim == 1 for arr in [thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none])

    return thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none


# ===================== 绘制DCA三条曲线 =====================
def plot_dca_curves(thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none, save_path="dca_plot.pdf"):
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. 绘制模型置信区间
    ax.fill_between(thresholds, lb_nb, ub_nb, color=COLOR_CI, alpha=ALPHA_CI, label='95% CI')
    # 2. 绘制模型净获益曲线
    ax.plot(thresholds, mean_nb, color=COLOR_MODEL, linewidth=2, label='Model to predict event')
    # 3. 绘制全干预曲线
    ax.plot(thresholds, nb_all, color=COLOR_ALL, linestyle='--', linewidth=1.5, label='Net benefit: treat All')
    # 4. 绘制无干预曲线
    ax.plot(thresholds, nb_none, color=COLOR_NONE, linestyle='--', linewidth=1.5, label='Net benefit: treat None')

    # 坐标轴与标题设置
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title('Decision curve analysis (bootstrapping, n=1000)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1)
    # ax.set_ylim(-1.0, 0.65)  # 匹配示例图表的纵轴范围
    ax.set_ylim(-1.0, max(np.max(nb_all), 0.65)+0.1)  # 匹配示例图表的纵轴范围

    # 网格与图例
    ax.grid(True, linestyle=':', alpha=1.0, color='#dddddd')
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='#333333', fontsize=10) #upper

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===================== 模拟数据+运行示例 =====================
if __name__ == "__main__":
    # 模拟数据（匹配示例图表的阳性发生率≈5%）
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.05, n_samples)  # 阳性发生率5%
    y_proba = np.random.beta(2, 5, n_samples)  # 模拟模型预测概率

    # Bootstrap计算DCA数据
    thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none = bootstrap_dca(y_true, y_proba, n_boot=1000)

    # 绘制三条核心曲线
    plot_dca_curves(thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none)
