import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve

# ===================== 全局论文级可视化参数 =====================
plt.rcParams['font.family'] = 'DejaVu Sans'  # 统一字体
plt.rcParams['font.size'] = 12  # 基础字号
plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线条粗细
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['figure.dpi'] = 300  # 高分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'  # 去除空白边缘

# 专业配色方案（学术期刊友好）
COLOR_PERFECT = '#333333'       # 完美校准线：深灰色
COLOR_MODEL = '#1f77b4'         # 模型均值曲线：深蓝色
COLOR_CI = '#aec7e8'            # 置信区间：浅蓝（半透明）
COLOR_YOUDEN = '#d62728'        # Youden J阈值：红色
ALPHA_CI = 0.3                  # 置信区间透明度
LINEWIDTH_MODEL = 2.0           # 模型曲线粗细
LINEWIDTH_PERFECT = 1.0         # 完美校准线粗细

# ===================== 辅助函数：计算Youden J最优阈值 =====================
def find_youden_threshold(y_true, y_proba):
    """计算Youden J指数对应的最优阈值（可选标注）"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]

# ===================== 核心函数：Bootstrapping计算校准曲线置信区间 =====================
def bootstrap_calibration(y_true, y_proba, n_bins=10, n_boot=1000, seed=42):
    """
    基于Bootstrapping计算校准曲线的均值和95%置信区间
    :param y_true: 真实标签（0/1）
    :param y_proba: 预测概率
    :param n_bins: 分组数（论文常用10组）
    :param n_boot: Bootstrapping次数（1000次保证稳健性）
    :param seed: 随机种子（可复现）
    :return: pred_bins, mean_true, lb_true, ub_true, brier_score
    """
    np.random.seed(seed)
    # 初始化存储每次Bootstrap的实际概率
    boot_true = []
    
    # 进度条显示
    for _ in tqdm(range(n_boot), desc="Bootstrapping Calibration"):
        # 有放回抽样
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_boot = np.array(y_true)[idx]
        y_proba_boot = np.array(y_proba)[idx]
        # 计算本次抽样的校准曲线（等频分组，避免样本不均）
        prob_true_boot, prob_pred_boot = calibration_curve(
            y_boot, y_proba_boot, n_bins=n_bins, strategy='quantile'
        )
        boot_true.append(prob_true_boot)
    
    # 计算均值和95%置信区间
    boot_true = np.array(boot_true)
    pred_bins = prob_pred_boot  # 分组的预测概率均值（取最后一次，或均值，差异极小）
    mean_true = np.mean(boot_true, axis=0)
    lb_true = np.percentile(boot_true, 2.5, axis=0)  # 下边界
    ub_true = np.percentile(boot_true, 97.5, axis=0)  # 上边界
    
    # 计算整体Brier分数（校准好坏的量化指标）
    brier_score = brier_score_loss(y_true, y_proba)
    
    return pred_bins, mean_true, lb_true, ub_true, brier_score

# ===================== 核心函数：绘制论文级校准曲线 =====================
def plot_paper_calibration(
    y_true, y_proba, model_name,
    n_bins=10, n_boot=1000,
    plot_youden=True,  # 是否标注Youden J阈值
    save_path="calibration_curve.pdf"
):
    """
    绘制论文级校准曲线（含Bootstrapping置信区间）
    :param plot_youden: 是否标注Youden J最优阈值
    :param save_path: 保存路径（PDF/PNG/SVG）
    """
    # 1. 计算校准曲线均值+置信区间+Brier分数
    pred_bins, mean_true, lb_true, ub_true, brier = bootstrap_calibration(
        y_true, y_proba, n_bins, n_boot
    )
    
    # 2. 计算Youden J阈值（可选）
    if plot_youden:
        youden_thresh = find_youden_threshold(y_true, y_proba)
        # 找到Youden阈值所在的分组
        youden_bin_idx = np.argmin(np.abs(pred_bins - youden_thresh))
        youden_actual = mean_true[youden_bin_idx]
    
    # 3. 绘制校准曲线
    fig, ax = plt.subplots(figsize=(8, 6))  # 论文常用尺寸（8×6英寸）
    
    # 绘制置信区间（浅蓝半透明填充）
    ax.fill_between(pred_bins, lb_true, ub_true, color=COLOR_CI, alpha=ALPHA_CI, label='95% CI')
    # 绘制模型校准曲线（深蓝色粗线）
    ax.plot(pred_bins, mean_true, color=COLOR_MODEL, linewidth=LINEWIDTH_MODEL, 
            label=f'Model (Brier = {brier:.3f})')
    # 绘制完美校准线（黑色虚线）
    ax.plot([0, 1], [0, 1], color=COLOR_PERFECT, linestyle='--', linewidth=LINEWIDTH_PERFECT,
            label='Perfect Calibration')
    
    # 4. 标注Youden J最优阈值（可选）
    if plot_youden:
        ax.axvline(x=youden_thresh, color=COLOR_YOUDEN, linestyle=':', linewidth=1.5,
                   label=f'Youden J Threshold: {youden_thresh:.3f}')
        # 标注该阈值对应的实际概率
        ax.scatter(youden_thresh, youden_actual, color=COLOR_YOUDEN, s=80, zorder=5)
        ax.annotate(f'Actual: {youden_actual:.3f}',
                    xy=(youden_thresh, youden_actual),
                    xytext=(youden_thresh+0.05, youden_actual-0.08),
                    arrowprops=dict(arrowstyle='->', color=COLOR_YOUDEN, lw=1.0))
    
    # 5. 坐标轴与标签设置（论文级样式）
    ax.set_xlabel('Predicted Probability of Positive Class', fontsize=12)
    ax.set_ylabel('Actual Probability of Positive Class', fontsize=12)
    ax.set_title('Calibration curve (bootstrapping, n=1000)', fontsize=14, fontweight='bold', pad=15)
    
    # 6. 坐标轴范围与刻度
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    
    # 7. 美化：去除顶部/右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_PERFECT)
    ax.spines['bottom'].set_color(COLOR_PERFECT)
    
    # 8. 图例设置（论文级样式）
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor=COLOR_PERFECT,
              fontsize=10, borderaxespad=0.5)
    
    # 9. 网格线（浅灰色细网格）
    ax.grid(True, linestyle=':', alpha=1.0, color='#dddddd')
    
    # 10. 保存图片（高分辨率，无空白）
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ===================== 主流程：模拟数据+运行示例 =====================
if __name__ == "__main__":
    # --------------- 步骤1：生成/加载数据（替换为真实数据） ---------------
    # 模拟二分类数据（真实数据需替换：y_test为0/1标签，y_proba为预测概率）
    X, y = make_classification(
        n_samples=5000, n_features=10, n_informative=5,
        random_state=42, weights=[0.7, 0.3]  # 类别不平衡（贴近真实场景）
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # --------------- 步骤2：训练模型（替换为任意二分类模型） ---------------
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]  # 正例概率
    
    # --------------- 步骤3：绘制论文级校准曲线 ---------------
    plot_paper_calibration(
        y_true=y_test,
        y_proba=y_proba,
        n_bins=10,          # 10组分组（论文常用）
        n_boot=1000,        # 1000次Bootstrapping
        plot_youden=True,   # 标注Youden J阈值（可选关闭）
        save_path="calibration_curve.pdf"  # 保存为PDF（矢量图）
    )
