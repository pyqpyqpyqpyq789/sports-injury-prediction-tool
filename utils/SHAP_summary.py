import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import shap


def advance_summary_plot(shap_values, data, show=True):
    feature_names = data.columns
    n_features = len(feature_names)
    shap_min = shap_values.min()  # 蜂群图最小值（条形图0点对齐此值）
    shap_max = shap_values.max()  # 蜂群图最大值
    print(f"蜂群图最小值：{shap_min:.2f}，条形图0点将对齐该位置")

    # 条形图自身数值范围（独立于蜂群图）
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    bar_min = 0  # 条形图自身最小值
    bar_max = mean_abs_shap.max() * 1.05  # 条形图自身最大值

    # 特征排序
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_idx = np.flip(sorted_idx)    # 将排序倒置
    shap_values_sorted = shap_values[:, sorted_idx]

    X_sorted = data.iloc[:, sorted_idx].values
    feature_names_sorted = [feature_names[i] for i in sorted_idx]
    mean_abs_shap_sorted = mean_abs_shap[sorted_idx]

    # --------------------------
    # 3. 学术样式配置（SCI标准）
    # --------------------------
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'figure.figsize': (12, 7)
    })

    # 自定义配色（学术严谨风）
    # cmap = LinearSegmentedColormap.from_list('custom', ['#3498db', '#ffffff', '#e74c3c'])  # 蓝-白-红
    # cmap = LinearSegmentedColormap.from_list('custom_deep', ['#003366', '#ffffff', '#990000'])  # 藏青蓝-白-酒红
    # cmap = LinearSegmentedColormap.from_list('custom_mid', ['#19486A', '#ffffff', '#C0392B'])   # 中深蓝-白-中深红
    # cmap = LinearSegmentedColormap.from_list('custom', ['#3498db', '#e74c3c'])  # 蓝-红
    # cmap = LinearSegmentedColormap.from_list('blue_yellow', ['#1F78B4', '#FFFFB3'])  # 深蓝-浅黄
    # cmap = LinearSegmentedColormap.from_list('custom_deep', ['#003366', '#ffffff', '#990000']) # 加深版蓝-白-红渐变
    # cmap = shap.plots.colors.blue_red
    cmap = plt.cm.get_cmap("viridis")

    bar_color = '#2c3e50'  # 深灰蓝条形图
    edge_color = '#7f8c8d'  # 边框浅灰
    plt.style.use('ggplot')

    # --------------------------
    # 4. 核心实现：单y轴 + 双独立x轴（蜂群图x轴 + 条形图x轴，无twinx()）
    # --------------------------
    fig, ax = plt.subplots(1, 1)  # 主坐标轴：承载蜂群图 + 绘制条形图
    y_pos = np.arange(n_features)

    # --------------------------
    # 4.1 绘制条形图（0点对齐蜂群图最小值，无双y轴）
    # --------------------------
    bar_width = 0.6
    bars = ax.barh(
        y_pos,
        mean_abs_shap_sorted,  # 条形图自身数值（对应独立x轴）
        left=shap_min,  # 条形图0点对齐蜂群图最小值（画布x轴位置）
        height=bar_width,
        color=bar_color,
        alpha=0.2,
        edgecolor=edge_color,
        linewidth=0.5,
        zorder=2
    )

    # --------------------------
    # 4.2 绘制蜂群图（主坐标轴，对应蜂群图独立x轴）
    # --------------------------
    jitter = np.random.normal(0, 0.1, size=shap_values_sorted.shape)
    jitter = np.clip(jitter, -0.2, 0.2)

    scatter = None
    for i in range(n_features):
        shap_i = shap_values_sorted[:, i]  # + shap_max
        x_i = X_sorted[:, i]
        x_norm = (x_i - x_i.min()) / (x_i.max() - x_i.min() + 1e-8)

        scatter = ax.scatter(
            shap_i,
            y_pos[i] + jitter[:, i],
            c=x_norm,
            cmap=cmap,
            s=12,
            alpha=1.0,
            edgecolor='none',
            zorder=3
        )

    # --------------------------
    # 4.3 单独生成条形图独立x轴（核心：ax.twiny()，双x轴而非双y轴）
    # --------------------------
    # 创建条形图独立x轴（共享y轴，独立x轴，放在顶部/底部均可，这里先放顶部，避免重叠）
    ax_bar_x = ax.twiny()  # 关键：twiny()创建独立x轴（无twinx()双y轴）

    # 步骤1：设置条形图x轴的自身范围（独立于蜂群图x轴）
    ax_bar_x.set_xlim(bar_min, bar_max)
    # 步骤2：设置条形图x轴标签（区分于蜂群图x轴）
    ax_bar_x.set_xlabel('Mean |SHAP| (Feature Importance)', labelpad=10, color='#e74c3c')
    # 步骤3：设置条形图x轴刻度颜色（便于区分两个x轴）
    ax_bar_x.tick_params(axis='x', colors='#e74c3c')
    ax_bar_x.spines['top'].set_color('#e74c3c')  # 条形图x轴边框颜色

    # --------------------------
    # 4.4 蜂群图x轴设置（独立于条形图x轴）
    # --------------------------
    ax.set_xlim(shap_min - 0.02, max(shap_max, shap_min + bar_max) + 0.02)
    ax.set_xlabel('SHAP Value (Bee Swarm Plot)', labelpad=10, color='#2980b9')
    ax.tick_params(axis='x', colors='#2980b9')
    ax.spines['bottom'].set_color('#2980b9')

    # --------------------------
    # 4.5 y轴与样式设置（单y轴，无冗余）
    # --------------------------
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names_sorted)
    ax.set_ylim(-0.5, n_features - 0.5)

    # 样式优化
    ax.grid(axis='x', alpha=0.3, linestyle='--', zorder=1)
    ax.spines['top'].set_visible(False)  # 隐藏主坐标轴上边框
    ax.spines['right'].set_visible(False)
    ax_bar_x.spines['right'].set_visible(False)  # 隐藏条形图x轴右边框

    # 条形图数值标签（精准定位）
    for i, bar in enumerate(bars):
        bar_len = bar.get_width()
        bar_left = bar.get_x()
        ax.text(
            # bar_left + bar_len + 0.02,
            bar_left,
            bar.get_y() + bar.get_height() / 2,
            f'{bar_len:.3f}',
            ha='left', va='center', fontsize=8
        )

    # 颜色条（框外显示）
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=-0.2, aspect=30,
                        anchor=(1.5, 0.5)  # 锚定颜色条位置，防止偏移
                        )
    cbar.set_label('Feature Value (Normalized)', labelpad=10)
    cbar.outline.set_linewidth(0.5)

    # --------------------------
    # 4.6 标题
    # --------------------------
    ax.set_title(
        'Summary Plot',
        fontsize=13, fontweight='bold', pad=20
    )

    # --------------------------
    # 5. 保存图片
    # --------------------------
    if show:
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # 预留条形图x轴标签空间
        plt.savefig(
            'bar_0_align_shap_min_precise.png',
            dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none'
        )
        plt.savefig(
            'bar_0_align_shap_min_precise.pdf',
            bbox_inches='tight', facecolor='none', edgecolor='none'
        )
        plt.show()


if __name__ == "__main__":
    # --------------------------
    # 1. 数据准备（可替换为真实数据）
    # --------------------------
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    feature_names = [f'Feature_{i + 1}' for i in range(n_features)]

    # 模拟SHAP值
    shap_values = np.random.randn(n_samples, n_features) * 0.8
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
    advance_summary_plot(shap_values=shap_values, data=X, show=True)