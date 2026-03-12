import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, \
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, cohen_kappa_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

import matplotlib.pyplot as plt

plt.style.use('ggplot')
# matplotlib.use('Agg')  # Set the backend to Agg

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.max_rows', None)# 显示所有行
pd.set_option('max_colwidth', 100)  # 设置value的显示长度
pd.set_option('display.width', 1000)  # 设置1000列时才换行

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
import itertools
import scipy.stats as stats
# from MLstatkit.stats import Delong_test
import shap
from pdpbox import pdp
# from dtreeviz.trees import *
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import pickle


def search_space(name):
    spaces = {
    # 1. 逻辑回归 (LogisticRegression)
    'Logistic Regression': {
        "C": Real(1e-4, 1e2, "log-uniform"),  # 正则化强度的倒数（越小正则化越强）
        "penalty": Categorical(["l2", "l1", "elasticnet"]),  # 正则化类型
        "solver": Categorical(["saga"]),  # 仅saga支持所有penalty，避免兼容性问题
        "l1_ratio": Real(0.0, 1.0),  # elasticnet的混合比例（仅penalty=elasticnet时有效）
        "multi_class": Categorical(["auto", "ovr"]),  # 多类处理方式
        "fit_intercept": Categorical([True, False])  # 是否计算截距
    },

    # 2. 支持向量机 (SVC)
    'SVC': {
        # 正则化强度：C越大，正则化越弱（模型越容易过拟合）
        "C": Real(1e-3, 1e3, "log-uniform"),  # 对数均匀分布，覆盖小值到较大值
        "kernel": Categorical(["linear", "rbf", "poly", "sigmoid"]),  # 核函数类型：线性核（linear）、RBF核（rbf）等
        "gamma": Real(1e-4, 1e2, "log-uniform"),
        # 核系数（仅对rbf/poly/sigmoid有效）："scale"表示1/(n_features * X.var())，"auto"表示1/n_features
        "degree": Integer(2, 5),  # 多项式核的阶数（仅对poly有效）, 通常2-5阶足够，过高会增加复杂度
        "shrinking": Categorical([True, False]),  # 是否使用收缩启发式（加速训练）
        "class_weight": Categorical(["balanced", None])  # 类别权重（处理不平衡数据）
    },

    # 3. K近邻 (KNeighborsClassifier)
    'KNN': {
        "n_neighbors": Integer(3, 20),  # 邻居数量
        "weights": Categorical(["uniform", "distance"]),  # 权重计算方式（均匀/距离加权）
        "metric": Categorical(["euclidean", "manhattan", "minkowski"]),  # 距离度量
        "p": Integer(1, 3)  # minkowski距离的参数（1=曼哈顿，2=欧氏）
    },

    # 4. 高斯朴素贝叶斯 (GaussianNB)
    'Naive Bayes': {
        "var_smoothing": Real(1e-10, 1e-6, "log-uniform")  # 方差平滑（数值稳定性）
    },

    # 5. 决策树 (DecisionTreeClassifier)
    'Decision Tree': {
        "criterion": Categorical(["gini", "entropy"]),  # 分裂标准（基尼系数/信息熵）
        "max_depth": Categorical([None] + list(range(3, 21))),  # 树最大深度（None表示不限制）
        "min_samples_split": Integer(2, 20),  # 分裂内部节点的最小样本数
        "min_samples_leaf": Integer(1, 10),  # 叶节点的最小样本数
        "max_features": Categorical(["sqrt", "log2", None]),  # 分裂时考虑的特征比例
        "splitter": Categorical(["best", "random"]),  # 分裂策略（最优/随机）
        "class_weight": Categorical(["balanced", None])  # 类别权重
    },

    # 6. 随机森林 (RandomForestClassifier)
    'Random Forest': {
        "n_estimators": Integer(50, 1000),  # 树的数量
        "criterion": Categorical(["gini", "entropy"]),  # 分裂标准
        "max_depth": Categorical([None] + list(range(3, 21))),  # 树最大深度
        "min_samples_split": Integer(2, 20),  # 分裂最小样本数
        "min_samples_leaf": Integer(1, 10),  # 叶节点最小样本数
        "max_features": Categorical(["sqrt", "log2", None]),  # 特征采样
        "bootstrap": Categorical([True, False]),  # 是否bootstrap抽样
        "class_weight": Categorical(["balanced", "balanced_subsample", None]),  # 类别权重
        # "oob_score": Categorical([True, False])  # 是否用袋外样本评估（仅bootstrap=True有效）
    },

    # 7. 梯度提升树 (GradientBoostingClassifier)
    'GBDT': {
        "n_estimators": Integer(50, 1000),  # 弱学习器数量
        "learning_rate": Real(0.001, 0.3, "log-uniform"),  # 学习率（衰减系数）
        "max_depth": Integer(3, 10),  # 树深度（控制复杂度）
        "min_samples_split": Integer(2, 20),  # 分裂最小样本数
        "min_samples_leaf": Integer(1, 10),  # 叶节点最小样本数
        "max_features": Categorical(["sqrt", "log2", None]),  # 特征采样
        "subsample": Real(0.5, 1.0),  # 样本采样比例（随机梯度提升）
        "criterion": Categorical(["friedman_mse", "squared_error"]),  # 分裂标准
        "loss": Categorical(["log_loss", "deviance"])  # 分类损失函数
    },

    # 8. 直方图梯度提升 (HistGradientBoostingClassifier)
    'HGBDT': {
        # "n_estimators": Integer(50, 1000),  # 弱学习器数量
        "max_iter": Integer(50, 1000),  # 迭代次数（树的数量），替代 n_estimators
        "learning_rate": Real(0.001, 0.3, "log-uniform"),  # 学习率
        "max_depth": Categorical([None] + list(range(3, 21))),  # 树深度
        "min_samples_leaf": Integer(1, 30),  # 叶节点最小样本数
        "max_bins": Integer(32, 255),  # 直方图的分箱数量（影响特征离散化）
        "l2_regularization": Real(1e-5, 1e2, "log-uniform"),  # L2正则化
        "early_stopping": Categorical([True, False])  # 是否早停（防止过拟合）
    },

    # 9. AdaBoost (AdaBoostClassifier)
    'AdaBoost': {
        "n_estimators": Integer(50, 500),  # 弱学习器数量
        "learning_rate": Real(0.001, 1.0, "log-uniform"),  # 学习率（衰减系数）
        "algorithm": Categorical(["SAMME", "SAMME.R"]),  # 算法（SAMME.R使用概率估计）
        # base_estimator默认使用决策树，此处不定义（保持默认）
    },

    # 10. CatBoost (CatBoostClassifier) - 需要安装catboost库
    'CatBoost': {
        "iterations": Integer(50, 1000),  # 迭代次数（类似n_estimators）
        "learning_rate": Real(0.001, 0.3, "log-uniform"),  # 学习率
        "depth": Integer(3, 10),  # 树深度
        "l2_leaf_reg": Real(1e-3, 1e2, "log-uniform"),  # 叶节点L2正则化
        "subsample": Real(0.5, 1.0),  # 样本采样比例
        "eval_metric": Categorical(["Logloss", "AUC"]),  # 评估指标
        "early_stopping_rounds": Categorical([None] + list(range(10, 51)))  # 早停轮数
    },

    # 11. 极端随机树 (ExtraTreesClassifier)
    'Extra Trees': {
        "n_estimators": Integer(50, 1000),  # 树的数量
        "criterion": Categorical(["gini", "entropy"]),  # 分裂标准
        "max_depth": Categorical([None] + list(range(3, 21))),  # 树最大深度
        "min_samples_split": Integer(2, 20),  # 分裂最小样本数
        "min_samples_leaf": Integer(1, 10),  # 叶节点最小样本数
        "max_features": Categorical(["sqrt", "log2", None]),  # 特征采样
        "bootstrap": Categorical([True, False]),  # 是否bootstrap抽样
        "class_weight": Categorical(["balanced", "balanced_subsample", None]),  # 类别权重
        # "oob_score": Categorical([True, False])  # 袋外评估（仅bootstrap=True有效）
    },

    # 12. LightGBM (LGBMClassifier) - 需要安装lightgbm库
    'LGBM': {
        "n_estimators": Integer(50, 1000),  # 迭代次数
        "learning_rate": Real(0.001, 0.3, "log-uniform"),  # 学习率
        "max_depth": Categorical([-1] + list(range(3, 11))),  # 树深度（-1表示不限制）
        "num_leaves": Integer(20, 150),  # 叶子数量（通常 < 2^max_depth）
        "min_child_samples": Integer(5, 100),  # 子节点最小样本数
        "subsample": Real(0.5, 1.0),  # 样本采样比例
        "colsample_bytree": Real(0.5, 1.0),  # 特征采样比例
        "reg_alpha": Real(1e-5, 1e2, "log-uniform"),  # L1正则化
        "reg_lambda": Real(1e-5, 1e2, "log-uniform")  # L2正则化
    },

    # 13. 定义超参数搜索空间（根据需求调整范围）
    'Bagging': {
        # BaggingClassifier自身的超参数
        'n_estimators': Integer(low=50, high=500),  # 集成的基分类器数量（50-500）
        'max_samples': Real(low=0.5, high=1.0),  # 每个基分类器使用的样本比例（50%-100%）
        'max_features': Real(low=0.5, high=1.0),  # 每个基分类器使用的特征比例（50%-100%）
        'bootstrap': Categorical(categories=[True, False]),  # 是否对样本有放回抽样
        'bootstrap_features': Categorical(categories=[True, False]),  # 是否对特征有放回抽样

        # 基分类器（DecisionTreeClassifier）的超参数（嵌套参数用"base_estimator__参数名"）
        'base_estimator__max_depth': Integer(low=3, high=10),  # 决策树最大深度（3-10）
        'base_estimator__min_samples_split': Integer(low=2, high=20)  # 决策树分裂的最小样本数（2-20）
    },
    # 14. XGBoost (XGBClassifier) - 需要安装xgboost库
    "XGBoost": {
        "n_estimators": Integer(50, 1000),  # 迭代次数
        "learning_rate": Real(0.001, 0.3, "log-uniform"),  # 学习率
        "max_depth": Integer(3, 10),  # 树深度
        "min_child_weight": Integer(1, 10),  # 子节点最小权重和
        "gamma": Real(1e-9, 10, "log-uniform"),  # 分裂所需最小损失减少
        "subsample": Real(0.5, 1.0),  # 样本采样比例
        "colsample_bytree": Real(0.5, 1.0),  # 特征采样比例
        "reg_alpha": Real(1e-5, 1e2, "log-uniform"),  # L1正则化
        "reg_lambda": Real(1e-5, 1e2, "log-uniform"),  # L2正则化
        "objective": Categorical(["binary:logistic"])  # 二分类目标函数（多分类需调整）
    },
    }
    return spaces.get(name, {})
    # if name == 'Logistic Regression':
    #     return logistic_space
    # elif name == 'SVM':
    #     return svc_space
    # elif name == 'KNN':
    #     return knn_space
    # elif name == 'Naive Bayes':
    #     return gaussian_nb_space
    # elif name == 'Decision Tree':
    #     return dt_space
    # elif name == 'Random Forest':
    #     return rf_space
    # elif name == 'GBDT':
    #     return gbt_space
    # elif name == 'HGBDT':
    #     return hist_gbt_space
    # elif name == 'AdaBoost':
    #     return adaboost_space
    # elif name == 'CatBoost':
    #     return catboost_space
    # elif name == 'Extra Trees':
    #     return extra_trees_space
    # elif name == 'LGBM':
    #     return lgbm_space
    # elif name == 'Bagging':
    #     return bagging_space
    # elif name == 'XGBoost':
    #     return xgb_space


def delong_auc_ci(y_true, y_score, alpha=0.05):
    """用DeLong方法计算AUC的(1-alpha)置信区间"""
    # 分离正类和负类的预测得分
    y1 = y_score[y_true == 1]  # 正类得分
    y0 = y_score[y_true == 0]  # 负类得分
    print('delong_auc_ci: y1', y1)
    n1, n0 = len(y1), len(y0)  # 正/负类样本量

    # 计算AUC的方差（DeLong核心公式）
    v10 = np.zeros((n1, n0))
    for i in range(n1):
        for j in range(n0):
            v10[i, j] = 1 if y1[i] > y0[j] else (0.5 if y1[i] == y0[j] else 0)
    auc = np.mean(v10)  # 验证AUC计算

    # 计算每个正类样本对AUC的贡献及其方差
    s1 = np.zeros(n1)
    for i in range(n1):
        s1[i] = np.mean((y1[i] > y0) - auc)
    var1 = np.var(s1, ddof=1) / n1

    # 计算每个负类样本对AUC的贡献及其方差
    s0 = np.zeros(n0)
    for j in range(n0):
        s0[j] = np.mean((y1 > y0[j]) - auc)
    var0 = np.var(s0, ddof=1) / n0

    # 总方差和标准误
    var_auc = var1 + var0
    se_auc = np.sqrt(var_auc)

    # 95%置信区间（正态近似）
    z = np.abs(np.percentile(np.random.normal(0, 1, 100000), 100*(1 - alpha/2)))  # ~1.96
    lower = auc - z * se_auc
    upper = auc + z * se_auc
    return (max(0, lower), min(1, upper))  # 确保在[0,1]范围内


def classification_metrics(y_true_, y_pred_, y_proba_, name_):
    """
    计算二分类模型的常用指标
    """
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true_, y_pred_).ravel()
    # 准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # 灵敏度/召回率
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    # 特异性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # 精确率/阳性预测值（PPV）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # F1分数
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    # 阴性预测值（NPV）
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    kappa = cohen_kappa_score(y_true_, y_pred_)
    # AUC及其95%置信区间
    fpr, tpr, _ = roc_curve(y_true_, y_proba_)
    roc_auc = auc(fpr, tpr)
    # var = auc_var(y_true_, y_proba_)
    # se = np.sqrt(roc_auc * (1 - roc_auc) + var)
    # ci_lower = roc_auc - 1.96 * se
    # ci_upper = roc_auc + 1.96 * se
    delong_ci = delong_auc_ci(y_true=y_true_, y_score=y_proba_)

    return {
        'Algorithm': name_,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        'tp': tp,
        "AUC": round(roc_auc, 3),
        "AUC95%CI": (round(delong_ci[0], 3), round(delong_ci[1], 3)),
        "Accuracy": round(accuracy, 3),
        "Sensitivity": round(sensitivity, 3),
        "Specificity": round(specificity, 3),
        "Precision": round(precision, 3),
        "F1": round(f1, 3),
        "PPV": round(precision, 3),
        "NPV": round(npv, 3),
        "Kappa": round(kappa, 3)
    }


def Calibration(proba_test, proba_train, y_true):
    """
    使用Youden’s J index校准
    :param probabilities: 概率
    :param y_true:
    :return: 预测值
    """
    # 定义阈值范围
    thresholds = np.arange(0, 1, 0.01)
    best_threshold = 0
    best_j = -1
    # 遍历阈值，计算每个阈值的 J 值
    for threshold in thresholds:
        # 将概率转换为二进制预测
        y_pred = (proba_train >= threshold).astype(int)
        # 计算混淆矩阵
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        tn, fp, fn, tp = cm.ravel()
        # 计算 sensitivity 和 specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        # 计算 J 值
        j = sensitivity + specificity - 1
        # 更新最佳阈值
        if j > best_j:
            best_j = j
            best_threshold = threshold
    print(f"最佳阈值: {best_threshold}")
    print(f"最大 J 值: {best_j}")
    return (proba_test >= best_threshold).astype(int)


def draw_confusion(results_, y_2, mode, output_file):
    # 先计算所有混淆矩阵
    cms_and_names = []
    for name, res in results_.items():
        if mode == 1:
            cm = confusion_matrix(y_2, res['y_final_pred'])
        else:
            cm = confusion_matrix(y_2, res['y_pred'])
        cms_and_names.append((cm, name))
        # df = pd.concat([df, pd.DataFrame([classification_metrics(y_2,
        #                                                          res['y_final_pred'],
        #                                                          res['y_proba'],
        #                                                          name)])], ignore_index=True)

    # 布局：12个子图，3行4列
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    # fig.suptitle('Confusion Matrices for All Models', fontsize=20)
    # 统一颜色范围
    vmax = max(cm.max() for cm, _ in cms_and_names)
    # 类别标签
    class_labels = ['0', '1']

    for i, (cm, name) in enumerate(cms_and_names):
        ax = axes.flat[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=vmax)

        ax.set_title(name, fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels, fontsize=12)
        ax.set_yticklabels(class_labels, fontsize=12)
        # 添加坐标轴标题
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        # 在每个格子中显示数量
        for x_ in range(cm.shape[0]):
            for y_ in range(cm.shape[1]):
                ax.text(y_, x_, str(cm[x_, y_]), ha='center', va='center',
                        color='black', fontsize=30)
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    # 添加统一的 colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=15)

    os.makedirs(output_file, exist_ok=True)  # 若目录不存在则创建，已存在则不报错
    if mode == 1:
        plt.savefig(output_file + 'celi_confusion_matrices.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file + 'all_confusion_matrices.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def draw_ROC(results_, output_file):
    roc_curves = []
    for name, res in results_.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
        auc_val = auc(fpr, tpr)
        # 计算AUC的95% CI（这里使用简单的正态近似，实际项目中可替换为更精确的方法）
        auc_ci = delong_auc_ci(y_true=np.array(res['y_true']), y_score=np.array(res['y_proba']))

        roc_curves.append({
            'name': name,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc_val,
            'auc_ci': auc_ci
        })

    # 按AUC值降序排序
    roc_curves_sorted = sorted(roc_curves, key=lambda x: x['auc'], reverse=True)

    # 绘制ROC曲线
    fig, ax = plt.subplots(figsize=(10, 8))
    # plt.figure(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(roc_curves_sorted)))  # 使用20种颜色
    linestyles = ['-', '--', '-.', ':']  # 使用4种线型
    for idx, curve in enumerate(roc_curves_sorted):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        plt.plot(curve['fpr'], curve['tpr'],
                 label=f"{curve['name']} (AUC = {curve['auc']:.3f} [{curve['auc_ci'][0]:.3f}~{curve['auc_ci'][1]:.3f}])",
                 color=color, linestyle=linestyle)

    # 添加基准线
    plt.plot([0, 1], [0, 1], 'k--')

    # 设置图表属性
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Sorted by AUC with 95% CI')
    plt.legend(loc='lower right', fontsize='small')  # 调整图例字体大小
    plt.grid(True)

    os.makedirs(output_file, exist_ok=True)  # 若目录不存在则创建，已存在则不报错
    plt.savefig(output_file + 'single_ROC.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    # return fig


def delong_test(y_true_, y_score1, y_score2):
    """
    执行DeLong检验，比较两个二分类模型的AUC差异是否显著

    参数:
    y_true: 数组，形状为(n_samples,)，真实标签（0或1）
    y_score1: 数组，形状为(n_samples,)，第一个模型的预测分数（正类概率）
    y_score2: 数组，形状为(n_samples,)，第二个模型的预测分数（正类概率）

    返回:
    result: 字典，包含AUC1、AUC2、AUC差异、Z统计量和p值
    """
    ci1 = delong_auc_ci(y_true_, y_score1)  # 计算置信区间
    ci2 = delong_auc_ci(y_true_, y_score2)

    # 将输入转换为numpy数组
    y_true_ = np.asarray(y_true_)
    y_score1 = np.asarray(y_score1)
    y_score2 = np.asarray(y_score2)

    # 分离正样本和负样本的预测分数（假设正类标签为1）
    pos_label = 1
    y_pos1 = y_score1[y_true_ == pos_label]  # 正样本在模型1的分数
    y_neg1 = y_score1[y_true_ != pos_label]  # 负样本在模型1的分数
    y_pos2 = y_score2[y_true_ == pos_label]  # 正样本在模型2的分数
    y_neg2 = y_score2[y_true_ != pos_label]  # 负样本在模型2的分数

    m = len(y_pos1)  # 正样本数量
    n = len(y_neg1)  # 负样本数量

    if m == 0 or n == 0:
        raise ValueError("真实标签中必须同时包含正样本和负样本")

    # 计算正样本对AUC的贡献（U统计量）
    def compute_U(y_pos, y_neg):
        """计算每个正样本对AUC的贡献"""
        U = []
        for pos_score in y_pos:
            # 比较正样本分数与所有负样本分数：>记1，=记0.5，<记0
            u = np.mean((pos_score > y_neg) + 0.5 * (pos_score == y_neg))
            U.append(u)
        return np.array(U)

    # 计算负样本对AUC的贡献（V统计量）
    def compute_V(y_pos, y_neg):
        """计算每个负样本对AUC的贡献"""
        V = []
        for neg_score in y_neg:
            # 比较所有正样本分数与负样本分数：>记1，=记0.5，<记0
            v = np.mean((y_pos > neg_score) + 0.5 * (y_pos == neg_score))
            V.append(v)
        return np.array(V)

    # 计算两个模型的U和V
    U1, U2 = compute_U(y_pos1, y_neg1), compute_U(y_pos2, y_neg2)
    V1, V2 = compute_V(y_pos1, y_neg1), compute_V(y_pos2, y_neg2)

    # 计算AUC
    auc1 = np.mean(U1)
    auc2 = np.mean(U2)
    auc_diff = auc1 - auc2

    # 计算AUC的方差和协方差
    # 模型1的AUC方差
    var_U1 = np.var(U1, ddof=1)  # 样本方差（自由度n-1）
    var_V1 = np.var(V1, ddof=1)
    var_auc1 = (var_U1 / m) + (var_V1 / n)

    # 模型2的AUC方差
    var_U2 = np.var(U2, ddof=1)
    var_V2 = np.var(V2, ddof=1)
    var_auc2 = (var_U2 / m) + (var_V2 / n)

    # 两个AUC的协方差
    cov_U = np.cov(U1, U2, ddof=1)[0, 1]  # U1和U2的协方差
    cov_V = np.cov(V1, V2, ddof=1)[0, 1]  # V1和V2的协方差
    cov_auc = (cov_U / m) + (cov_V / n)

    # 计算AUC差异的方差
    var_diff = var_auc1 + var_auc2 - 2 * cov_auc

    # 计算Z统计量（若方差为0，Z设为0）
    if var_diff == 0:
        z_stat = 0.0
    else:
        z_stat = auc_diff / np.sqrt(var_diff)

    # 计算双边检验的p值
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return {
        "AUC1": str(round(auc1, 3))+str(ci1),
        "AUC2": str(round(auc2, 3))+str(ci2),
        "AUC_diff": round(auc_diff, 3),
        "Z_statistic": round(z_stat, 3),
        "p_value": p_value
    }


def base_ml_train(classifiers, project_path, input_path, output_file, K=5):
    global target, fold_results_dir
    # 读取数据
    data = pd.read_csv(project_path + input_path)

    print(data.describe())

    X_ = data.drop([target], axis=1).values

    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X_)
    # X = X_  #若不进行标准化，则使用这一行

    y = data[target]

    # 保存scaler用于后续预测
    with open(os.path.join(project_path, 'models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # ----------------------------1. 贝叶斯超参数寻优（结合5折交叉验证）--------------------------
    # -----------------------------初始化分层 K 折交叉验证-------------------------
    results = {}
    df_single = pd.DataFrame(
        columns=['Algorithm', "tn", "fp", "fn", 'tp', "AUC", "AUC95%CI", "Accuracy", "Sensitivity", "Specificity",
                 "Precision", "F1", "PPV", "NPV", "Kappa"])
    df_single_cali, df_single_whole, df_single_whole_cali = df_single.copy(), df_single.copy(), df_single.copy()
    df_params = pd.DataFrame(columns=['Algorithm', "fold", "best_params"])
    Name = ''
    for name, clf in classifiers:
        Name = name
        y_pred_list = []
        y_proba_list = []
        y_true_list = []
        y_final_pred_list = []
        all_shap_values = []
        params = {}
        kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
        flag = 1
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # 数据平衡处理
            smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

            # 贝叶斯超参数搜索
            bayes_search = BayesSearchCV(
                estimator=clf,
                search_spaces=search_space(name=name),
                n_iter=10,
                scoring='roc_auc',
                random_state=42,
                n_jobs=-1
            )
            print(f"\n{name}第{fold_idx}/{K}折 - 开始贝叶斯超参数寻优...")
            bayes_search.fit(X_resampled, y_resampled)

            # 训练最佳模型
            clf.set_params(**bayes_search.best_params_)
            clf.fit(X_resampled, y_resampled)
            # 保存模型
            if not os.path.exists(project_path + 'models'):
                os.makedirs(project_path + 'models')
            with open(project_path + f'models/{name}_fold_{fold_idx}.pkl', 'wb') as file:
                pickle.dump(clf, file)
            print(f"模型已保存到{file}")

            # 预测结果
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]  # 验证集概率
            y_proba2 = clf.predict_proba(X_resampled)[:, 1]  # 训练集概率
            y_final_pred = Calibration(proba_test=y_proba, proba_train=y_proba2, y_true=y_resampled,)  # 对结果进行校准
            y_true = y_test.values  # 真实标签

            # 新增：保存当前折的结果
            fold_data = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'y_final_pred': y_final_pred,
                'test_id': test_idx,
            })
            # 文件名格式：模型名_折数.csv
            fold_filename = f"{name.replace(' ', '_')}_fold_{fold_idx}.csv"
            fold_data.to_csv(os.path.join(fold_results_dir, fold_filename), index=False)
            print(f"已保存{name}第{fold_idx}折结果至：{fold_filename}")

            df_single = pd.concat([df_single,
                                   pd.DataFrame([classification_metrics(y_true_=y_test.values,
                                                                        y_pred_=clf.predict(X_test),
                                                                        y_proba_=y_proba,
                                                                        name_=name + f'_{fold_idx}-fold')])],
                                  ignore_index=True)

            df_single_cali = pd.concat([df_single_cali,
                                        pd.DataFrame([classification_metrics(y_true_=y_test.values,
                                                                             y_pred_=y_final_pred,
                                                                             y_proba_=y_proba,
                                                                             name_=name + f'_{fold_idx}-fold_calibrated')])],
                                       ignore_index=True)

            # 累计结果用于整体评估
            y_pred_list.extend(y_pred)
            y_proba_list.extend(y_proba)
            y_true_list.extend(y_true)
            y_final_pred_list.extend(y_final_pred)

            params['fold'] = fold_idx
            params['Algorithm'] = name
            params['best_params'] = str(bayes_search.best_params_)
            df_params = pd.concat([df_params, pd.DataFrame(params, index=[0])], ignore_index=True)

            flag += 1
        df_single_whole_cali = pd.concat([df_single_whole_cali,
                                          pd.DataFrame([classification_metrics(y_true_=np.array(y_true_list),
                                                                               y_pred_=np.array(y_final_pred_list),
                                                                               y_proba_=np.array(y_proba_list),
                                                                               name_=name + '_calibrated')])],
                                         ignore_index=True)
        df_single_whole = pd.concat([df_single_whole,
                                     pd.DataFrame([classification_metrics(y_true_=np.array(y_true_list),
                                                                          y_pred_=np.array(y_pred_list),
                                                                          y_proba_=np.array(y_proba_list),
                                                                          name_=name)])],
                                    ignore_index=True)
        # 后续整体结果处理（保持不变）
        results[name] = {
            'y_pred': y_pred_list,
            'y_proba': y_proba_list,
            'y_final_pred': y_final_pred_list,
            'y_true': y_true_list,
        }

    print(results)

    draw_ROC(results_=results, output_file=project_path + "out/")

    pd.DataFrame(results).to_csv(output_file + f'Prediction_of_{len(classifiers)}_ML_models.csv')  # 保存输出
    df_single.to_csv(output_file + f'CV_results_of_{len(classifiers)}_ML_models_in_every_fold.csv',
                     index=False)  # 保存每折评估结果
    df_single_cali.to_csv(output_file + f'Calibrated_CV_results_of_{len(classifiers)}_ML_models_in_every_fold.csv',
                          index=False)  # 保存每折校准后的评估结果
    df_single_whole.to_csv(output_file + f'{K}-fold_CV_results_of_{len(classifiers)}_ML_models_on_whole_dataset.csv',
                           index=False)  # 保存总数据集的评估结果
    df_single_whole_cali.to_csv(
        output_file + f'Calibrated_{K}-fold_CV_results_of_{len(classifiers)}_ML_models_on_whole_dataset.csv',
        index=False)  # 保存校准后的总数据集的评估结果
    df_params.to_csv(output_file + 'params.csv')

    # 绘制混淆矩阵
    # df = draw_confusion(results_=results, y_2=y, df=df_single_whole_cali)
    draw_confusion(results_=results, y_2=results[Name]['y_true'], mode=1, output_file=project_path + 'out/')
    draw_confusion(results_=results, y_2=results[Name]['y_true'], mode=2, output_file=project_path + 'out/')

    return results

# '''
# if __name__ == "__main__":
#
#     project_path = 'MLB-Injuries-Analysis/'
#     input_path = "processed_data.csv"
#     target = 'Injured'
#     output_file = project_path + 'out/'
#     # 确保输出目录存在
#     os.makedirs(output_file, exist_ok=True)
#     # 新增：创建折数结果保存子目录
#     fold_results_dir = os.path.join(output_file, "fold_results/")
#     os.makedirs(fold_results_dir, exist_ok=True)
#
#     # 初始化分类器列表
#     classifiers = [
#         ('Logistic Regression', LogisticRegression()),
#         # ('SVM', SVC(probability=True)),
#         ('KNN', KNeighborsClassifier()),
#         ('Naive Bayes', GaussianNB()),
#         ('Decision Tree', DecisionTreeClassifier()),
#         ('Random Forest', RandomForestClassifier()),
#         ('GBDT', GradientBoostingClassifier()),
#         ('HGBDT', HistGradientBoostingClassifier()),
#         ('AdaBoost', AdaBoostClassifier()),
#         # ('CatBoost', CatBoostClassifier(logging_level='Silent')),  # 关键：禁用日志输出，不创建catboost_info目录
#         ('Extra Trees', ExtraTreesClassifier()),
#         ('LGBM', LGBMClassifier()),
#         ('Bagging', BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42)),
#         ("XGBoost", XGBClassifier(eval_metric='logloss', use_label_encoder=False))
#     ]
#

#     base_ml_train(classifiers, project_path, input_path, output_file, K=5)
