import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score
import scipy.stats as stats
from sklearn.ensemble import StackingClassifier, VotingClassifier
import pickle
from utils.SHAP_summary import advance_summary_plot
import shap
from tqdm import tqdm
from utils.DCA import bootstrap_dca, plot_dca_curves
from utils.calibration_curve import plot_paper_calibration
plt.style.use('ggplot')
# 设置中文字体
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


# 评估指标计算函数
def classification_metrics(y_true, y_pred, y_proba, name, full=False):
    """计算二分类模型的常用指标"""
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
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
    # Kappa系数
    kappa = cohen_kappa_score(y_true, y_pred)
    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    delong_ci = delong_auc_ci(y_true, y_proba)

    if full:
        return {
            "Algorithm": name,
            # "AUC": roc_auc,
            "AUC (95%CI)": str(round(roc_auc, 3))+f"({round(delong_ci[0], 3)},{round(delong_ci[1], 3)})",
            "Accuracy": round(accuracy, 3),
            "Sensitivity": round(sensitivity, 3),
            "Specificity": round(specificity, 3),
            "Precision": round(precision, 3),
            "F1": round(f1, 3),
            "PPV": round(precision, 3),
            "NPV": round(npv, 3),
            "Kappa": round(kappa, 3),
        }
    # 'Algorithm', "AUC (95%CI)", "Accuracy", "Sensitivity",
    #                                    "Specificity", "Precision", "F1", "PPV", "NPV", "Kappa"
    else:
        return {
            # "Algorithm": name,
            "AUC": roc_auc,
            # "AUC95%CI": f"({round(delong_ci[0], 3)},{round(delong_ci[1], 3)})",
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "F1": f1,
            "PPV": precision,
            "NPV": npv,
            "Kappa": kappa
        }


# 计算AUC的置信区间
def delong_auc_ci(y_true, y_score, alpha=0.05):
    """用DeLong方法计算AUC的(1-alpha)置信区间"""
    y1 = y_score[y_true == 1]  # 正类得分
    y0 = y_score[y_true == 0]  # 负类得分
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
    z = np.abs(np.percentile(np.random.normal(0, 1, 100000), 100 * (1 - alpha / 2)))
    lower = auc - z * se_auc
    upper = auc + z * se_auc
    return (max(0, lower), min(1, upper))


# DeLong检验，比较两个模型的AUC差异
def delong_test(y_true, y_score1, y_score2, model1_name, model2_name):
    """执行DeLong检验，比较两个模型的AUC差异"""
    # 分离正样本和负样本的预测分数
    pos_label = 1
    y_pos1 = y_score1[y_true == pos_label]
    y_neg1 = y_score1[y_true != pos_label]
    y_pos2 = y_score2[y_true == pos_label]
    y_neg2 = y_score2[y_true != pos_label]

    m = len(y_pos1)  # 正样本数量
    n = len(y_neg1)  # 负样本数量

    if m == 0 or n == 0:
        raise ValueError("真实标签中必须同时包含正样本和负样本")

    # 计算正样本对AUC的贡献
    def compute_U(y_pos, y_neg):
        U = []
        for pos_score in y_pos:
            u = np.mean((pos_score > y_neg) + 0.5 * (pos_score == y_neg))
            U.append(u)
        return np.array(U)

    # 计算负样本对AUC的贡献
    def compute_V(y_pos, y_neg):
        V = []
        for neg_score in y_neg:
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
    var_U1 = np.var(U1, ddof=1)
    var_V1 = np.var(V1, ddof=1)
    var_auc1 = (var_U1 / m) + (var_V1 / n)

    var_U2 = np.var(U2, ddof=1)
    var_V2 = np.var(V2, ddof=1)
    var_auc2 = (var_U2 / m) + (var_V2 / n)

    cov_U = np.cov(U1, U2, ddof=1)[0, 1]
    cov_V = np.cov(V1, V2, ddof=1)[0, 1]
    cov_auc = (cov_U / m) + (cov_V / n)

    # 计算AUC差异的方差
    var_diff = var_auc1 + var_auc2 - 2 * cov_auc

    # 计算Z统计量
    if var_diff == 0:
        z_stat = 0.0
    else:
        z_stat = auc_diff / np.sqrt(var_diff)

    # 计算双边检验的p值
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    return {
        "Model1": model1_name,
        "Model2": model2_name,
        "AUC1": round(auc1, 3),
        "AUC2": round(auc2, 3),
        "AUC_diff": round(auc_diff, 3),
        "Z_statistic": round(z_stat, 3),
        "p_value": round(p_value, 4),
        "p<0.05": "Yes" if p_value < 0.05 else "No",
        "p<0.01": "Yes" if p_value < 0.01 else "No",
    }


# # 读取fold_results中的结果
# def load_fold_results(results_dir):
#     """加载所有折的结果并按模型组织"""
#     model_results = {}
#
#     # 遍历所有结果文件
#     for filename in os.listdir(results_dir):
#         if filename.endswith(".csv") and "_fold_" in filename:
#             # 解析文件名获取模型名和折数
#             model_name = filename.split("_fold_")[0].replace("_", " ")
#             fold = int(filename.split("_fold_")[1].split(".")[0])
#
#             # 读取文件
#             file_path = os.path.join(results_dir, filename)
#             df = pd.read_csv(file_path)
#
#             # 按模型和折数存储
#             if model_name not in model_results:
#                 model_results[model_name] = {}
#             model_results[model_name][fold] = df
#
#     return model_results


def load_fold_results(results_dir):
    """加载所有折的结果并按模型组织"""
    model_results = {}

    # 遍历所有结果文件
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv") and "_fold_" in filename:
            # 解析文件名获取模型名和折数
            model_name = filename.split("_fold_")[0].replace("_", " ")
            fold = int(filename.split("_fold_")[1].split(".")[0])

            # 读取文件
            file_path = os.path.join(results_dir, filename)
            df = pd.read_csv(file_path)

            # 按模型和折数存储
            if model_name not in model_results:
                model_results[model_name] = {}
            model_results[model_name][fold] = df

    model_metrics = {}
    for model_name, fold_data in model_results.items():
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        all_y_final_pred = []
        X_id = []
        for fold, df in fold_data.items():
            all_y_true.extend(df['y_true'].values)
            all_y_pred.extend(df['y_pred'].values)
            all_y_proba.extend(df['y_proba'].values)
            all_y_final_pred.extend(df['y_final_pred'].values)
            X_id.extend(df['test_id'].values)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)
        all_y_final_pred = np.array(all_y_final_pred)
        model_metrics[model_name] = {
            'y_true': all_y_true,
            'y_proba': all_y_proba,
            'y_pred': all_y_pred,
            'y_final_pred': all_y_final_pred,
            'X_id': X_id,
            # 'auc_ci': auc_ci
        }
    return model_metrics, model_results


# 汇总模型的所有折结果并计算性能指标
def evaluate_models(model_results):
    """评估每个模型的整体性能"""
    model_metrics = {}

    for model_name, fold_data in model_results.items():
        # 汇总所有折的数据
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        all_y_final_pred = []

        for fold, df in fold_data.items():
            all_y_true.extend(df['y_true'].values)
            all_y_pred.extend(df['y_pred'].values)
            all_y_proba.extend(df['y_proba'].values)
            all_y_final_pred.extend(df['y_final_pred'].values)

        # 转换为numpy数组
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)
        all_y_final_pred = np.array(all_y_final_pred)

        # 计算指标
        metrics = classification_metrics(all_y_true, all_y_final_pred, all_y_proba, model_name)
        metrics2 = classification_metrics(all_y_true, all_y_pred, all_y_proba, model_name)
        # 计算AUC置信区间
        auc_ci = delong_auc_ci(all_y_true, all_y_proba)

        model_metrics[model_name] = {
            'metrics': metrics,
            'metrics2': metrics2,
            'y_true': all_y_true,
            'y_proba': all_y_proba,
            'y_pred': all_y_pred,
            'y_final_pred': all_y_final_pred,
            'auc_ci': auc_ci
        }

    return model_metrics


# 筛选出所有指标均大于0.7的模型
def filter_models(model_metrics, threshold=0.7, mode=1):
    """筛选出所有评估指标均大于阈值的模型"""
    filtered_models = {}

    for model_name, data in model_metrics.items():
        if mode == 1:
            metrics = data['metrics']
        else:
            metrics = data['metrics2']
        # 检查所有指标是否都大于阈值
        all_above = all(value >= threshold for value in metrics.values())

        if all_above:
            filtered_models[model_name] = data

    return filtered_models


def filter_models_top_5(model_metrics, mode=1):
    """
    筛选出AUC排名前五的模型（按AUC降序排列）

    参数：
        model_metrics (dict): 模型评估指标字典，结构为{模型名: {'metrics/metrics2': {'AUC': 数值, ...}, ...}}
        mode (int): 1-使用metrics中的AUC，2-使用metrics2中的AUC

    返回：
        dict: 包含AUC排名前五的模型信息的字典（按AUC降序排列）
    """
    # 存储模型名和对应的AUC值
    model_auc = {}

    for model_name, data in model_metrics.items():
        # 根据mode选择对应的指标集
        metrics_key = 'metrics' if mode == 1 else 'metrics2'
        if metrics_key not in data:
            print(f"警告：模型{model_name}缺少{metrics_key}指标集，跳过")
            continue

        metrics = data[metrics_key]
        # 提取AUC值（需确保指标键为'AUC'，若实际键名不同可修改此处）
        if 'AUC' not in metrics:
            print(f"警告：模型{model_name}的{metrics_key}中无AUC指标，跳过")
            continue

        auc_value = metrics['AUC']
        # 校验AUC值的合法性（0-1之间）
        if not isinstance(auc_value, (int, float)) or auc_value < 0 or auc_value > 1:
            print(f"警告：模型{model_name}的AUC值{auc_value}不合法，跳过")
            continue

        model_auc[model_name] = auc_value

    # 按AUC降序排序模型（若AUC相同，按模型名升序）
    sorted_models = sorted(
        model_auc.items(),
        key=lambda x: (-x[1], x[0])  # 负号表示降序，x[0]是模型名用于同分排序
    )

    # 取前五名，构建最终结果字典
    top5_models = {
        model_name: model_metrics[model_name]
        for model_name, _ in sorted_models[:5]
    }

    return top5_models


def Calibration(probabilities, y_true):
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
        y_pred = (probabilities >= threshold).astype(int)
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
    return (probabilities >= best_threshold).astype(int)


# 生成模型的所有可能组合并评估
def evaluate_model_combinations(filtered_models, min_combination_size=2):
    """评估所有可能的模型组合"""
    model_names = list(filtered_models.keys())
    combinations_metrics = []
    df_eva = pd.DataFrame(columns=['Algorithm', "AUC (95%CI)", "Accuracy", "Sensitivity",
                                   "Specificity", "Precision", "F1", "PPV", "NPV", "Kappa"])
    # 生成所有可能的组合（从min_combination_size到模型总数）
    for k in range(min_combination_size, len(model_names) + 1):
        for combo in combinations(model_names, k):
            # 组合名称
            combo_name = " + ".join(combo)

            # 获取组合中所有模型的预测概率并平均
            y_proba_list = [filtered_models[model]['y_proba'] for model in combo]
            combined_proba = np.mean(y_proba_list, axis=0)

            # 获取真实标签（所有模型的真实标签应该是相同的）
            y_true = filtered_models[model_names[0]]['y_true']

            # # # 使用Youden's J指数确定最佳阈值
            best_pred = Calibration(probabilities=combined_proba, y_true=y_true)  # 对结果进行校准
            # threshold = 0.5  # 默认阈值
            # best_pred = (combined_proba >= threshold).astype(int)  # 概率≥阈值则为1，否则为0

            # 计算组合模型的指标
            metrics = classification_metrics(y_true, best_pred, combined_proba, combo_name, full=True)
            metrics2 = classification_metrics(y_true, best_pred, combined_proba, combo_name)
            auc_ci = delong_auc_ci(y_true, combined_proba)

            combinations_metrics.append({
                'name': combo_name,
                'model_count': k,
                'metrics': metrics2,
                'y_proba': combined_proba,
                'y_pred': best_pred,
                'y_true': y_true,
                'auc_ci': auc_ci,
                'models': combo
            })

            Metri = pd.DataFrame([list(metrics.values())], columns=metrics.keys())
            df_eva = pd.concat([df_eva, Metri], ignore_index=True)

    df_eva.to_csv(project_path+'out/all_combo_metrics.csv')
    # 按AUC排序
    combinations_metrics.sort(key=lambda x: x['metrics']['AUC'], reverse=True)
    return combinations_metrics


# 执行所有模型间的DeLong检验
def perform_delong_tests(single_models, combo_models, output_file):
    """对所有单个模型和组合模型进行两两DeLong检验"""
    # 收集所有要比较的模型（单个模型+组合模型）
    all_models = []

    # 添加单个模型
    for name, data in single_models.items():
        all_models.append({
            'name': name,
            'y_proba': data['y_proba'],
            'y_true': data['y_true'],
            'type': 'single'
        })

    # 添加组合模型
    for combo in combo_models:
        all_models.append({
            'name': combo['name'],
            'y_proba': combo['y_proba'],
            'y_true': combo['y_true'],
            'type': 'combination'
        })

    # 执行所有两两比较
    delong_results = []
    for i in range(len(all_models)):
        for j in range(i + 1, len(all_models)):
            model1 = all_models[i]
            model2 = all_models[j]

            # 确保真实标签一致
            if not np.array_equal(model1['y_true'], model2['y_true']):
                print(f"警告: {model1['name']} 和 {model2['name']} 的真实标签不一致，跳过DeLong检验")
                continue

            # 执行DeLong检验
            result = delong_test(
                y_true=model1['y_true'],
                y_score1=model1['y_proba'],
                y_score2=model2['y_proba'],
                model1_name=model1['name'],
                model2_name=model2['name']
            )
            delong_results.append(result)

    # 保存结果
    df = pd.DataFrame(delong_results)
    df.to_csv(output_file, index=False)
    print(f"DeLong检验结果已保存至: {output_file}")
    return df


# 绘制ROC曲线（按AUC排序图例）
def plot_roc_curves(top_single_models, top_combos, output_file):
    """绘制前五名组合与前五名单个模型的ROC曲线，图例按AUC排序"""
    plt.figure(figsize=(10, 8))

    # 收集所有要绘制的模型数据
    plot_data = []

    # 添加单个模型
    for model_name, data in top_single_models.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
        roc_auc = data['metrics']['AUC']
        ci = data['auc_ci']
        plot_data.append({
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            # 'label': f'{model_name} (AUC = {roc_auc:.3f}, 95%CI: [{ci[0]:.3f}, {ci[1]:.3f}])',
            'label': f'{model_name} (AUC={roc_auc:.3f}[{ci[0]:.3f}~{ci[1]:.3f}])',
            'linestyle': '-'
        })

    # 添加组合模型
    for combo in top_combos:
        fpr, tpr, _ = roc_curve(combo['y_true'], combo['y_proba'])
        roc_auc = combo['metrics']['AUC']
        ci = combo['auc_ci']
        plot_data.append({
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            # 'label': f'组合 {combo["name"]} (AUC = {roc_auc:.3f}, 95%CI: [{ci[0]:.3f}, {ci[1]:.3f}])',
            'label': f'{combo["name"]} (AUC={roc_auc:.3f}[{ci[0]:.3f}~{ci[1]:.3f}])',
            'linestyle': '--'
        })

    # 按AUC降序排序
    plot_data.sort(key=lambda x: x['auc'], reverse=True)
    print('全场AUC最高模型：', plot_data[0]['label'])
    whole_best = plot_data[0]['label'].split(' (')[0]
    # 绘制所有曲线
    for item in plot_data:
        plt.plot(item['fpr'], item['tpr'], lw=2, linestyle=item['linestyle'], label=item['label'])

    # 绘制随机猜测的基准线
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle=':')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of ROC curves between individual and combined models')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"ROC曲线已保存至: {output_file}")
    plt.show()
    return whole_best


def Explain(model_results, sample_id, model_metr, target, names, project_path):
    df = pd.read_csv(project_path + 'processed_data.csv')
    print(df.head())  # 看表头
    print('缺失值统计：', df.isnull().sum())  # 查看有无缺失值
    # 生成特征变量数据集和因变量数据集
    X = df.drop(target, axis=1)
    y = df[target]

    base_list = names.split(' + ')

    # y_proba_list = [model_metrics[clf]['y_proba'] for clf in base_list]
    y_true_list = [model_metr[clf]['y_true'] for clf in base_list]
    X_id = [model_metr[clf]['X_id'] for clf in base_list][0]
    print('len(X_id)', len(X_id))

    pkl_file = project_path + 'models/'
    # df_esti = pd.DataFrame(columns=['fold', 'name', 'model', 'data_id'])

    Estimators = []
    print('model_results.items()', model_results.items())
    for model_name, fold_data in model_results.items():
        if model_name in base_list:
            print(model_name)
            print('fold_data',fold_data)
            for fold, df_ in fold_data.items():
                print('fold', fold)
                estimator = {}
                with open(pkl_file + model_name + '_fold_' + str(fold) + '.pkl', 'rb') as file:
                    loaded_model = pickle.load(file)
                    print('loaded_model', loaded_model)

                    estimator['fold'] = fold
                    estimator['name'] = model_name
                    estimator['model'] = loaded_model
                    estimator['data_id'] = df_['test_id'].values
                    print('estimator', estimator)
                    print(f"第{fold}折{estimator}模型已加载")
                Estimators.append(estimator)
    print('Estimators', Estimators)
    merged_shap = np.empty((0, len(X.columns)))
    Expected = 0
    y_pred_list = []
    y_positive_proba = []
    y_list = []
    for i in [1, 2, 3, 4, 5]:
        estimators = []
        id_ = []
        for est in Estimators:
            if est['fold'] == i:
                estimators.append((est['name'], est['model']))
                if est['name'] == base_list[0]:
                    # id_.append(est['data_id'])
                    id_ = est['data_id']
        print('estimators', estimators)
        # 软投票集成
        voting_soft = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=None,
        )
        print(len(id_), id_)
        voting_soft.fit(X.iloc[id_], y[id_])

        y_list.extend(y[id_])
        y_positive_proba.extend(voting_soft.predict_proba(X.iloc[id_])[:, 1])  # 阳性事件概率
        y_pred_list.extend(voting_soft.predict(X.iloc[id_]))

        explainer = shap.PermutationExplainer(voting_soft.predict, X.iloc[id_], npermutations=3)    # 排列次数从默认10→3，提速3倍+
        shap_values = explainer(X.iloc[id_])
        expected = shap_values.base_values[0] * len(id_) / X.shape[
            0]  # 获取基准值（原expected_value），base_values是数组，所有样本的基准值相同，取第一个即可
        Expected += expected
        # print('explainer.expected_value:', explainer.expected_value)
        merged_shap = np.concatenate([merged_shap, shap_values.values], axis=0)
        # print('shap_values.shape, merged_shap.shape', shap_values.shape, merged_shap.shape)

    # --------------- 绘制论文级校准曲线 ---------------
    os.makedirs(project_path+'plots/', exist_ok=True)
    plot_paper_calibration(
            y_true=y_list,
            y_proba=y_positive_proba,
            model_name=names,
            n_bins=10,  # 10组分组（论文常用）
            n_boot=1000,  # 1000次Bootstrapping
            plot_youden=True,  # 标注Youden J阈值（可选关闭）
            save_path=project_path + "plots/calibration_curve.jpg"  # 保存为PDF（矢量图）
        )
    # --------------- Bootstrapping计算DCA ---------------
    print('阳性样本数量：', np.sum(np.array(y_list) == 1))
    # Bootstrap计算DCA数据
    thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none = bootstrap_dca(np.array(y_list), np.array(y_positive_proba), n_boot=1000)
    print(f"thresholds 维度：{thresholds.ndim}")
    # 绘制三条核心曲线
    plot_dca_curves(thresholds, mean_nb, lb_nb, ub_nb, nb_all, nb_none, save_path=project_path+"plots/dca_bootstrap.jpg")

    advance_summary_plot(merged_shap, X.iloc[X_id], show=False)
    plt.tight_layout()
    plt.savefig(project_path + 'plots/summary.jpg', dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close()

    # shap.summary_plot(merged_shap, X.iloc[X_id], plot_type="bar", show=False)
    # plt.title('SHAP feature importance')
    # plt.tight_layout()
    # plt.savefig(project_path + 'plots/bar.jpg', dpi=300, bbox_inches='tight')
    # plt.close()
    #
    # shap.summary_plot(merged_shap, X.iloc[X_id], show=False)
    # plt.title('SHAP bees warm plot')
    # plt.tight_layout()
    # plt.savefig(project_path + 'plots/summary.jpg', dpi=300, bbox_inches='tight')
    # plt.close()

    # shap.initjs()
    # shap.save_html(project_path + 'plots/force_plot.html',
    #                shap.force_plot(Expected, merged_shap, X.iloc[X_id]))

    # 绘制shap决策图
    error_indices = np.where(y_pred_list != y_list)[0]
    print('error_indices', error_indices)
    # # 多样本可视化探索
    shap.decision_plot(Expected, merged_shap, X.iloc[X_id],
                       highlight=error_indices,  # 突出显示错误样本（核心参数）
                       show=False,
                       ignore_warnings=True)
    plt.title('SHAP decision plot')
    plt.tight_layout()
    plt.savefig(project_path + 'plots/decision.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    # Deci_plot(y_pred=y_pred_list, y_true=y_true_list, sample_index=3)  # 绘制shap决策图

    # 指定样本绘制瀑布图
    # sample_index = 19
    shap.plots.waterfall(shap_values=explainer(X.iloc[X_id])[sample_id], show=False)
    plt.title(f'SHAP waterfall plot (sample ID = {sample_id})')
    plt.tight_layout()
    plt.savefig(project_path + f'plots/waterfall_{sample_id}.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    # 使用heatmap绘制SHAP值
    shap_explainer = shap.Explanation(
        values=merged_shap,  # 你的 SHAP 值数组
        base_values=Expected,  # 背景期望
        data=X.iloc[id_].values,  # 对应的特征数据
        feature_names=X.columns.tolist()
    )
    shap.plots.heatmap(shap_explainer, show=False)
    plt.title('SHAP heatmap')
    plt.tight_layout()
    plt.savefig(project_path + 'plots/heatmap.jpg', dpi=300, bbox_inches='tight')
    plt.close()


# 主函数
def main():
    # 1. 加载折结果
    print("加载折结果中...")
    # model_results = load_fold_results(results_dir=project_path+"out/fold_results/")
    model_metr, model_results = load_fold_results(results_dir=project_path + "out/fold_results/")
    # Explain(model_results=model_results, model_metr=model_metr, target=target, names='HGBDT + XGBoost')

    # 2. 评估每个模型
    print("评估单个模型性能...")
    model_metrics = evaluate_models(model_results)

    # 3. 筛选符合条件的模型
    print("筛选性能优良的模型...")
    filtered_models = filter_models_top_5(model_metrics, mode=2)
    print(f"找到排名前5的单模型: {', '.join(filtered_models.keys())}")

    # filtered_models = filter_models(model_metrics, threshold=0.7, mode=2)
    #
    # # if not filtered_models:
    # #     print("没有找到所有指标均大于0.7的模型，无法进行组合")
    # #     return
    # if not filtered_models:
    #     print("没有找到所有指标均大于0.7的模型，选择AUC排名前五的模型")
    #     filtered_models = filter_models_top_5(model_metrics, mode=2)
    # else:
    #     print(f"找到{len(filtered_models)}个符合条件的模型: {', '.join(filtered_models.keys())}")

    # 4. 评估模型组合
    print("评估模型组合...")
    combo_metrics = evaluate_model_combinations(filtered_models)

    if not combo_metrics:
        print("无法生成有效的模型组合")
        return
    else:
        pd.DataFrame(combo_metrics).to_csv(project_path+'out/combo_results_and_metrics.csv')
    # 5. 选择前五名的组合和单个模型
    top_n = 5
    # best_id = 1
    top_combos = combo_metrics[:top_n]
    # bast_combos = combo_metrics[:best_id][0]['name']
    # print('最佳组合', bast_combos)

    # 按AUC排序单个模型
    sorted_single_models = sorted(filtered_models.items(),
                                  key=lambda x: x[1]['metrics']['AUC'],
                                  reverse=True)
    top_single_models = dict(sorted_single_models[:top_n])

    # 6. 输出结果
    print("\n前五名单个模型:")
    for i, (name, data) in enumerate(top_single_models.items(), 1):
        print(f"{i}. {name} - AUC: {data['metrics']['AUC']:.3f}")

    print("\n前五名模型组合:")
    for i, combo in enumerate(top_combos, 1):
        print(f"{i}. {combo['name']} - AUC: {combo['metrics']['AUC']:.3f}")

    # 7. 执行DeLong检验
    print("\n执行DeLong检验...")
    perform_delong_tests(top_single_models, top_combos, project_path+"out/delong_test_results.csv")

    # 8. 绘制ROC曲线（按AUC排序图例）
    whole_best = plot_roc_curves(top_single_models, top_combos, project_path+"out/combination_vs_single_roc.pdf")

    # 进行模型解释
    Explain(model_results=model_results, model_metr=model_metr, target=target, names=whole_best)


if __name__ == "__main__":
    project_path = 'Multimodal Sports Injury Prediction Dataset/'
    target = 'injury_risk'
    if not os.path.exists(project_path+'plots'):
        os.makedirs(project_path+'plots')
    main()



