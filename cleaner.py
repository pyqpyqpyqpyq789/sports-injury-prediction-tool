import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 关键！！！即便没有调用也不能注释
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegressionCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.pipeline import Pipeline  # 支持重采样+模型的管道
from imblearn.combine import SMOTETomek
import pandas as pd

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('max_colwidth', 100)  # 设置value的显示长度
pd.set_option('display.width', 1000)  # 设置1000列时才换行


def MICE_Impute(df_1, target):
    # 将字符串型分类特征进行编码
    label_encoders = {}
    for col in df_1.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_1[col] = le.fit_transform(df_1[col])
        label_encoders[col] = le

    # 初始化MICE插补器
    mice_imputer = IterativeImputer(
        initial_strategy='mean',  # 用均值初始填充
        max_iter=10,
        random_state=42,
        estimator=None,  # 使用默认估计器（BayesianRidge，贝叶斯岭回归）
    )

    # 拟合和转换数据集以填补缺失值
    df_imp = mice_imputer.fit_transform(df_1)
    # 将拟合数据转换回pandas DataFrame
    df_imp = pd.DataFrame(df_imp, columns=df_1.columns)
    return df_imp


def epv_estimation(data, target_var):
    """基于EPV法估算样本量（适用于二分类因变量）"""
    # 检查因变量是否为二分类
    if not np.all(np.isin(data[target_var].dropna(), [0, 1])):
        raise ValueError("EPV法要求因变量为二分类（0/1）")

    # 自变量数量（排除因变量）
    n_features = data.shape[1] - 1
    if n_features == 0:
        raise ValueError("数据中没有自变量")

    # 事件数（因变量=1的样本）
    events = data[target_var].sum()
    total_samples = data.shape[0]
    event_rate = events / total_samples if total_samples > 0 else 0

    # 当前EPV
    current_epv = events / n_features if n_features > 0 else np.nan

    # 目标EPV=10时的样本量估算
    target_epv = 10
    required_events = target_epv * n_features
    required_samples = required_events / event_rate if event_rate > 0 else np.inf

    return {
        "自变量数量": n_features,
        "当前样本量": total_samples,
        "事件数": int(events),
        "事件发生率": round(event_rate, 3),
        "当前EPV": round(current_epv, 2) if not np.isnan(current_epv) else None,
        f"目标EPV({target_epv})所需事件数": required_events,
        f"目标EPV({target_epv})所需总样本量": round(required_samples) if event_rate > 0 else None
    }


def Variable_Selection(data, target):  # 新增target参数
    X_df = data.drop(target, axis=1)
    y = data[target]
    # 特征标准化（正则化必须做！）
    scaler = StandardScaler()  # 标准化（均值0，方差1）
    X_scaled = scaler.fit_transform(X_df)  # 对特征缩放

    # ElasticNetCV：结合L1和L2正则化，适合特征高相关的场景
    smotetomek_pipeline = Pipeline([
        ('smotetomek', SMOTETomek(random_state=42)),  # 过采样少数类，平衡样本
        ('clf', LogisticRegressionCV(
            cv=5,
            Cs=np.logspace(-4, 1, 20),
            l1_ratios=[0.5],
            penalty='elasticnet',
            solver='saga',
            class_weight=None,  # 已通过SMOTE平衡，无需额外加权
            random_state=42,
            max_iter=1000
        ))
    ])

    smotetomek_pipeline.fit(X_scaled, y)

    # 提取弹性网络选中的特征
    en_coef = smotetomek_pipeline[1].coef_[0]
    print('en_coef:', en_coef)
    en_exclude = [X_df.columns[i] for i in range(len(en_coef)) if np.abs(en_coef[i]) < 1e-5]
    print(f"\n平衡数据后弹性网络剔除的特征：{en_exclude}")
    return data.drop(en_exclude, axis=1)


# 主程序
if __name__ == "__main__":
    input_file = 'Multimodal Sports Injury Prediction Dataset/'
    raw_csv = 'sports_multimodal_data.csv'
    output_file = ''
    target = 'injury_risk'  # 定义target变量
    Irrelevant = []  # 需要直接剔除的明显无关特征

    # 读取数据
    df = pd.read_csv(input_file + raw_csv).dropna(subset=[target])  # 剔除目标缺失事件

    # 剔除明显无关特征
    df = df.drop(Irrelevant, axis=1)

    # 多重插补
    df_imputed = MICE_Impute(df_1=df, target=target)
    print(df_imputed.head())

    # 移除高相关性阈值筛选步骤

    # 移除VIF阈值筛选步骤

    # 样本量估算
    rr = epv_estimation(data=df_imputed, target_var=target)
    print('原始数据集样本量估算', rr)

    if rr['当前样本量'] <= rr['目标EPV(10)所需总样本量']:
        # 变量选择（传入target参数）
        final_data = Variable_Selection(df_imputed, target)
        # EPV样本量估算
        print('特征选择之后', epv_estimation(data=final_data, target_var=target))
    else:
        final_data = df_imputed

    # 保存最终数据
    final_data.to_csv(input_file + "processed_data.csv", index=False)
    print(f"\n处理完成，最终数据已保存至'{output_file}processed_data.csv'")