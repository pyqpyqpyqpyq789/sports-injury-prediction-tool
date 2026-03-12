import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shutil
import tempfile

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 设置页面配置
st.set_page_config(
    page_title="运动损伤机器学习分析平台",
    page_icon="📊",
    layout="wide"
)

# 确保中文显示正常
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 临时目录设置
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.processed_data = None
    st.session_state.models_trained = False
    st.session_state.voting_done = False

# # 临时目录用于存储中间结果
# if not os.path.exists("temp"):
#     os.makedirs("temp")
# if not os.path.exists("temp/plots"):
#     os.makedirs("temp/plots")
# if not os.path.exists("temp/models"):
#     os.makedirs("temp/models")
# if not os.path.exists("temp/fold_results"):
#     os.makedirs("temp/fold_results")
# if not os.path.exists("temp/results"):
#     os.makedirs("temp/results")

# 全局变量存储数据和结果（新增excluded_vars存储用户选择排除的变量）
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'voting_results' not in st.session_state:
    st.session_state.voting_results = None
if 'plots' not in st.session_state:
    st.session_state.plots = {}
if 'excluded_vars' not in st.session_state:  # 新增：存储用户选择排除的变量
    st.session_state.excluded_vars = []

# 侧边栏导航
st.sidebar.title("导航")
page = st.sidebar.radio("选择功能", ["项目介绍", "数据上传", "数据清洗", "模型训练", "集成学习", "结果下载"])

# ========== 新增：项目介绍页（含网络示意图） ==========
if page == "项目介绍":
    st.title("📊 运动损伤机器学习分析平台")
    st.markdown("#### @Developer：Yiqun Pang")
    st.divider()  # 分隔线，优化排版

    # 1. 项目概述
    st.subheader("一、项目概述")
    st.markdown("""
    本平台是一款基于Streamlit搭建的可视化机器学习分析工具，专为分类任务（二分类）设计， 
    若涉及多分类任务请事先将结局变量转换为哑变量并分批建模，
    提供从**数据上传→数据清洗→模型训练→集成学习→结果下载**的全流程自动化分析能力，
    无需编写复杂代码，通过可视化交互即可完成高质量的机器学习建模与评估。
    """)

    # 2. 核心功能
    st.subheader("二、核心功能")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 📤 **数据上传与配置**：支持CSV格式数据上传，自定义结局变量与排除变量
        - 🧹 **智能数据清洗**：内置MICE缺失值插补、EPV样本量估算、自动变量选择
        - 🤖 **基础模型训练**：支持多种经典分类模型，自定义K折交叉验证
        """)
    with col2:
        st.markdown("""
        - 📈 **模型评估可视化**：自动绘制ROC曲线、混淆矩阵、DCA曲线、校准曲线
        - 🔗 **集成学习优化**：基于投票机制提升模型性能，支持模型间AUC比较（DeLong检验）
        - 📥 **结果批量下载**：打包所有数据、模型、图表结果，方便后续整理与汇报
        """)

    # 3. 操作流程
    st.subheader("三、操作流程")
    st.markdown("""
    1.  **数据上传**：上传CSV文件，选择结局变量（二分类），勾选需要排除的无关变量
    2.  **数据清洗**：确认配置后，一键运行清洗流程，查看EPV分析结果
    3.  **模型训练**：选择待训练的基础模型，设置K折数，一键训练并查看评估结果
    4.  **集成学习**：基于已训练基础模型，运行投票集成，查看优化后模型性能
    5.  **结果下载**：打包下载所有处理后数据、模型文件、可视化图表
    """)

    # ========== 新增：插入本地示意图 ==========
    st.subheader("四、流程示意图")
    # 本地图片路径（图片与App.py同目录，直接写文件名；否则写完整路径，如：D:/xxx/workflow.png）
    local_image_path = "workflow.png"  # 替换为你的本地图片文件名/路径
    # 插入本地图片
    try:
        st.image(
            local_image_path,
            caption="机器学习分析全流程示意图",
            width=1500,
            use_container_width=False
        )
    except FileNotFoundError:
        st.warning("未找到本地示意图文件，请确认图片路径正确！")

    # 4. 注意事项（原第四部分，调整序号为五）
    st.subheader("五、注意事项")
    st.warning("""
    - 请确保上传的CSV文件为二分类任务数据，结局变量仅包含两个离散取值
    - 数据中避免包含特殊字符（如#、$、@），建议列名使用中文或英文字母
    - 模型训练时间与数据量、模型数量相关，大样本数据请耐心等待
    - 若需重新分析，可点击侧边栏「清理所有数据」按钮重置环境
    """)

    st.divider()
    st.markdown("### 💡 祝您使用愉快！如有问题，请检查各步骤配置或清理数据后重新尝试。")

# 1. 数据上传页面
if page == "数据上传":
    st.title("数据上传与设置")

    # 上传CSV文件
    uploaded_file = st.file_uploader("上传CSV数据文件", type=["csv"])

    if uploaded_file is not None:
        # 读取数据
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_data = df

        # 显示数据预览
        st.subheader("数据预览")
        st.dataframe(df.head())

        # 显示数据信息
        st.subheader("数据信息")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"样本量: {df.shape[0]}")
            st.write(f"特征数: {df.shape[1]}")
        with col2:
            st.write("缺失值情况:")
            st.write(df.isnull().sum())

        # 设置结局变量
        st.subheader("分析设置")
        target_options = df.columns.tolist()
        st.session_state.target = st.selectbox("选择结局变量", target_options)

        # 设置需要排除的变量
        exclude_vars = st.multiselect("选择需要直接排除的变量",
                                      [col for col in target_options if col != st.session_state.target])
        st.session_state.exclude_vars = exclude_vars

        # 保存设置
        if st.button("保存设置并进入数据清洗"):
            # 保存原始数据到临时目录
            df.to_csv(os.path.join(st.session_state.temp_dir, "raw_data.csv"), index=False)
            st.success("数据上传和设置完成！请前往数据清洗步骤。")

# 2. 数据清洗页面
elif page == "数据清洗":
    st.title("数据清洗")

    if 'raw_data' not in st.session_state:
        st.warning("请先上传数据并完成设置")
    else:
        # 显示当前设置
        st.subheader("当前设置")
        st.write(f"结局变量: {st.session_state.target}")
        st.write(f"排除变量: {', '.join(st.session_state.exclude_vars) if st.session_state.exclude_vars else '无'}")

        # 数据清洗参数
        st.subheader("清洗参数")
        max_iter = st.slider("MICE插补迭代次数", 5, 20, 10)

        # 运行数据清洗
        if st.button("运行数据清洗"):
            from cleaner import MICE_Impute, epv_estimation, Variable_Selection

            with st.spinner("正在进行数据清洗..."):
                # 读取数据
                df = st.session_state.raw_data

                # 剔除指定变量
                df = df.drop(st.session_state.exclude_vars, axis=1)
                # 剔除结局变量缺失的样本
                df = df.dropna(subset=[st.session_state.target])

                # 多重插补
                st.info("正在进行缺失值插补...")
                df_imputed = MICE_Impute(df_1=df, target=st.session_state.target)

                # 样本量估算
                st.info("正在进行样本量估算...")
                epv_result = epv_estimation(data=df_imputed, target_var=st.session_state.target)

                # 显示EPV结果
                st.subheader("EPV分析结果")
                epv_df = pd.DataFrame(list(epv_result.items()), columns=["指标", "值"])
                st.dataframe(epv_df)

                # 变量选择
                final_data = None
                if epv_result['当前样本量'] <= epv_result['目标EPV(10)所需总样本量']:
                    st.info("样本量不足，进行变量选择...")
                    final_data = Variable_Selection(df_imputed, st.session_state.target)
                    # 显示选择后的EPV
                    st.subheader("变量选择后的EPV结果")
                    post_epv = epv_estimation(data=final_data, target_var=st.session_state.target)
                    post_epv_df = pd.DataFrame(list(post_epv.items()), columns=["指标", "值"])
                    st.dataframe(post_epv_df)
                else:
                    final_data = df_imputed
                    st.info("样本量充足，不进行变量选择")

                # 保存处理后的数据
                final_data.to_csv(os.path.join(st.session_state.temp_dir, "processed_data.csv"), index=False)
                st.session_state.processed_data = final_data

                st.success("数据清洗完成！")

                # 显示处理后的数据
                st.subheader("处理后的数据预览")
                st.dataframe(final_data.head())

# # 3. 模型训练页面
elif page == "模型训练":
    st.title("基础模型训练")

    if st.session_state.processed_data is None:
        st.warning("请先完成数据清洗步骤")
    else:
        # 模型选择
        st.subheader("选择模型")


        model_options = {
            'Logistic Regression': LogisticRegression(),
            # 'SVM': SVC(probability=True)),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'GBDT': GradientBoostingClassifier(),
            'HGBDT': HistGradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            # # 'CatBoost': CatBoostClassifier(logging_level='Silent')),  # 关键：禁用日志输出，不创建catboost_info目录
            'Extra Trees': ExtraTreesClassifier(),
            'LGBM': LGBMClassifier(),
            'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42), random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        }

        selected_models = st.multiselect(
            "选择要训练的模型",
            list(model_options.keys()),
            default=["Logistic Regression", "Decision Tree"]
        )

        # 交叉验证设置
        st.subheader("交叉验证设置")
        k_fold = st.slider("K折交叉验证", 3, 10, 5)

        # 训练模型
        if st.button("开始训练模型") and selected_models:
            # 需要修改base_ML.py以适应Streamlit环境
            import base_ML
            # from sklearn.model_selection import StratifiedKFold
            # from imblearn.combine import SMOTETomek
            # from sklearn.preprocessing import StandardScaler

            # 创建结果目录
            results_dir = os.path.join(st.session_state.temp_dir, "results")
            models_dir = os.path.join(st.session_state.temp_dir, "models")
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)

            # 设置base_ML.py需要的全局变量
            base_ML.target = st.session_state.target
            base_ML.fold_results_dir = results_dir

            # 准备分类器列表
            classifiers = [(name, model_options[name]) for name in selected_models]
            print("正在训练模型...")
            with st.spinner("正在训练模型..."):
                # 调用修改后的base_ml_train函数
                results = base_ML.base_ml_train(
                    classifiers=classifiers,
                    project_path=st.session_state.temp_dir + "/",
                    input_path="processed_data.csv",
                    output_file=results_dir + "/",
                    K=k_fold
                )

                st.session_state.models_trained = True
                st.session_state.model_results = results
                # print('st.session_state.model_results', st.session_state.model_results)
                st.session_state.models = selected_models

                st.success("模型训练完成！")

        # 显示训练结果
        if st.session_state.models_trained:
            st.subheader("模型性能汇总")

            # 加载结果
            results_dir = os.path.join(st.session_state.temp_dir, "results")
            try:
                # 加载整体结果
                summary_df = pd.read_csv(os.path.join(results_dir,
                                                      f'{k_fold}-fold_CV_results_of_{len(selected_models)}_ML_models_on_whole_dataset.csv'))
                st.dataframe(summary_df)

                # 显示AUC和混淆矩阵
                st.subheader("模型评估可视化")

                # 加载模型结果
                from Voting2 import load_fold_results, Explain

                model_metrics, _ = load_fold_results(results_dir)
                #
                # # 展示ROC曲线
                st.image(plt.imread(st.session_state.temp_dir + "/out/single_ROC.jpg"), width=800)

                # 绘制混淆矩阵
                st.subheader("混淆矩阵")
                model_name = st.selectbox("选择模型查看混淆矩阵", list(model_metrics.keys()))

                metrics = model_metrics[model_name]
                cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('y prediction')
                ax.set_ylabel('y true')
                ax.set_title(f'{model_name} confusion matrix')
                st.pyplot(fig)
                # st.image(fig)

            except Exception as e:
                st.error(f"加载结果时出错: {str(e)}")


# 4. 集成学习页面
elif page == "集成学习":
    st.header("集成学习与模型解释")

    if st.session_state.model_results is None:
        st.warning("请先完成模型训练")
    else:
        # 选择要集成的模型
        available_models = list(st.session_state.model_results.keys())
        if len(available_models) < 2:
            st.warning("至少需要两个模型才能进行集成学习")
        else:
            st.subheader("选择集成模型")
            selected_models = st.multiselect(
                "选择要集成的模型",
                available_models,
                available_models[:2]
            )

            X_data = st.session_state.processed_data.drop(st.session_state.target, axis=1)
            total_samples = len(X_data)
            sample_id = st.number_input(
                f"Waterfall图绘制：输入样本编号（0~{total_samples-1}）",
                min_value=1,
                max_value=total_samples,
                value=1,  # 默认选择第1个样本
                step=1
            )
            if sample_id > total_samples:
                st.warning(f"不能超出{total_samples-1}")
            elif len(selected_models) < 2:
                st.warning("请至少选择两个模型")
            else:
                # 开始集成学习按钮
                if st.button("开始集成学习"):
                    with st.spinner("正在进行集成学习与模型解释..."):
                        try:
                            # 准备数据
                            df = st.session_state.processed_data
                            target = st.session_state.target
                            model_results = {}
                            model_metr = {}

                            from Voting2 import load_fold_results, delong_test, classification_metrics
                            with st.spinner("正在进行集成学习..."):
                                results_dir = os.path.join(st.session_state.temp_dir, "results")
                                model_metrics, model_results = load_fold_results(results_dir)
                                print("model_results", model_results)

                            # 保存模型到临时文件
                            for name in selected_models:
                                model_results[name] = {}
                                model_metr[name] = {
                                    'y_true': st.session_state.model_results[name]['y_true'],
                                    'y_proba': st.session_state.model_results[name]['y_proba'],
                                    'X_id': np.arange(len(st.session_state.model_results[name]['y_true']))
                                }

                            # 运行Voting2.py中的Explain函数
                            # df.to_csv("temp/processed_data.csv", index=False)
                            from Voting2 import Explain, load_fold_results

                            model_metr, model_results = load_fold_results(results_dir=results_dir)
                            Explain(
                                # model_results=st.session_state.model_results,
                                model_results=model_results,
                                sample_id=sample_id,
                                model_metr=model_metr,
                                target=target,
                                names=" + ".join(selected_models),
                                project_path=st.session_state.temp_dir + "/",
                            )

                            st.session_state.voting_done = True
                            st.success("集成学习与模型解释完成！")

                            # 显示所有绘图结果
                            st.subheader("校准曲线")
                            calib_plot = plt.imread(st.session_state.temp_dir + "/plots/calibration_curve.jpg")#plots/calibration_curve.pdf
                            st.image(calib_plot, width=1000)

                            st.subheader("DCA曲线")
                            dca_plot = plt.imread(st.session_state.temp_dir + "/plots/dca_bootstrap.jpg")
                            st.image(dca_plot, width=1000)

                            # st.subheader("SHAP特征重要性（条形图）")
                            # shap_bar = plt.imread(st.session_state.temp_dir + "/plots/bar.jpg")
                            # st.image(shap_bar, width=800)
                            #
                            # st.subheader("SHAP特征重要性（蜂群图）")
                            # shap_summary = plt.imread(st.session_state.temp_dir +"/plots/summary.jpg")
                            # st.image(shap_summary, width=800)

                            st.subheader("SHAP摘要图（特征重要性+蜂群图）")
                            shap_summary = plt.imread(st.session_state.temp_dir + "/plots/summary.jpg")
                            st.image(shap_summary, width=1000)

                            st.subheader("SHAP决策图")
                            shap_decison = plt.imread(st.session_state.temp_dir + "/plots/decision.jpg")
                            st.image(shap_decison, width=1000)

                            st.subheader("SHAP瀑布图")
                            shap_waterfall = plt.imread(st.session_state.temp_dir + f"/plots/waterfall_{sample_id}.jpg")
                            st.image(shap_waterfall, width=1000)

                            st.subheader("SHAP热图")
                            shap_heatmap = plt.imread(st.session_state.temp_dir + "/plots/heatmap.jpg")
                            st.image(shap_heatmap, width=1000)

                        except Exception as e:
                            st.error(f"集成学习出错: {str(e)}")

# 5. 结果下载页面
elif page == "结果下载":
    st.title("结果下载")

    if not st.session_state.voting_done:
        st.warning("请先完成集成学习步骤")
    else:
        # 创建一个压缩文件包含所有结果
        from zipfile import ZipFile

        # 准备要下载的文件
        files_to_zip = []

        # 处理后的数据
        processed_data_path = os.path.join(st.session_state.temp_dir, "processed_data.csv")
        if os.path.exists(processed_data_path):
            files_to_zip.append(processed_data_path)

        # 模型结果
        results_dir = os.path.join(st.session_state.temp_dir, "results")
        if os.path.exists(results_dir):
            for root, _, files in os.walk(results_dir):
                for file in files:
                    files_to_zip.append(os.path.join(root, file))

        # 模型文件
        models_dir = os.path.join(st.session_state.temp_dir, "models")
        if os.path.exists(models_dir):
            for root, _, files in os.walk(models_dir):
                for file in files:
                    files_to_zip.append(os.path.join(root, file))

        # 基础模型可视化
        models_dir = os.path.join(st.session_state.temp_dir, "out")
        if os.path.exists(models_dir):
            for root, _, files in os.walk(models_dir):
                for file in files:
                    files_to_zip.append(os.path.join(root, file))

        # 可视化结果
        models_dir = os.path.join(st.session_state.temp_dir, "plots")
        if os.path.exists(models_dir):
            for root, _, files in os.walk(models_dir):
                for file in files:
                    files_to_zip.append(os.path.join(root, file))

        # 创建ZIP文件
        zip_path = os.path.join(st.session_state.temp_dir, "analysis_results.zip")
        with ZipFile(zip_path, 'w') as zipf:
            for file in files_to_zip:
                arcname = os.path.relpath(file, st.session_state.temp_dir)
                zipf.write(file, arcname=arcname)

        # 提供下载
        with open(zip_path, "rb") as f:
            st.download_button(
                label="下载所有结果",
                data=f,
                file_name="analysis_results.zip",
                mime="application/zip"
            )

# 清理临时文件（可选）
if st.sidebar.button("清理所有数据"):
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)
    st.session_state.clear()
    st.success("所有临时数据已清理")
    st.rerun()
