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

# ====================== 1. 多语言配置 ======================
# 定义双语字典
LANGUAGES = {
    "zh": {
        # 导航
        "nav_title": "导航",
        "nav_project_intro": "项目介绍",
        "nav_data_upload": "数据上传",
        "nav_data_cleaning": "数据清洗",
        "nav_model_training": "模型训练",
        "nav_ensemble_learning": "集成学习",
        "nav_result_download": "结果下载",
        # 项目介绍页
        "project_title": "📊 运动损伤机器学习分析平台",
        "project_developer": "@Developer：Yiqun Pang",
        "project_overview": "一、项目概述",
        "project_overview_text": "本平台是一款基于Streamlit搭建的可视化机器学习分析工具，专为分类任务（二分类）设计，若涉及多分类任务请事先将结局变量转换为哑变量并分批建模，提供从**数据上传→数据清洗→模型训练→集成学习→结果下载**的全流程自动化分析能力，无需编写复杂代码，通过可视化交互即可完成高质量的机器学习建模与评估。",
        "project_features": "二、核心功能",
        "project_feature1": "- 📤 **数据上传与配置**：支持CSV格式数据上传，自定义结局变量与排除变量",
        "project_feature2": "- 🧹 **智能数据清洗**：内置MICE缺失值插补、EPV样本量估算、自动变量选择",
        "project_feature3": "- 🤖 **基础模型训练**：支持多种经典分类模型，自定义K折交叉验证",
        "project_feature4": "- 📈 **模型评估可视化**：自动绘制ROC曲线、混淆矩阵、DCA曲线、校准曲线",
        "project_feature5": "- 🔗 **集成学习优化**：基于投票机制提升模型性能，支持模型间AUC比较（DeLong检验）",
        "project_feature6": "- 📥 **结果批量下载**：打包所有数据、模型、图表结果，方便后续整理与汇报",
        "project_process": "三、操作流程",
        "project_process_text": "1.  **数据上传**：上传CSV文件，选择结局变量（二分类），勾选需要排除的无关变量\n2.  **数据清洗**：确认配置后，一键运行清洗流程，查看EPV分析结果\n3.  **模型训练**：选择待训练的基础模型，设置K折数，一键训练并查看评估结果\n4.  **集成学习**：基于已训练基础模型，运行投票集成，查看优化后模型性能\n5.  **结果下载**：打包下载所有处理后数据、模型文件、可视化图表",
        "project_workflow": "四、流程示意图",
        "project_notice": "五、注意事项",
        "project_notice_text": "- 请确保上传的CSV文件为二分类任务数据，结局变量仅包含两个离散取值\n- 数据中避免包含特殊字符（如#、$、@），建议列名使用中文或英文字母\n- 模型训练时间与数据量、模型数量相关，大样本数据请耐心等待\n- 若需重新分析，可点击侧边栏「清理所有数据」按钮重置环境",
        "project_wish": "💡 祝您使用愉快！如有问题，请检查各步骤配置或清理数据后重新尝试。",
        # 数据上传页
        "upload_title": "数据上传与设置",
        "upload_file": "上传CSV数据文件",
        "upload_preview": "数据预览",
        "upload_info": "数据信息",
        "upload_samples": "样本量:",
        "upload_features": "特征数:",
        "upload_missing": "缺失值情况:",
        "upload_settings": "分析设置",
        "upload_target": "选择结局变量",
        "upload_exclude": "选择需要直接排除的变量",
        "upload_save": "保存设置并进入数据清洗",
        "upload_success": "数据上传和设置完成！请前往数据清洗步骤。",
        # 数据清洗页
        "clean_title": "数据清洗",
        "clean_current_settings": "当前设置",
        "clean_target": "结局变量:",
        "clean_exclude": "排除变量:",
        "clean_params": "清洗参数",
        "clean_mice_iter": "MICE插补迭代次数",
        "clean_run": "运行数据清洗",
        "clean_imputing": "正在进行缺失值插补...",
        "clean_epv": "正在进行样本量估算...",
        "clean_epv_result": "EPV分析结果",
        "clean_epv_insufficient": "样本量不足，进行变量选择...",
        "clean_epv_post": "变量选择后的EPV结果",
        "clean_epv_sufficient": "样本量充足，不进行变量选择",
        "clean_success": "数据清洗完成！",
        "clean_preview": "处理后的数据预览",
        "clean_warning": "请先上传数据并完成设置",
        # 模型训练页
        "train_title": "基础模型训练",
        "train_select_model": "选择模型",
        "train_cv": "交叉验证设置",
        "train_kfold": "K折交叉验证",
        "train_start": "开始训练模型",
        "train_success": "模型训练完成！",
        "train_summary": "模型性能汇总",
        "train_visual": "模型评估可视化",
        "train_cm": "混淆矩阵",
        "train_select_cm_model": "选择模型查看混淆矩阵",
        "train_warning": "请先完成数据清洗步骤",
        "train_error": "加载结果时出错:",
        # 集成学习页
        "ensemble_title": "集成学习与模型解释",
        "ensemble_select_models": "选择集成模型",
        "ensemble_warning_min": "至少需要两个模型才能进行集成学习",
        "ensemble_waterfall_id": "Waterfall图绘制：输入样本编号（0~{0}）",
        "ensemble_warning_id": "不能超出{0}",
        "ensemble_warning_select": "请至少选择两个模型",
        "ensemble_start": "开始集成学习",
        "ensemble_running": "正在进行集成学习与模型解释...",
        "ensemble_running_voting": "正在进行集成学习...",
        "ensemble_success": "集成学习与模型解释完成！",
        "ensemble_calib_curve": "校准曲线",
        "ensemble_dca_curve": "DCA曲线",
        "ensemble_shap_summary": "SHAP摘要图（特征重要性+蜂群图）",
        "ensemble_shap_waterfall": "SHAP瀑布图",
        "ensemble_shap_heatmap": "SHAP热图",
        "ensemble_warning": "请先完成模型训练",
        "ensemble_error": "集成学习出错:",
        # 结果下载页
        "download_title": "结果下载",
        "download_button": "下载所有结果",
        "download_warning": "请先完成集成学习步骤",
        # 通用
        "clean_all": "清理所有数据",
        "clean_all_success": "所有临时数据已清理",
        "warning": "警告",
        "error": "错误",
        "info": "信息"
    },
    "en": {
        # 导航
        "nav_title": "Navigation",
        "nav_project_intro": "Project Introduction",
        "nav_data_upload": "Data Upload",
        "nav_data_cleaning": "Data Cleaning",
        "nav_model_training": "Model Training",
        "nav_ensemble_learning": "Ensemble Learning",
        "nav_result_download": "Result Download",
        # 项目介绍页
        "project_title": "📊 Sports Injury Machine Learning Analysis Platform",
        "project_developer": "@Developer：Yiqun Pang",
        "project_overview": "1. Project Overview",
        "project_overview_text": "This platform is a visual machine learning analysis tool built with Streamlit, designed specifically for binary classification tasks. For multi-class classification tasks, please convert the outcome variable into dummy variables and build models in batches in advance. It provides end-to-end automated analysis capabilities from **Data Upload → Data Cleaning → Model Training → Ensemble Learning → Result Download**, eliminating the need to write complex code. High-quality machine learning modeling and evaluation can be completed through visual interaction.",
        "project_features": "2. Core Features",
        "project_feature1": "- 📤 **Data Upload & Configuration**: Support CSV data upload, customize outcome variables and excluded variables",
        "project_feature2": "- 🧹 **Intelligent Data Cleaning**: Built-in MICE missing value imputation, EPV sample size estimation, automatic variable selection",
        "project_feature3": "- 🤖 **Basic Model Training**: Support multiple classic classification models, customize K-fold cross-validation",
        "project_feature4": "- 📈 **Model Evaluation Visualization**: Automatically plot ROC curves, confusion matrices, DCA curves, calibration curves",
        "project_feature5": "- 🔗 **Ensemble Learning Optimization**: Improve model performance based on voting mechanism, support AUC comparison between models (DeLong test)",
        "project_feature6": "- 📥 **Batch Result Download**: Package all data, models, and chart results for subsequent sorting and reporting",
        "project_process": "3. Operation Process",
        "project_process_text": "1.  **Data Upload**: Upload CSV file, select outcome variable (binary), check irrelevant variables to exclude\n2.  **Data Cleaning**: Confirm configuration, run cleaning process with one click, view EPV analysis results\n3.  **Model Training**: Select basic models to train, set K-fold number, train with one click and view evaluation results\n4.  **Ensemble Learning**: Run voting ensemble based on trained basic models, view optimized model performance\n5.  **Result Download**: Download all processed data, model files, and visualization charts in a package",
        "project_workflow": "4. Workflow Diagram",
        "project_notice": "5. Notes",
        "project_notice_text": "- Ensure the uploaded CSV file is for binary classification tasks, with the outcome variable containing only two discrete values\n- Avoid special characters (e.g., #, $, @) in the data; it is recommended to use Chinese or English letters for column names\n- Model training time is related to data volume and number of models; please wait patiently for large sample data\n- To re-analyze, click the \"Clear All Data\" button in the sidebar to reset the environment",
        "project_wish": "💡 Wish you a pleasant use! If you encounter any problems, please check the configuration of each step or try again after clearing the data.",
        # 数据上传页
        "upload_title": "Data Upload & Settings",
        "upload_file": "Upload CSV Data File",
        "upload_preview": "Data Preview",
        "upload_info": "Data Information",
        "upload_samples": "Sample Size:",
        "upload_features": "Feature Count:",
        "upload_missing": "Missing Values:",
        "upload_settings": "Analysis Settings",
        "upload_target": "Select Outcome Variable",
        "upload_exclude": "Select Variables to Exclude Directly",
        "upload_save": "Save Settings and Enter Data Cleaning",
        "upload_success": "Data upload and settings completed! Please proceed to the data cleaning step.",
        # 数据清洗页
        "clean_title": "Data Cleaning",
        "clean_current_settings": "Current Settings",
        "clean_target": "Outcome Variable:",
        "clean_exclude": "Excluded Variables:",
        "clean_params": "Cleaning Parameters",
        "clean_mice_iter": "MICE Imputation Iterations",
        "clean_run": "Run Data Cleaning",
        "clean_imputing": "Performing missing value imputation...",
        "clean_epv": "Performing sample size estimation...",
        "clean_epv_result": "EPV Analysis Results",
        "clean_epv_insufficient": "Insufficient sample size, performing variable selection...",
        "clean_epv_post": "EPV Results After Variable Selection",
        "clean_epv_sufficient": "Sufficient sample size, no variable selection performed",
        "clean_success": "Data cleaning completed!",
        "clean_preview": "Processed Data Preview",
        "clean_warning": "Please upload data and complete settings first",
        # 模型训练页
        "train_title": "Basic Model Training",
        "train_select_model": "Select Models",
        "train_cv": "Cross-Validation Settings",
        "train_kfold": "K-Fold Cross-Validation",
        "train_start": "Start Training Models",
        "train_success": "Model training completed!",
        "train_summary": "Model Performance Summary",
        "train_visual": "Model Evaluation Visualization",
        "train_cm": "Confusion Matrix",
        "train_select_cm_model": "Select Model to View Confusion Matrix",
        "train_warning": "Please complete the data cleaning step first",
        "train_error": "Error loading results:",
        # 集成学习页
        "ensemble_title": "Ensemble Learning & Model Interpretation",
        "ensemble_select_models": "Select Ensemble Models",
        "ensemble_warning_min": "At least two models are required for ensemble learning",
        "ensemble_waterfall_id": "Waterfall Plot: Enter Sample ID (0~{0})",
        "ensemble_warning_id": "Cannot exceed {0}",
        "ensemble_warning_select": "Please select at least two models",
        "ensemble_start": "Start Ensemble Learning",
        "ensemble_running": "Performing ensemble learning and model interpretation...",
        "ensemble_running_voting": "Performing ensemble learning...",
        "ensemble_success": "Ensemble learning and model interpretation completed!",
        "ensemble_calib_curve": "Calibration Curve",
        "ensemble_dca_curve": "DCA Curve",
        "ensemble_shap_summary": "SHAP Summary Plot (Feature Importance + Beeswarm Plot)",
        "ensemble_shap_waterfall": "SHAP Waterfall Plot",
        "ensemble_shap_heatmap": "SHAP Heatmap",
        "ensemble_warning": "Please complete the model training step first",
        "ensemble_error": "Error in ensemble learning:",
        # 结果下载页
        "download_title": "Result Download",
        "download_button": "Download All Results",
        "download_warning": "Please complete the ensemble learning step first",
        # 通用
        "clean_all": "Clear All Data",
        "clean_all_success": "All temporary data has been cleared",
        "warning": "Warning",
        "error": "Error",
        "info": "Info"
    }
}

# # 初始化语言状态（默认中文）
# if 'lang' not in st.session_state:
#     st.session_state.lang = "zh"
# 初始化语言状态（默认英文）
if 'lang' not in st.session_state:
    st.session_state.lang = "en"

# 获取当前语言的翻译函数
def t(key):
    """获取当前语言的翻译文本"""
    return LANGUAGES[st.session_state.lang].get(key, key)

# ====================== 2. 页面基础配置 ======================
# 设置页面配置
st.set_page_config(
    page_title=t("project_title").replace("📊 ", ""),
    page_icon="📊",
    layout="wide"
)

# 确保中文显示正常
plt.rcParams["axes.unicode_minus"] = False
if st.session_state.lang == "zh":
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
else:
    plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]

# 临时目录设置
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.processed_data = None
    st.session_state.models_trained = False
    st.session_state.voting_done = False

# 全局变量存储数据和结果
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
if 'excluded_vars' not in st.session_state:
    st.session_state.excluded_vars = []

# ====================== 3. 侧边栏（语言切换 + 导航） ======================
st.sidebar.title(t("nav_title"))

# 语言切换按钮
col_lang1, col_lang2 = st.sidebar.columns(2)
with col_lang1:
    if st.button("中文", use_container_width=True):
        st.session_state.lang = "zh"
        st.rerun()
with col_lang2:
    if st.button("English", use_container_width=True):
        st.session_state.lang = "en"
        st.rerun()

# 导航菜单
page = st.sidebar.radio(
    "",
    [t("nav_project_intro"), t("nav_data_upload"), t("nav_data_cleaning"), 
     t("nav_model_training"), t("nav_ensemble_learning"), t("nav_result_download")]
)

# ====================== 4. 项目介绍页 ======================
if page == t("nav_project_intro"):
    st.title(t("project_title"))
    st.markdown(f"#### @Developer：Yiqun Pang")
    st.divider()

    # 1. 项目概述
    st.subheader(t("project_overview"))
    st.markdown(t("project_overview_text"))

    # 2. 核心功能
    st.subheader(t("project_features"))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"{t('project_feature1')}\n{t('project_feature2')}\n{t('project_feature3')}")
    with col2:
        st.markdown(f"{t('project_feature4')}\n{t('project_feature5')}\n{t('project_feature6')}")

    # 3. 操作流程
    st.subheader(t("project_process"))
    st.markdown(t("project_process_text"))

    # 4. 流程示意图
    st.subheader(t("project_workflow"))
    local_image_path = "workflow.png"
    try:
        st.image(
            local_image_path,
            caption="Machine Learning Analysis Workflow Diagram" if st.session_state.lang == "en" else "机器学习分析全流程示意图",
            width=1500,
        )
    except FileNotFoundError:
        st.warning(t("warning") + ": " + ("Local workflow diagram file not found! Please check the file path." if st.session_state.lang == "en" else "未找到本地示意图文件，请确认图片路径正确！"))

    # 5. 注意事项
    st.subheader(t("project_notice"))
    st.warning(t("project_notice_text"))

    st.divider()
    st.markdown(f"### {t('project_wish')}")

# ====================== 5. 数据上传页 ======================
elif page == t("nav_data_upload"):
    st.title(t("upload_title"))

    # 上传CSV文件
    uploaded_file = st.file_uploader(t("upload_file"), type=["csv"])

    if uploaded_file is not None:
        # 读取数据
        df = pd.read_csv(uploaded_file)
        st.session_state.raw_data = df

        # 显示数据预览
        st.subheader(t("upload_preview"))
        st.dataframe(df.head())

        # 显示数据信息
        st.subheader(t("upload_info"))
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{t('upload_samples')} {df.shape[0]}")
            st.write(f"{t('upload_features')} {df.shape[1]}")
        with col2:
            st.write(t("upload_missing") + ":")
            st.write(df.isnull().sum())

        # 设置结局变量
        st.subheader(t("upload_settings"))
        target_options = df.columns.tolist()
        st.session_state.target = st.selectbox(t("upload_target"), target_options)

        # 设置需要排除的变量
        exclude_vars = st.multiselect(
            t("upload_exclude"),
            [col for col in target_options if col != st.session_state.target]
        )
        st.session_state.exclude_vars = exclude_vars

        # 保存设置
        if st.button(t("upload_save")):
            df.to_csv(os.path.join(st.session_state.temp_dir, "raw_data.csv"), index=False)
            st.success(t("upload_success"))

# ====================== 6. 数据清洗页 ======================
elif page == t("nav_data_cleaning"):
    st.title(t("clean_title"))

    if 'raw_data' not in st.session_state:
        st.warning(t("clean_warning"))
    else:
        # 显示当前设置
        st.subheader(t("clean_current_settings"))
        st.write(f"{t('clean_target')} {st.session_state.target}")
        st.write(f"{t('clean_exclude')} {', '.join(st.session_state.exclude_vars) if st.session_state.exclude_vars else t('无') if st.session_state.lang == 'zh' else 'None'}")

        # 数据清洗参数
        st.subheader(t("clean_params"))
        max_iter = st.slider(t("clean_mice_iter"), 5, 20, 10)

        # 运行数据清洗
        if st.button(t("clean_run")):
            from cleaner import MICE_Impute, epv_estimation, Variable_Selection

            with st.spinner(t("clean_imputing")):
                # 读取数据
                df = st.session_state.raw_data

                # 剔除指定变量
                df = df.drop(st.session_state.exclude_vars, axis=1)
                # 剔除结局变量缺失的样本
                df = df.dropna(subset=[st.session_state.target])

                # 多重插补
                st.info(t("clean_imputing"))
                df_imputed = MICE_Impute(df_1=df, target=st.session_state.target)

                # 样本量估算
                st.info(t("clean_epv"))
                epv_result = epv_estimation(data=df_imputed, target_var=st.session_state.target)

                # 显示EPV结果
                st.subheader(t("clean_epv_result"))
                epv_df = pd.DataFrame(list(epv_result.items()), columns=[t("指标") if st.session_state.lang == "zh" else "Indicator", t("值") if st.session_state.lang == "zh" else "Value"])
                st.dataframe(epv_df)

                # 变量选择
                final_data = None
                if epv_result['当前样本量'] <= epv_result['目标EPV(10)所需总样本量']:
                    st.info(t("clean_epv_insufficient"))
                    final_data = Variable_Selection(df_imputed, st.session_state.target)
                    # 显示选择后的EPV
                    st.subheader(t("clean_epv_post"))
                    post_epv = epv_estimation(data=final_data, target_var=st.session_state.target)
                    post_epv_df = pd.DataFrame(list(post_epv.items()), columns=[t("指标") if st.session_state.lang == "zh" else "Indicator", t("值") if st.session_state.lang == "zh" else "Value"])
                    st.dataframe(post_epv_df)
                else:
                    final_data = df_imputed
                    st.info(t("clean_epv_sufficient"))

                # 保存处理后的数据
                final_data.to_csv(os.path.join(st.session_state.temp_dir, "processed_data.csv"), index=False)
                st.session_state.processed_data = final_data

                st.success(t("clean_success"))

                # 显示处理后的数据
                st.subheader(t("clean_preview"))
                st.dataframe(final_data.head())

# ====================== 7. 模型训练页 ======================
elif page == t("nav_model_training"):
    st.title(t("train_title"))

    if st.session_state.processed_data is None:
        st.warning(t("train_warning"))
    else:
        # 模型选择
        st.subheader(t("train_select_model"))

        # 模型名称适配多语言
        model_names_zh = {
            'Logistic Regression': '逻辑回归',
            'KNN': 'K近邻',
            'Naive Bayes': '朴素贝叶斯',
            'Decision Tree': '决策树',
            'Random Forest': '随机森林',
            'GBDT': '梯度提升树',
            'HGBDT': '直方图梯度提升树',
            'AdaBoost': '自适应提升',
            'Extra Trees': '极端随机树',
            'LGBM': '轻量梯度提升机',
            'Bagging': '装袋法',
            "XGBoost": '极端梯度提升'
        }
        model_options = {
            'Logistic Regression': LogisticRegression(),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'GBDT': GradientBoostingClassifier(),
            'HGBDT': HistGradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
            'Extra Trees': ExtraTreesClassifier(),
            'LGBM': LGBMClassifier(),
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        }
        # 模型显示名称（根据语言切换）
        display_model_names = {
            k: (model_names_zh[k] if st.session_state.lang == "zh" else k)
            for k in model_options.keys()
        }
        reverse_display_names = {v: k for k, v in display_model_names.items()}

        selected_display_models = st.multiselect(
            "",
            list(display_model_names.values()),
            default=[model_names_zh["Logistic Regression"] if st.session_state.lang == "zh" else "Logistic Regression", 
                     model_names_zh["Decision Tree"] if st.session_state.lang == "zh" else "Decision Tree"]
        )
        # 转换为原始模型名称
        selected_models = [reverse_display_names[name] for name in selected_display_models]

        # 交叉验证设置
        st.subheader(t("train_cv"))
        k_fold = st.slider(t("train_kfold"), 3, 10, 5)

        # 训练模型
        if st.button(t("train_start")) and selected_models:
            import base_ML

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
            with st.spinner(t("train_start").replace("开始", "正在") if st.session_state.lang == "zh" else "Training models..."):
                results = base_ML.base_ml_train(
                    classifiers=classifiers,
                    project_path=st.session_state.temp_dir + "/",
                    input_path="processed_data.csv",
                    output_file=results_dir + "/",
                    K=k_fold
                )

                st.session_state.models_trained = True
                st.session_state.model_results = results
                st.session_state.models = selected_models

                st.success(t("train_success"))

        # 显示训练结果
        if st.session_state.models_trained:
            st.subheader(t("train_summary"))

            # 加载结果
            results_dir = os.path.join(st.session_state.temp_dir, "results")
            try:
                # 加载整体结果
                summary_file = f'{k_fold}-fold_CV_results_of_{len(selected_models)}_ML_models_on_whole_dataset.csv'
                summary_df = pd.read_csv(os.path.join(results_dir, summary_file))
                st.dataframe(summary_df)

                # 显示AUC和混淆矩阵
                st.subheader(t("train_visual"))

                # 加载模型结果
                from Voting2 import load_fold_results, Explain

                model_metrics, _ = load_fold_results(results_dir)

                # 展示ROC曲线
                st.image(plt.imread(st.session_state.temp_dir + "/out/single_ROC.jpg"), width=800)

                # 绘制混淆矩阵
                st.subheader(t("train_cm"))
                # 混淆矩阵模型选择适配多语言
                cm_model_display_names = [display_model_names[name] for name in list(model_metrics.keys())]
                selected_cm_display = st.selectbox(t("train_select_cm_model"), cm_model_display_names)
                selected_cm_model = reverse_display_names[selected_cm_display]

                metrics = model_metrics[selected_cm_model]
                cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('y prediction' if st.session_state.lang == "en" else '预测值')
                ax.set_ylabel('y true' if st.session_state.lang == "en" else '真实值')
                ax.set_title(f'{selected_cm_display} {t("train_cm")}')
                st.pyplot(fig)

            except Exception as e:
                st.error(f"{t('train_error')} {str(e)}")

# ====================== 8. 集成学习页 ======================
elif page == t("nav_ensemble_learning"):
    st.header(t("ensemble_title"))

    if st.session_state.model_results is None:
        st.warning(t("ensemble_warning"))
    else:
        # 选择要集成的模型
        available_models = list(st.session_state.model_results.keys())
        if len(available_models) < 2:
            st.warning(t("ensemble_warning_min"))
        else:
            # 模型名称适配多语言
            model_names_zh = {
                'Logistic Regression': '逻辑回归',
                'KNN': 'K近邻',
                'Naive Bayes': '朴素贝叶斯',
                'Decision Tree': '决策树',
                'Random Forest': '随机森林',
                'GBDT': '梯度提升树',
                'HGBDT': '直方图梯度提升树',
                'AdaBoost': '自适应提升',
                'Extra Trees': '极端随机树',
                'LGBM': '轻量梯度提升机',
                'Bagging': '装袋法',
                "XGBoost": '极端梯度提升'
            }
            display_available = [model_names_zh[m] if st.session_state.lang == "zh" else m for m in available_models]
            reverse_display = {v: k for k, v in zip(available_models, display_available)}

            st.subheader(t("ensemble_select_models"))
            selected_display = st.multiselect(
                "",
                display_available,
                display_available[:2]
            )
            selected_models = [reverse_display[name] for name in selected_display]

            X_data = st.session_state.processed_data.drop(st.session_state.target, axis=1)
            total_samples = len(X_data)
            sample_id = st.number_input(
                t("ensemble_waterfall_id").format(total_samples-1),
                min_value=1,
                max_value=total_samples,
                value=1,
                step=1
            )
            if sample_id > total_samples:
                st.warning(t("ensemble_warning_id").format(total_samples-1))
            elif len(selected_models) < 2:
                st.warning(t("ensemble_warning_select"))
            else:
                # 开始集成学习按钮
                if st.button(t("ensemble_start")):
                    with st.spinner(t("ensemble_running")):
                        try:
                            # 准备数据
                            df = st.session_state.processed_data
                            target = st.session_state.target
                            model_results = {}
                            model_metr = {}

                            from Voting2 import load_fold_results, delong_test, classification_metrics
                            with st.spinner(t("ensemble_running_voting")):
                                results_dir = os.path.join(st.session_state.temp_dir, "results")
                                model_metrics, model_results = load_fold_results(results_dir)

                            # 保存模型到临时文件
                            for name in selected_models:
                                model_results[name] = {}
                                model_metr[name] = {
                                    'y_true': st.session_state.model_results[name]['y_true'],
                                    'y_proba': st.session_state.model_results[name]['y_proba'],
                                    'X_id': np.arange(len(st.session_state.model_results[name]['y_true']))
                                }

                            # 运行Voting2.py中的Explain函数
                            from Voting2 import Explain, load_fold_results

                            model_metr, model_results = load_fold_results(results_dir=results_dir)
                            Explain(
                                model_results=model_results,
                                sample_id=sample_id,
                                model_metr=model_metr,
                                target=target,
                                names=" + ".join([model_names_zh[m] if st.session_state.lang == "zh" else m for m in selected_models]),
                                project_path=st.session_state.temp_dir + "/",
                            )

                            st.session_state.voting_done = True
                            st.success(t("ensemble_success"))

                            # 显示所有绘图结果
                            st.subheader(t("ensemble_calib_curve"))
                            calib_plot = plt.imread(st.session_state.temp_dir + "/plots/calibration_curve.jpg")
                            st.image(calib_plot, width=1000)

                            st.subheader(t("ensemble_dca_curve"))
                            dca_plot = plt.imread(st.session_state.temp_dir + "/plots/dca_bootstrap.jpg")
                            st.image(dca_plot, width=1000)

                            st.subheader(t("ensemble_shap_summary"))
                            shap_summary = plt.imread(st.session_state.temp_dir + "/plots/summary.jpg")
                            st.image(shap_summary, width=1000)

                            st.subheader(t("ensemble_shap_waterfall"))
                            shap_waterfall = plt.imread(st.session_state.temp_dir + f"/plots/waterfall_{sample_id}.jpg")
                            st.image(shap_waterfall, width=1000)

                            st.subheader(t("ensemble_shap_heatmap"))
                            shap_heatmap = plt.imread(st.session_state.temp_dir + "/plots/heatmap.jpg")
                            st.image(shap_heatmap, width=1000)

                        except Exception as e:
                            st.error(f"{t('ensemble_error')} {str(e)}")

# ====================== 9. 结果下载页 ======================
elif page == t("nav_result_download"):
    st.title(t("download_title"))

    if not st.session_state.voting_done:
        st.warning(t("download_warning"))
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
                label=t("download_button"),
                data=f,
                file_name="analysis_results.zip",
                mime="application/zip"
            )

# ====================== 10. 清理数据 ======================
if st.sidebar.button(t("clean_all")):
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        shutil.rmtree(st.session_state.temp_dir)
    st.session_state.clear()
    st.success(t("clean_all_success"))
    st.rerun()

