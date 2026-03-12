#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
# 新增：导入准确率评估指标
from sklearn.metrics import accuracy_score

# ===================== 核心配置（动态相对路径） =====================
# 获取脚本目录，作为基准点
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 项目根目录（scripts的父目录）
CORE_FOLDER = os.path.join(PROJECT_ROOT, "Kickstarter_Clean")

# 定义需要检查缺失的核心字段（可根据实际需求增减）
CHECK_MISSING_FIELDS = {
    "deadline": "截止时间",
    "launched_at": "发布时间",
    "goal": "筹款目标",
    "fx_rate": "汇率",
    "country": "国家",
    "category": "分类信息",
    "name": "项目名称",
    "id": "项目ID"
}

# 特征名称映射（改为英文，适配图表）
FEATURE_NAME_MAP = {
    "duration_days": "Funding Duration (days)",
    "cat_enc": "Project Category",
    "country_enc": "Country",
    "goal_usd": "Funding Goal (USD)",
    "fund_sim": "Goal Reasonableness",
    "time_ratio": "Duration Ratio (vs 7 days)"
}

# ===================== 初始化Matplotlib =====================
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False 
# ===================== 核心类：全量CSV训练 + 全量CSV预测 =====================
class KSFullCSVProcessor:
    def __init__(self):
        # 1. 自动创建模型子文件夹，统一存放模型文件
        self.model_dir = os.path.join(CORE_FOLDER, "model_output")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 2. 自动创建预测结果子文件夹，存放单个文件的预测结果
        self.pred_result_dir = os.path.join(CORE_FOLDER, "单个文件预测结果")
        os.makedirs(self.pred_result_dir, exist_ok=True)
        
        # 模型文件路径改为子文件夹内
        self.model_files = {
            "model": os.path.join(self.model_dir, "ks_model.pkl"),
            "scaler": os.path.join(self.model_dir, "ks_scaler.pkl"),
            "feats": os.path.join(self.model_dir, "ks_features.json"),
            "le_cat": os.path.join(self.model_dir, "ks_le_cat.pkl"),
            "le_country": os.path.join(self.model_dir, "ks_le_country.pkl"),
            "goal_median": os.path.join(self.model_dir, "ks_goal_median.json"),
            "feature_importance": os.path.join(self.model_dir, "feature_importance.json"),
            # 新增：保存模型准确率的文件路径
            "model_accuracy": os.path.join(self.model_dir, "model_accuracy.json")  
        }

    # -------------------------- 获取实时北京时间时间戳 --------------------------
    def _get_beijing_time_timestamp(self):
        """获取实时北京时间的秒级时间戳（UTC+8）"""
        beijing_tz = timezone(timedelta(hours=8))
        now = datetime.now(beijing_tz)
        return int(now.timestamp())

    # -------------------------- 精准识别缺失字段 --------------------------
    def _get_missing_fields_detail(self, row):
        """
        识别单条数据中缺失的具体字段，返回可读的缺失描述
        :param row: DataFrame的单行数据
        :return: 缺失字段描述（如"缺失截止时间、缺失国家"或"无缺失"）
        """
        missing_fields = []
        for field_code, field_name in CHECK_MISSING_FIELDS.items():            # 检查字段是否存在缺失（NaN/空字符串/0（仅针对非数值字段））
            if pd.isna(row[field_code]):
                missing_fields.append(field_name)
            elif isinstance(row[field_code], str) and row[field_code].strip() == "":
                missing_fields.append(field_name)
            # 对数值型关键字段（如id、goal），0也视为缺失
            elif field_code in ["id", "goal"] and row[field_code] == 0:
                missing_fields.append(field_name)
        
        if not missing_fields:
            return "无缺失"
        else:
            return "、".join([f"缺失{field}" for field in missing_fields])

    # -------------------------- 原有工具函数 --------------------------
    def _safe_ts_convert(self, ts):
        if pd.isna(ts) or str(ts).lower() in ["nan", "", "None"]:
            return datetime(2020, 1, 1)
        try:
            ts_int = int(float(ts))
            ts_int = ts_int // 1000 if ts_int > 1e12 else ts_int
            return datetime.fromtimestamp(ts_int)
        except:
            return datetime(2020, 1, 1)

    def _safe_parse_category(self, cat_str):
        if pd.isna(cat_str) or str(cat_str).lower() in ["nan", "", "None"]:
            return "OTHER"
        try:
            cat_dict = json.loads(cat_str.replace("'", '"').strip())
            return cat_dict.get("parent_name", cat_dict.get("name", "OTHER")).strip()
        except:
            return "OTHER"    # -------------------------- 提取并保存特征重要性（英文图表） --------------------------
    def _extract_feature_importance(self, model, feature_cols):
        """
        从随机森林模型中提取特征重要性，保存并返回可视化的结果（英文图表）
        :param model: 训练好的随机森林模型
        :param feature_cols: 特征列名列表
        :return: 特征重要性字典（英文名称: 重要性值）
        """
        # 获取特征重要性
        importances = model.feature_importances_
        # 构建特征重要性字典（代码特征名 → 英文业务名称 → 重要性值）
        feat_imp_dict = {}
        for col, imp in zip(feature_cols, importances):
            feat_imp_dict[FEATURE_NAME_MAP.get(col, col)] = round(imp, 4)
        
        # 按重要性降序排序
        feat_imp_sorted = dict(sorted(feat_imp_dict.items(), key=lambda x: x[1], reverse=True))
        
        # 保存到文件
        with open(self.model_files["feature_importance"], "w") as f:
            json.dump(feat_imp_sorted, f, ensure_ascii=False, indent=4)
        
        # 绘制英文图表
        plt.figure(figsize=(10, 6))
        plt.bar(feat_imp_sorted.keys(), feat_imp_sorted.values(), color='#3498db')
        plt.title("Core Factors Affecting Project Success/Failure (Feature Importance)", fontsize=12)
        plt.xlabel("Influencing Factors", fontsize=10)
        plt.ylabel("Importance Value", fontsize=10)
        plt.xticks(rotation=45, ha='right')  
        plt.tight_layout()  
        # 保存图片到model_output文件夹
        plt.savefig(os.path.join(self.model_dir, "feature_importance_analysis.png"), dpi=300)
        plt.close()
        
        return feat_imp_sorted
        # -------------------------- 分析失败影响因素 --------------------------
    def _analyze_failure_factors(self, df_train, feat_imp_sorted):
        """
        结合特征重要性和训练数据，分析“哪些因素导致项目失败”（英文结论）
        :param df_train: 训练数据集
        :param feat_imp_sorted: 排序后的特征重要性字典
        :return: 失败影响因素的英文结论
        """
        failure_factors = []
        # 1. 分析筹款目标的影响
        top_feat = list(feat_imp_sorted.keys())[0]
        if top_feat == "Funding Goal (USD)":
            # 计算失败/成功项目的筹款目标均值
            fail_goal_mean = df_train[df_train["label"] == 0]["goal_usd"].mean()
            succ_goal_mean = df_train[df_train["label"] == 1]["goal_usd"].mean()
            failure_factors.append(f"✅ Core Failure Factor: Excessively high funding goal (Failed projects: ${round(fail_goal_mean, 2)}, Successful projects: ${round(succ_goal_mean, 2)})")
        
        # 2. 分析众筹周期的影响
        if "Funding Duration (days)" in feat_imp_sorted:
            fail_duration_mean = df_train[df_train["label"] == 0]["duration_days"].mean()
            succ_duration_mean = df_train[df_train["label"] == 1]["duration_days"].mean()
            failure_factors.append(f"✅ Important Failure Factor: Unreasonable funding duration (Failed projects: {round(fail_duration_mean, 1)} days, Successful projects: {round(succ_duration_mean, 1)} days)")
        
        # 3. 分析分类/国家的影响
        if "Project Category" in feat_imp_sorted:
            failure_factors.append(f"✅ Important Failure Factor: Project category (Significant differences in failure rates across categories)")
        if "Country" in feat_imp_sorted:
            failure_factors.append(f"✅ Important Failure Factor: Country (Large differences in success rates across regions)")
        
        # 4. 补充字段缺失的影响（非模型特征，但业务上重要）
        failure_factors.append(f"✅ Critical Failure Factor: Missing core fields (e.g., deadline/goal missing directly reduces success rate)")
        
        return failure_factors

    # -------------------------- 步骤1：全量CSV训练 --------------------------    
    def full_csv_train(self):
        print("📌 Starting training (reading all CSV files in core folder)...")
        all_csv_files = [f for f in os.listdir(CORE_FOLDER) if f.endswith(".csv")]
        if not all_csv_files:
            raise ValueError(f"❌ No CSV files found in core folder: {CORE_FOLDER}!")
        
        df_list = []
        for file in all_csv_files:
            file_path = os.path.join(CORE_FOLDER, file)
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except:
                df = pd.read_csv(file_path, encoding="gbk")
            df_list.append(df)
            print(f"✅ Loaded training file: {file} (Records: {len(df)})")
        df_train = pd.concat(df_list, ignore_index=True)
        print(f"\n📊 Total training data after merging: {len(df_train)} records\n")

        df_train = df_train[df_train["state"].str.lower().isin(["successful", "failed"])].copy()
        df_train["label"] = df_train["state"].map({"successful": 1, "failed": 0})

        df_train["fx_rate"] = df_train["fx_rate"].fillna(1.0)
        df_train["goal"] = df_train["goal"].fillna(0)
        df_train["country"] = df_train["country"].fillna("OTHER")
        df_train["category"] = df_train["category"].fillna('{"name":"OTHER"}')

        df_train["goal_usd"] = df_train["goal"] * df_train["fx_rate"]
        df_train["duration_days"] = df_train.apply(
            lambda x: (self._safe_ts_convert(x["deadline"]) - self._safe_ts_convert(x["launched_at"])).days,
            axis=1
        )
        df_train["main_category"] = df_train["category"].apply(self._safe_parse_category)

        le_cat = LabelEncoder()
        df_train["cat_enc"] = le_cat.fit_transform(df_train["main_category"])
        le_country = LabelEncoder()
        df_train["country_enc"] = le_country.fit_transform(df_train["country"])

        # 修复：训练阶段就确保中位数≥0.01，从根源避免除以0
        goal_median = df_train.groupby("main_category")["goal_usd"].median().to_dict()
        goal_median = {k: max(v, 0.01) for k, v in goal_median.items()}
        
        df_train["goal_reasonable"] = df_train.apply(
            lambda x: x["goal_usd"] / goal_median.get(x["main_category"], 1), axis=1
        ).clip(0, 1000)
        df_train["fund_sim"] = (1 / df_train["goal_reasonable"]) * df_train["cat_enc"]
        df_train["time_ratio"] = df_train["duration_days"].apply(lambda x: max(0.01, (x-7)/max(x, 1)))

        feature_cols = ["duration_days", "cat_enc", "country_enc", "goal_usd", "fund_sim", "time_ratio"]
        with open(self.model_files["feats"], "w") as f:
            json.dump(feature_cols, f)
        with open(self.model_files["goal_median"], "w") as f:
            json.dump(goal_median, f)
        joblib.dump(le_cat, self.model_files["le_cat"])
        joblib.dump(le_country, self.model_files["le_country"])

        # 训练前清理异常值
        X = df_train[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        X = X.clip(-1e18, 1e18)
        
        y = df_train["label"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # 提前保存标准化后的特征矩阵
        model = RandomForestClassifier(random_state=42, n_estimators=150)
        model.fit(X_scaled, y)

        joblib.dump(model, self.model_files["model"])        
        joblib.dump(scaler, self.model_files["scaler"])
        
        # ===================== 新增：计算并保存模型整体准确率 =====================
        # 用训练集预测，计算准确率
        y_pred = model.predict(X_scaled)
        model_accuracy = accuracy_score(y, y_pred)
        # 保存准确率到文件
        accuracy_dict = {
            "overall_accuracy": round(model_accuracy, 4),
            "accuracy_percentage": round(model_accuracy * 100, 2),
            "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.model_files["model_accuracy"], "w") as f:
            json.dump(accuracy_dict, f, ensure_ascii=False, indent=4)
        
        # ===================== 提取特征重要性 + 分析失败因素 =====================
        print("\n📊 Extracting feature importance (failure factors)...")
        feat_imp_sorted = self._extract_feature_importance(model, feature_cols)
        failure_factors = self._analyze_failure_factors(df_train, feat_imp_sorted)
        
        # 打印特征重要性和失败因素结论
        print("\n🎯 Feature Importance Ranking (Impact on Success/Failure):")
        for idx, (feat, imp) in enumerate(feat_imp_sorted.items(), 1):
            print(f"   {idx}. {feat}: {imp}")
        
        # 新增：打印模型整体准确率
        print(f"\n📈 Random Forest Model Overall Accuracy:")
        print(f"   - Training Set Accuracy: {accuracy_dict['overall_accuracy']} ({accuracy_dict['accuracy_percentage']}%)")
        
        print("\n📖 Core Failure Factors Analysis:")
        for factor in failure_factors:
            print(f"   {factor}")
        
        print("\n✅ Training completed! Model + feature importance saved to model_output folder\n")

    # -------------------------- 步骤2：单文件预测核心逻辑 --------------------------
    def _predict_single_csv(self, csv_path):
        model = joblib.load(self.model_files["model"])
        scaler = joblib.load(self.model_files["scaler"])
        le_cat = joblib.load(self.model_files["le_cat"])
        le_country = joblib.load(self.model_files["le_country"])
        with open(self.model_files["feats"], "r") as f:
            feature_cols = json.load(f)
        with open(self.model_files["goal_median"], "r") as f:
            goal_median = json.load(f)

        try:
            df_pred = pd.read_csv(csv_path, encoding="utf-8")
        except:
            df_pred = pd.read_csv(csv_path, encoding="gbk")
        file_name = os.path.basename(csv_path)
        print(f"🔍 Processing prediction file: {file_name} (Total projects: {len(df_pred)})")        
        # 拆分id字段的填充逻辑
        fill_defaults = {
            "fx_rate": 1.0,
            "goal": 0,
            "country": "OTHER",
            "category": '{"name":"OTHER"}',
            "launched_at": 1577836800,
            "deadline": self._get_beijing_time_timestamp(),
            "name": "未知项目"
        }
        # 1. 处理其他字段（先填充默认值，再识别缺失）
        for col, val in fill_defaults.items():
            if col not in df_pred.columns:
                df_pred[col] = val
            # 先标记原始缺失值，再填充（避免填充后无法识别）
            df_pred[f"{col}_original"] = df_pred[col]
            df_pred[col] = df_pred[col].fillna(val)
        
        # 2. 单独处理id字段
        if "id" not in df_pred.columns:
            df_pred["id_original"] = np.nan
            df_pred["id"] = list(range(len(df_pred)))
        else:
            df_pred["id_original"] = df_pred["id"]
            df_pred["id"] = df_pred["id"].fillna(0)

        # 计算核心特征
        df_pred["goal_usd"] = df_pred["goal"] * df_pred["fx_rate"]
        df_pred["duration_days"] = df_pred.apply(
            lambda x: (self._safe_ts_convert(x["deadline"]) - self._safe_ts_convert(x["launched_at"])).days,
            axis=1
        )
        df_pred["main_category"] = df_pred["category"].apply(self._safe_parse_category)

        df_pred["cat_enc"] = df_pred["main_category"].apply(            
            lambda x: le_cat.transform([x])[0] if x in le_cat.classes_ else le_cat.transform(["OTHER"])[0]
        )
        df_pred["country_enc"] = df_pred["country"].apply(
            lambda x: le_country.transform([x])[0] if x in le_country.classes_ else le_country.transform(["OTHER"])[0]
        )

        # 分母最小设为0.01，避免除以0产生无穷大
        df_pred["goal_reasonable"] = df_pred.apply(
            lambda x: x["goal_usd"] / max(goal_median.get(x["main_category"], 1), 0.01), axis=1
        ).clip(0, 1000)
        df_pred["fund_sim"] = (1 / df_pred["goal_reasonable"]) * df_pred["cat_enc"]
        df_pred["time_ratio"] = df_pred["duration_days"].apply(lambda x: max(0.01, (x-7)/max(x, 1)))

        # 清理无穷大/异常值，避免模型报错
        X_pred = df_pred[feature_cols].fillna(0)
        X_pred = X_pred.replace([np.inf, -np.inf], 0)
        X_pred = X_pred.clip(-1e18, 1e18)
        
        X_scaled = scaler.transform(X_pred)
        pred_prob = model.predict_proba(X_scaled)[:, 1]
        pred_state = ["successful" if p > 0.5 else "failed" for p in pred_prob]

        # 计算M3所需的风险特征列
        df_pred["goal_ratio"] = df_pred["goal_reasonable"].clip(0, 50)
        df_pred["time_penalty"] = np.exp(df_pred["duration_days"].clip(lower=1) / 30) - 1
        df_pred["combined_risk"] = df_pred["goal_ratio"] * df_pred["time_penalty"]
        df_pred["urgency_score"] = 7 / df_pred["duration_days"].clip(lower=1)
        df_pred["_pred_failed"] = (np.array(pred_state) == "failed").astype(int)
        df_pred["category_risk"] = df_pred.groupby("main_category")["_pred_failed"].transform("mean")
        df_pred["country_factor"] = df_pred.groupby("country")["_pred_failed"].transform("mean")

       
        # 恢复原始字段用于缺失检测
        for field in CHECK_MISSING_FIELDS.keys():
            if f"{field}_original" in df_pred.columns:
                df_pred[field] = df_pred[f"{field}_original"]
                df_pred.drop(columns=[f"{field}_original"], inplace=True)
        
        # 生成缺失字段详情
        df_pred["数据缺失情况"] = df_pred.apply(self._get_missing_fields_detail, axis=1)

        result_df = pd.DataFrame({
            "Project ID": df_pred["id"],  
            "Project Name": df_pred["name"].apply(lambda x: str(x)[:30]+"..." if len(str(x))>30 else str(x)),            
            "Main Category": df_pred["main_category"],
            "Country": df_pred["country"],
            "Funding Goal (USD)": df_pred["goal_usd"].round(2),
            "Funding Duration (days)": df_pred["duration_days"],
            "goal_ratio": df_pred["goal_ratio"],
            "time_penalty": df_pred["time_penalty"],
            "category_risk": df_pred["category_risk"],
            "combined_risk": df_pred["combined_risk"],
            "country_factor": df_pred["country_factor"],
            "urgency_score": df_pred["urgency_score"],
            "Predicted Status": pred_state,
            "Success Probability": np.round(pred_prob, 4),
            "Missing Fields": df_pred["数据缺失情况"], 
            "Source File": file_name
        })

        # 单个预测结果保存到「单个文件预测结果」子文件夹
        single_result_path = os.path.join(self.pred_result_dir, f"prediction_result_{file_name}")
        result_df.to_csv(single_result_path, index=False, encoding="utf-8-sig")
        print(f"✅ Prediction completed for {file_name}! Saved to: {single_result_path}\n")
        return result_df

    # -------------------------- 步骤3：全量CSV预测 + 强制合并汇总 --------------------------
    def full_csv_predict(self):
        print("📌 Starting batch prediction (reading all CSV files in core folder)...\n")
        all_csv_files = [f for f in os.listdir(CORE_FOLDER) if f.endswith(".csv")]
        if not all_csv_files:
            raise ValueError(f"❌ No CSV files found in core folder: {CORE_FOLDER}!")
        
        # 过滤掉已生成的预测结果文件，只处理原始数据文件
        pred_files = [f for f in all_csv_files if not f.startswith("prediction_result") and not f.startswith("full_prediction_summary")]
        if not pred_files:
            print("⚠️ No original CSV files found for prediction (filtered result files)")
            return
        
        all_results = []
        processed_files = []
        for file in pred_files:
            file_path = os.path.join(CORE_FOLDER, file)
            try:
                result_df = self._predict_single_csv(file_path)                
                all_results.append(result_df)
                processed_files.append(file)
            except Exception as e:
                print(f"❌ Failed to process {file}: {str(e)}, skipped")
                continue
        
        # 强制合并所有成功处理的结果
        if all_results:
            # 合并所有结果
            merge_df = pd.concat(all_results, ignore_index=True)
            # 清理合并后的异常值（避免空值/特殊字符）
            merge_df = merge_df.fillna("Unknown")
            merge_df = merge_df.replace([np.inf, -np.inf], 0)
            
            # 生成带时间戳的汇总文件（避免覆盖），汇总文件仍保存在核心文件夹
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merge_result_path = os.path.join(CORE_FOLDER, f"full_prediction_summary_{timestamp}.csv")
            
            # 保存汇总文件（utf-8-sig兼容Excel）
            merge_df.to_csv(merge_result_path, index=False, encoding="utf-8-sig")
            
            # 统计缺失情况（新增）
            missing_stats = merge_df["Missing Fields"].value_counts()
            
            # ===================== 读取特征重要性和模型准确率，加入汇总日志 =====================
            print("\n🎯 Core Factors Affecting Project Failure (Extracted from Model):")
            if os.path.exists(self.model_files["feature_importance"]):
                with open(self.model_files["feature_importance"], "r") as f:
                    feat_imp_sorted = json.load(f)
                for idx, (feat, imp) in enumerate(feat_imp_sorted.items(), 1):
                    print(f"   {idx}. {feat} (Importance: {imp})")
            
            # 新增：打印模型准确率
            if os.path.exists(self.model_files["model_accuracy"]):
                with open(self.model_files["model_accuracy"], "r") as f:
                    accuracy_dict = json.load(f)
                print(f"\n📈 Random Forest Model Performance:")
                print(f"   - Overall Accuracy: {accuracy_dict['overall_accuracy']} ({accuracy_dict['accuracy_percentage']}%)")
            
            print("="*60)
            print(f"✅ Batch prediction completed!")
            print(f"📁 Single prediction results path: {self.pred_result_dir}")            
            print(f"📁 Full summary file path: {merge_result_path}")
            print(f"📊 Summary Statistics:")
            print(f"   - Processed files: {len(processed_files)}")
            print(f"   - Total projects: {len(merge_df)}")
            print(f"   - Predicted successful: {len(merge_df[merge_df['Predicted Status']=='successful'])}")
            print(f"   - Predicted failed: {len(merge_df[merge_df['Predicted Status']=='failed'])}")
            print(f"\n📋 Missing Fields Statistics:")
            for missing_type, count in missing_stats.head(10).items():
                print(f"   - {missing_type}: {count} records")
            print("="*60)
        else:
            print("❌ No files processed successfully, cannot generate summary")

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    processor = KSFullCSVProcessor()
    
    try:
        # 先训练（如果模型不存在）
        if not os.path.exists(processor.model_files["model"]):
            processor.full_csv_train()
        # 执行预测+汇总
        processor.full_csv_predict()
    except Exception as e:
        print(f"\n❌ Program execution error: {str(e)}")


# In[1]:


import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import roc_curve, auc

# 忽略警告
warnings.filterwarnings('ignore')

# ===================== 核心配置（动态相对路径） =====================
# 继承上面定义的CORE_FOLDER和PROJECT_ROOT（如果当前代码段单独执行时重新定义）
if 'CORE_FOLDER' not in dir():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    CORE_FOLDER = os.path.join(PROJECT_ROOT, "Kickstarter_Clean")

# PROCESSED_FILE 将在ROCBasedRiskAnalyzer中动态查找最新的预测结果文件
PROCESSED_FILE = None  # 将在使用时动态查找

# 特征名称映射
FEATURE_MAP = {
    "goal_ratio": "Goal vs Category Median (Ratio)",
    "time_penalty": "Time Penalty (Exponential)",
    "category_risk": "Category Failure Rate",
    "combined_risk": "Combined Risk (Goal x Time)",
    "country_factor": "Country Risk Factor",
    "urgency_score": "Urgency Score (7d Ratio)"
}

# 扩展有效状态（包含取消/暂停，提升数据量）
EXTENDED_VALID_STATES = ['failed', 'successful', 'Failed', 'Successful',
                         'canceled', 'Canceled', 'suspended', 'Suspended']

# 风险标签映射（取消/暂停视为高风险）
RISK_LABEL_MAP = {
    'failed': 1, 'Failed': 1,
    'canceled': 1, 'Canceled': 1,
    'suspended': 1, 'Suspended': 1,
    'successful': 0, 'Successful': 0
}

# ===================== 绘图配置 =====================
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ===================== 核心分析类 =====================
class ROCBasedRiskAnalyzer:
    def __init__(self):
        # 动态查找最新的预测结果文件
        self.processed_file = self._find_latest_processed_file(CORE_FOLDER)
        self.df = None
        self.raw_df = None  # 保存原始全量数据
        self.rules = {}     # ROC最优阈值
        self.auc_results = {}  # 特征AUC值
    
    def _find_latest_processed_file(self, core_folder):
        """查找core_folder中最新的full_prediction_summary_*.csv文件"""
        try:
            if not os.path.exists(core_folder):
                print(f"⚠️ 目录不存在: {core_folder}")
                return None
            
            # 查找所有full_prediction_summary_*.csv文件
            csv_files = [f for f in os.listdir(core_folder) 
                        if f.startswith("full_prediction_summary_") and f.endswith(".csv")]
            
            if not csv_files:
                print(f"⚠️ 在 {core_folder} 中未找到预测结果文件")
                return None
            
            # 按修改时间排序，获取最新的
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(core_folder, x)), reverse=True)
            latest_file = os.path.join(core_folder, csv_files[0])
            print(f"✓ 找到最新预测结果文件: {csv_files[0]}")
            return latest_file
        except Exception as e:
            print(f"⚠️ 查找预测结果文件失败: {str(e)}")
            return None

    # -------------------------- 加载预处理文件 --------------------------
    def load_processed_data(self):
        # 检查是否找到了预测结果文件
        if self.processed_file is None:
            raise FileNotFoundError(
                f"❌ 无法找到预测结果文件！\n"
                f"   请确保已在 {CORE_FOLDER} 中生成 full_prediction_summary_*.csv 文件\n"
                f"   需先运行 KSFullCSVProcessor 进行特征工程与模型训练"
            )
        
        print(f"📥 加载预处理文件: {self.processed_file}")
        
        # 检查文件是否存在
        if not os.path.exists(self.processed_file):
            raise FileNotFoundError(
                f"❌ 预处理文件不存在！路径: {self.processed_file}"
            )
        
        # 读取文件
        try:
            self.raw_df = pd.read_csv(self.processed_file, encoding="utf-8-sig")
        except Exception as e:
            raise ValueError(f"❌ 读取文件失败: {str(e)}")
        
        # 打印原始数据统计
        print(f"📈 原始文件总项目数: {len(self.raw_df)}")
        print(f"📋 原始数据状态分布:")
        # 仅修改：将state改为Predicted Status
        state_counts = self.raw_df['Predicted Status'].value_counts(dropna=False)
        for state, count in state_counts.items():
            print(f"   - {state}: {count} 条")
        
        # 筛选扩展有效状态
        # 仅修改：将state改为Predicted Status
        self.df = self.raw_df[self.raw_df['Predicted Status'].isin(EXTENDED_VALID_STATES)].copy()
        
        # 构建风险标签
        # 仅修改：将state改为Predicted Status
        self.df['label'] = self.df['Predicted Status'].map(RISK_LABEL_MAP)
        self.df = self.df.dropna(subset=['label'])
        self.df['label'] = self.df['label'].astype(int)
        
        # 检查核心字段
        required_cols = ['goal_usd', 'duration_days', 'main_category', 'country']
        # 仅新增：映射实际列名到代码预期字段名
        col_mapping = {
            'Funding Goal (USD)': 'goal_usd',
            'Funding Duration (days)': 'duration_days',
            'Main Category': 'main_category',
            'Country': 'country'
        }
        for old_col, new_col in col_mapping.items():
            if old_col in self.df.columns and new_col not in self.df.columns:
                self.df.rename(columns={old_col: new_col}, inplace=True)
        
        # 检查核心字段
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"❌ 缺少关键字段: {missing_cols}")
        
        # 数据清理
        self.df['duration'] = self.df['duration_days'].clip(1, 180)
        self.df['goal_usd'] = self.df['goal_usd'].clip(0, 1e6)
        
        print(f"✅ 预处理文件加载完成！")
        print(f"   📊 扩展后有效项目数: {len(self.df)} ")
        print(f"   📌 高风险项目（失败/取消/暂停）: {len(self.df[self.df['label']==1])}")
        print(f"   📌 低风险项目（成功）: {len(self.df[self.df['label']==0])}")
        print(f"   📌 品类数: {self.df['main_category'].nunique()}, 国家数: {self.df['country'].nunique()}")

    # -------------------------- 构建函数特征 --------------------------
    def build_functional_features(self):
        print("\n🔧 构建函数关系特征...")
        
        # A. 目标倍率（项目目标/同品类中位数）
        cat_medians = self.df.groupby('main_category')['goal_usd'].median()
        self.df['goal_ratio'] = self.df.apply(
            lambda x: x['goal_usd'] / max(cat_medians.get(x['main_category'], 1), 0.01),
            axis=1
        ).clip(0, 50)

        # B. 时间惩罚指数（e^(周期/30)-1）
        self.df['time_penalty'] = np.exp(self.df['duration'] / 30) - 1

        # C. 品类风险系数（品类高风险率）
        cat_fail_rates = self.df.groupby('main_category')['label'].mean()
        self.df['category_risk'] = self.df['main_category'].map(cat_fail_rates)

        # D. 组合风险（目标倍率×时间惩罚）
        self.df['combined_risk'] = self.df['goal_ratio'] * self.df['time_penalty']

        # E. 国家风险因子（国家高风险率）
        country_fail_rates = self.df.groupby('country')['label'].mean()
        self.df['country_factor'] = self.df['country'].map(country_fail_rates)

        # F. 紧迫感分数（7/周期）
        self.df['urgency_score'] = 7 / self.df['duration']

        # 验证特征
        built_features = ['goal_ratio', 'time_penalty', 'category_risk', 
                          'combined_risk', 'country_factor', 'urgency_score']
        for feat in built_features:
            if feat not in self.df.columns:
                raise ValueError(f"❌ 特征 {feat} 构建失败")
        print(f"✅ 6个函数特征构建完成！")

    # -------------------------- ROC+AUC找最优阈值 --------------------------
    def _get_optimal_threshold(self, feature):
        y_true = self.df['label']
        y_score = self.df[feature]
        
        # 计算ROC参数
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc_value = auc(fpr, tpr)
        
        # 约登指数找最优阈值
        j_scores = tpr - fpr
        best_idx = j_scores.argmax()
        best_threshold = thresholds[best_idx]
        best_tpr = tpr[best_idx]
        best_fpr = fpr[best_idx]
        
        return best_threshold, auc_value, best_tpr, best_fpr

    # -------------------------- 可视化+规则提取 --------------------------
    def analyze_and_extract_rules(self):
        print("\n📊 用ROC+AUC分析风险规则...")
        features = ['goal_ratio', 'time_penalty', 'category_risk', 
                    'combined_risk', 'country_factor', 'urgency_score']
        
        # 1. 绘制ROC汇总图
        plt.figure(figsize=(10, 8))
        for feat in features:
            y_true = self.df['label']
            y_score = self.df[feat]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{FEATURE_MAP[feat]} (AUC={auc_val:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC=0.5)')
        plt.xlabel('False Positive Rate (FPR) - 误判率', fontsize=11)
        plt.ylabel('True Positive Rate (TPR) - 识别率', fontsize=11)
        plt.title('ROC Curves for All Risk Features', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        roc_path = os.path.join(CORE_FOLDER, "roc_curve_summary.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC汇总图已保存: {roc_path}")

        # 2. 绘制特征分布+提取规则
        plt.figure(figsize=(18, 10))
        for i, feat in enumerate(features):
            best_thresh, auc_val, tpr, fpr = self._get_optimal_threshold(feat)
            self.rules[feat] = best_thresh
            self.auc_results[feat] = auc_val
            
            # 绘制高风险项目分布
            high_risk_df = self.df[self.df['label'] == 1]
            plt.subplot(2, 3, i+1)
            sns.histplot(high_risk_df[feat], kde=True, color='#e74c3c', bins=30, alpha=0.8)
            plt.axvline(best_thresh, color='#2c3e50', linestyle='--', linewidth=2,
                        label=f'Optimal Threshold: {best_thresh:.2f}\nAUC: {auc_val:.3f}\nTPR: {tpr:.2f}, FPR: {fpr:.2f}')
            plt.title(f"High-Risk Projects: {FEATURE_MAP[feat]}", fontsize=12)
            plt.xlabel(FEATURE_MAP[feat], fontsize=10)
            plt.ylabel("Number of High-Risk Projects", fontsize=10)
            plt.legend(fontsize=9)
            plt.grid(alpha=0.3)

            # 打印规则详情
            self._print_rule_detail(feat, best_thresh, auc_val, tpr, fpr)

        # 保存特征分布图
        dist_path = os.path.join(CORE_FOLDER, "risk_feature_distribution.png")
        plt.tight_layout()
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 特征分布图已保存: {dist_path}")

        # 打印AUC汇总
        print("\n📈 各特征AUC值汇总:")
        print("   " + "-"*60)
        print(f"   {'特征':<25} {'AUC值':<10} {'等级':<10}")
        print("   " + "-"*60)
        for feat in features:
            auc_val = self.auc_results[feat]
            level = "优秀" if auc_val >= 0.8 else "良好" if auc_val >= 0.7 else "一般"
            print(f"   {FEATURE_MAP[feat]:<25} {auc_val:.3f}      {level}")
        print("   " + "-"*60)

    def _print_rule_detail(self, feat, thresh, auc_val, tpr, fpr):
        """打印单个特征的规则详情"""
        print(f"\n--- {FEATURE_MAP[feat]} ---")
        print(f"   🎯 最优阈值: {thresh:.2f}")
        print(f"   📊 模型性能: AUC={auc_val:.3f}, 识别率={tpr:.2f}, 误判率={fpr:.2f}")
        print(f"   🔴 风险规则: ", end="")
        
        if feat == 'goal_ratio':
            print(f"当 (项目目标 / 同品类中位数) > {thresh:.2f} 倍时，判定为高风险")
        elif feat == 'time_penalty':
            days = np.log(thresh + 1) * 30
            print(f"当时间惩罚指数 > {thresh:.2f}（约{days:.0f}天）时，判定为高风险")
        elif feat == 'category_risk':
            print(f"当品类历史高风险率 > {thresh:.2%} 时，判定为高风险")
        elif feat == 'combined_risk':
            print(f"当 (目标倍率 × 时间惩罚指数) > {thresh:.2f} 时，判定为高风险")
        elif feat == 'country_factor':
            print(f"当国家历史高风险率 > {thresh:.2%} 时，判定为高风险")
        elif feat == 'urgency_score':
            days = 7 / thresh
            print(f"当紧迫感分数 > {thresh:.2f}（约{days:.0f}天）时，判定为高风险")

    # -------------------------- 导出风险规则代码（修复缩进+变量） --------------------------
    def export_risk_rules(self):
        print("\n💾 导出风险规则代码...")
        
        # 计算基准数据
        cat_medians = self.df.groupby('main_category')['goal_usd'].median().to_dict()
        cat_risk_rates = self.df.groupby('main_category')['label'].mean().to_dict()
        
        # 生成规则代码（统一缩进，避免格式错误）
        rule_code = f"""import numpy as np

# ===================== 众筹项目失败风险规则（ROC+AUC优化） =====================
# 数据来源：{PROCESSED_FILE}
# 扩展：失败/取消/暂停=高风险，成功=低风险

# 基准数据
CATEGORY_MEDIAN_GOAL = {cat_medians}
CATEGORY_RISK_RATE = {cat_risk_rates}
OPTIMAL_THRESHOLDS = {self.rules}
FEATURE_AUC = {self.auc_results}

def judge_project_risk(project_data):
    risk_reasons = []
    is_high_risk = False

    # 1. 目标倍率风险
    median_goal = CATEGORY_MEDIAN_GOAL.get(project_data['main_category'], 5000)
    goal_ratio_val = project_data['goal_usd'] / max(median_goal, 0.01)
    if goal_ratio_val > OPTIMAL_THRESHOLDS['goal_ratio']:
        risk_reasons.append(f"目标倍率过高（{{goal_ratio_val:.2f}} > 阈值{{OPTIMAL_THRESHOLDS['goal_ratio']:.2f}}）")
        is_high_risk = True

    # 2. 周期风险
    time_penalty_val = np.exp(project_data['duration_days'] / 30) - 1
    if time_penalty_val > OPTIMAL_THRESHOLDS['time_penalty']:
        risk_reasons.append(f"周期风险过高（惩罚指数{{time_penalty_val:.2f}} > 阈值{{OPTIMAL_THRESHOLDS['time_penalty']:.2f}}）")
        is_high_risk = True

    # 3. 组合风险
    combined_risk_val = goal_ratio_val * time_penalty_val
    if combined_risk_val > OPTIMAL_THRESHOLDS['combined_risk']:
        risk_reasons.append(f"组合风险过高（{{combined_risk_val:.2f}} > 阈值{{OPTIMAL_THRESHOLDS['combined_risk']:.2f}}）")
        is_high_risk = True

    # 4. 品类风险
    cat_risk_val = CATEGORY_RISK_RATE.get(project_data['main_category'], 0.5)
    if cat_risk_val > OPTIMAL_THRESHOLDS['category_risk']:
        risk_reasons.append(f"品类风险过高（高风险率{{cat_risk_val:.2%}} > 阈值{{OPTIMAL_THRESHOLDS['category_risk']:.2%}}）")
        is_high_risk = True

    if not risk_reasons:
        risk_reasons.append("无高风险因素，风险可控")
    return is_high_risk, risk_reasons

# 使用示例
if __name__ == "__main__":
    test_project = {{
        'goal_usd': 15000,
        'duration_days': 60,
        'main_category': 'Technology',
        'country': 'US'
    }}
    risk_result, reasons = judge_project_risk(test_project)
    print(f"项目风险评估：{{'高风险' if risk_result else '低风险'}}")
    print("风险原因：")
    for i, reason in enumerate(reasons, 1):
        print(f"  {{i}}. {{reason}}")
        """
        
        # 保存规则文件
        rule_path = os.path.join(CORE_FOLDER, "crowdfunding_risk_rules.py")
        with open(rule_path, "w", encoding="utf-8") as f:
            f.write(rule_code)
        
        print(f"✅ 规则代码已保存至: {rule_path}")
        print(f"   📌 使用：导入 judge_project_risk 函数即可判断风险")

    # -------------------------- 运行全流程 --------------------------
    def run_full_analysis(self):
        try:
            self.load_processed_data()
            self.build_functional_features()
            self.analyze_and_extract_rules()
            self.export_risk_rules()
            print("\n🎉 风险规则分析全流程完成！")
        except Exception as e:
            print(f"\n❌ 分析出错: {str(e)}")
            import traceback
            traceback.print_exc()

# ===================== 主执行入口 =====================
if __name__ == "__main__":
    analyzer = ROCBasedRiskAnalyzer()
    analyzer.run_full_analysis()

