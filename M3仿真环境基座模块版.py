"""
M3仿真系统 - 一键运行版本
整合了配置系统和主程序功能
只需运行此文件即可执行M3仿真
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
import os
import json
import shutil
from io import StringIO
import sys
from datetime import datetime
from typing import List, Dict, Any

# 设置pandas选项以避免FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# ==========================================
# 配置系统
# ==========================================
class M3Config:
    def __init__(self):
        # 核心数据文件夹路径 - 一键修改即可
        self.core_folder = "Kickstarter_2025-12-18T03_20_24_296Z"
        self.datasets_dir = "datasets"
        self.output_dir = "Kickstarter_Clean"
        
        # 输出文件夹
        self.m3_output_folder = "m3_results"
        
        # 模型文件夹
        self.model_folder = "m3_models"
        
        # 随机种子
        self.random_seed = 42
        
        # 默认风险等级
        self.default_risk_level = 0.5
        
        # 风险等级列表（用于多项目仿真）
        self.risk_levels = [0.3, 0.5, 0.7]
        
        # 策略列表
        self.strategies = ["balanced", "marketing", "product"]
        
        # 验证路径是否存在
        self._validate_paths()
    
    def _validate_paths(self):
        """验证路径是否存在"""
        if not os.path.exists(self.core_folder):
            raise FileNotFoundError(f"数据路径不存在: {self.core_folder}")
        
        # 创建必要的目录
        for folder in [self.datasets_dir, self.output_dir, self.m3_output_folder, self.model_folder]:
            os.makedirs(folder, exist_ok=True)
            print(f"创建文件夹: {folder}")
    
    def update_data_path(self, new_path):
        """更新数据路径"""
        self.core_folder = new_path
        self._validate_paths()
        print(f"✓ 数据路径已更新为: {new_path}")
    
    def get_data_path(self):
        """获取当前数据路径"""
        return self.core_folder
    
    def get_cleaned_data_path(self):
        """获取清洗后数据的路径"""
        return os.path.join(self.output_dir, "kickstarter_cleaned.csv")
    
    def __str__(self):
        return f"M3配置 - 数据路径: {self.core_folder}, 清洗输出: {self.output_dir}, 仿真输出: {self.m3_output_folder}"

# ==========================================
# M2 特征工程处理器
# ==========================================
class M2Preprocessor:
    """
    M2特征工程处理器 - 实现从原始数据到特征向量的转换
    根据您提供的数据结构设计
    """
    def __init__(self):
        # 初始化编码器（基于您提供的数据结构）
        self.category_encoder = self._create_category_encoder()
        self.country_encoder = self._create_country_encoder()
        self.feature_cols = [
            "duration_days", "cat_enc", "country_enc", 
            "goal_usd", "fund_sim", "time_ratio"
        ]
        
        # 基于您提供的数据统计初始化中位数
        self.goal_medians = {
            "Art": 2600, "Comics": 3000, "Dance": 1500, "Design": 4000,
            "Fashion": 3500, "Food": 2000, "Film & Video": 2500, "Games": 8000,
            "Journalism": 3000, "Music": 5000, "Photography": 2500, "Technology": 10000,
            "Theater": 3000, "Publishing": 4000, "Crafts": 2000, "OTHER": 10000
        }

    def _create_category_encoder(self):
        """创建类别编码器 - 基于您提供的主分类"""
        from sklearn.preprocessing import LabelEncoder
        categories = [
            "Art", "Comics", "Dance", "Design", "Fashion", "Food", 
            "Film & Video", "Games", "Journalism", "Music", "Photography", 
            "Technology", "Theater", "Publishing", "Crafts", "OTHER"
        ]
        encoder = LabelEncoder()
        encoder.fit(categories)
        return encoder

    def _create_country_encoder(self):
        """创建国家编码器 - 基于您提供的国家分布"""
        from sklearn.preprocessing import LabelEncoder
        countries = ["US", "GB", "CA", "AU", "DE", "OTHER"]
        encoder = LabelEncoder()
        encoder.fit(countries)
        return encoder

    def _safe_encode(self, encoder, value, default_value=0):
        """安全编码，处理未知标签"""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # 处理未知标签，返回默认值
            return default_value

    def _safe_ts_convert(self, ts):
        """时间戳安全转换"""
        if pd.isna(ts) or str(ts).lower() in ["nan", "", "none"]:
            return datetime(2020, 1, 1)
        try:
            ts_int = int(float(ts))
            # 处理毫秒级时间戳
            ts_int = ts_int // 1000 if ts_int > 1e12 else ts_int
            return datetime.fromtimestamp(ts_int)
        except:
            return datetime(2020, 1, 1)

    def _safe_parse_category(self, cat_str):
        """类别 JSON 安全解析"""
        if pd.isna(cat_str) or str(cat_str).lower() in ["nan", "", "none"]:
            return "OTHER"
        try:
            # 假设输入是 JSON 字符串
            if isinstance(cat_str, str):
                cat_dict = json.loads(cat_str.replace("'", '"').strip())
            elif isinstance(cat_str, dict):
                cat_dict = cat_str
            else:
                return "OTHER"
            return cat_dict.get("parent_name", cat_dict.get("name", "OTHER")).strip()
        except:
            return "OTHER"

    def process(self, data):
        """
        核心方法：将原始数据转换为特征向量
        Args:
            data: 原始数据（可以是字典、Series或DataFrame）
        Returns:
            numpy.ndarray: 标准化的特征向量
        """
        # 1. 将数据转为 DataFrame (方便利用 pandas 进行批量处理)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = data.to_frame().T  # 将Series转换为单行DataFrame
        else:
            df = data.copy()

        # 2. 缺失值与基础字段填充 (M2 逻辑)
        fill_defaults = {
            "goal": 0, "pledged": 0, "backers_count": 0, 
            "fx_rate": 1.0, "country": "OTHER", "category": '{"name":"OTHER"}',
            "launched_at": 1577836800, "deadline": int(datetime.now().timestamp())
        }
        for col, val in fill_defaults.items():
            if col not in df.columns: df[col] = val
            df[col] = df[col].fillna(val)

        # 3. 特征衍生 (Feature Derivation)
        # 3.1 计算目标金额（USD）
        df["goal_usd"] = df["goal"]
        
        # 3.2 计算持续天数
        df["duration_days"] = df.apply(
            lambda x: (self._safe_ts_convert(x["deadline"]) - self._safe_ts_convert(x["launched_at"])).days,
            axis=1
        )

        # 3.3 解析主类别
        df["main_category"] = df["category"].apply(self._safe_parse_category)

        # 4. 特征编码 (Encoding)
        df["cat_enc"] = df["main_category"].apply(
            lambda x: self._safe_encode(self.category_encoder, x)
        )
        df["country_enc"] = df["country"].apply(
            lambda x: self._safe_encode(self.country_encoder, x)
        )

        # 5. 高级特征计算
        # 确保分母不为 0
        median_goal = self.goal_medians.get(df["main_category"].iloc[0], 10000)
        median_goal = max(median_goal, 0.01) 
        
        df["goal_reasonable"] = (df["goal_usd"] / median_goal).clip(0, 1000)
        df["fund_sim"] = (1 / df["goal_reasonable"].replace(0, 0.01)) * df["cat_enc"]
        
        df["duration_days_safe"] = df["duration_days"].apply(lambda x: max(x, 1)) # 避免除以0
        df["time_ratio"] = df["duration_days"].apply(lambda x: max(0.01, (x - 7) / max(x, 1)))

        # 6. 选择特征列
        features_df = df[self.feature_cols].fillna(0)
        # 替换无穷大值
        features_df = features_df.replace([np.inf, -np.inf], 0)
        # 截断极端值
        features_df = features_df.clip(-1e18, 1e18)
        
        # 7. 标准化（使用简单的标准化，因为M2阶段没有训练好的scaler）
        features_df = (features_df - features_df.mean()) / (features_df.std() + 1e-8)
        
        # 返回展平的 numpy 向量 (作为环境的 Observation)
        return features_df.values.flatten().astype(np.float32)

# ==========================================
# M3 Core Class: 仿真环境基座
# ==========================================
class StartupEnv(gym.Env):
    """
    符合 Gymnasium 标准的初创项目仿真环境。
    直接使用M2特征向量作为输入，完成M3仿真。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(StartupEnv, self).__init__()

        # 特征维度（基于M2特征工程）
        self.feature_dim = 6  # duration_days, cat_enc, country_enc, goal_usd, fund_sim, time_ratio

        # 定义状态空间 (Observation Space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32
        )

        # 定义动作空间 (Action Space)
        # 在这个评估系统中，Agent 的动作是"更新状态"
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_dim,), dtype=np.float32
        )

        self.current_state_vector = None
        self.steps_taken = 0
        self.max_steps = 12 # 模拟12个月的变化

    def reset(self, *, seed=None, options=None):
        """
        重置环境。
        关键：通过 options 传入M2特征向量。
        """
        super().reset(seed=seed)
        self.steps_taken = 0

        # 获取M2特征向量
        feature_vector = options.get("feature_vector") if options else None
        if feature_vector is None:
            # 如果没有提供特征向量，使用默认的 Mock 向量
            print("⚠️ [Env] 未提供特征向量，使用默认 Mock 向量进行重置。")
            feature_vector = np.zeros(self.feature_dim, dtype=np.float32)

        # 设置当前状态为M2特征向量
        self.current_state_vector = feature_vector.astype(np.float32)
        
        info = {"status": "Initialized with M2 feature vector"}
        return self.current_state_vector, info

    def step(self, action):
        """
        环境交互步。
        Args:
            action: 外部 Agent 计算好的新的状态向量。
        """
        self.steps_taken += 1
        
        # 在这个架构中，环境信任 Agent 的计算结果，直接更新状态
        self.current_state_vector = action.astype(np.float32)

        # 判断终止条件 (例如达到最大模拟时长)
        truncated = (self.steps_taken >= self.max_steps)
        terminated = False # 需要根据具体业务逻辑定义成功/失败条件

        # 奖励定义 (需要根据具体目标设定，这里暂设为 0)
        reward = 0.0
        
        info = {"steps": self.steps_taken}

        return self.current_state_vector, reward, terminated, truncated, info

    def render(self):
        """简单打印当前状态向量"""
        print(f"Step {self.steps_taken} | Current State Vector: {self.current_state_vector}")

# ==========================================
# M3 主函数
# ==========================================
def run_m3_simulation():
    """
    运行M3仿真系统 - 完整的M2+M3流程
    1. 数据清洗
    2. 读取清洗后的数据
    3. 完成M2特征工程
    4. 基于M2特征向量完成M3仿真
    """
    print("="*60)
    print("M3仿真系统 - M2+M3完整流程启动")
    print("="*60)
    
    try:
        # 初始化配置
        config = M3Config()
        print(f"使用配置: {config}")
        
        # 1. 数据清洗
        print("开始数据清洗...")
        # 创建日志捕获器
        class LogCapture:
            def __init__(self, log_file):
                self.log_file = log_file
                self.log_content = StringIO()
                self.terminal = sys.stdout

            def write(self, message):
                self.log_content.write(message)
                self.terminal.write(message)

            def flush(self):
                self.terminal.flush()

            def save_log(self):
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write(self.log_content.getvalue())
        
        # 启用日志记录
        log_capture = LogCapture(os.path.join(config.output_dir, "cleaning_log.txt"))
        sys.stdout = log_capture
        
        # 检查数据路径
        data_path = config.core_folder
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        
        # 读取数据集
        if os.path.isfile(data_path) and data_path.endswith(".csv"):
            # 如果是单个 CSV 文件
            df = pd.read_csv(data_path)
            print(f"读取单个 CSV 文件: {data_path}")
        elif os.path.isdir(data_path):
            # 如果是文件夹，读取所有 CSV 文件并合并
            csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(f"文件夹 {data_path} 中没有找到 CSV 文件。")
            df_list = []
            for file in csv_files:
                file_path = os.path.join(data_path, file)
                df_temp = pd.read_csv(file_path)
                df_list.append(df_temp)
                print(f"读取 CSV 文件: {file_path}")
            df = pd.concat(df_list, ignore_index=True)
            print(f"合并了 {len(csv_files)} 个 CSV 文件，总行数: {len(df)}")
        else:
            raise ValueError(f"错误：{data_path} 不是有效的 CSV 文件或文件夹。")
        
        # 开始清洗日志
        print("=== 数据清洗日志 ===")
        
        # 1. 数据基本信息
        print(f"数据行数: {df.shape[0]}")
        print(f"数据列数: {df.shape[1]}")
        print(f"所有字段名称: {list(df.columns)}")
        print("\n前几条数据示例:")
        print(df.head())
        
        # 2. 数据详细信息
        print("\n数据详细信息（字段数据类型、非空值数量）:")
        df.info()
        
        # 3. 数值型字段描述性统计
        print("\n数值型字段描述性统计（均值、最值、分位数等）:")
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("无数值型字段。")
        
        # 步骤 2：重复值处理
        print("\n=== 步骤 2：重复值处理 ===")
        
        # 查找用于去重的列（优先使用 id 字段）
        id_column = None
        for col_name in ["id", "ID", "project_id", "name"]:
            if col_name in df.columns:
                id_column = col_name
                break
        
        if id_column:
            duplicate_count = df.duplicated(subset=[id_column]).sum()
            print(f"重复记录总数（基于 {id_column}）: {duplicate_count}")
            if duplicate_count > 0:
                df = df.drop_duplicates(subset=[id_column], keep="first")
                print("已执行去重操作，保留第一条重复记录。")
            else:
                print("无重复记录。")
        else:
            print("警告：未找到用于去重的唯一标识列（id/ID/project_id/name），跳过去重操作。")
            print(f"数据集列名: {list(df.columns)}")
        
        print(f"去重后数据行数: {df.shape[0]}")
        
        # 步骤 3：缺失值处理
        print("\n=== 步骤 3：缺失值处理 ===")
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_sorted = missing_percent.sort_values(ascending=False)
        print("各字段缺失值比例（从高到低）:")
        print(missing_sorted)
        
        # 定义核心字段（根据实际业务理解和字段列表）
        # id: 唯一标识，goal: 融资目标，state: 项目状态，launched_at/deadline: 时间
        available_core_fields = ["id", "goal", "state", "launched_at", "deadline"]
        core_fields = [col for col in available_core_fields if col in df.columns]
        
        if not core_fields:
            print("警告：未找到定义的核心字段，跳过核心字段缺失值删除。")
        else:
            print(f"\n核心字段: {core_fields}")
            initial_rows = len(df)
            df = df.dropna(subset=core_fields)
            print(
                f"删除核心字段缺失的记录后，数据行数: {len(df)} (删除 {initial_rows - len(df)} 行)"
            )
        
        
        # 处理非核心字符串字段缺失：填充"未知"
        string_fields = df.select_dtypes(include=["object"]).columns
        non_core_string = [col for col in string_fields if col not in core_fields]
        for col in non_core_string:
            df[col] = df[col].fillna("未知")
        
        # 处理非核心数值型字段缺失：填充中位数
        numeric_fields = df.select_dtypes(include=["number"]).columns
        non_core_numeric = [col for col in numeric_fields if col not in core_fields]
        for col in non_core_numeric:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"字段 {col} 缺失值填充中位数: {median_val}")
        
        total_missing_after = df.isnull().sum().sum()
        print(f"处理后总缺失值数量: {total_missing_after}")
        
        # 步骤 4：异常值处理
        print("\n=== 步骤 4：异常值处理 ===")
        
        # 处理数值型字段异常
        # 重点字段：goal（目标）、pledged（已筹）、backers_count（支持者数）、converted_pledged_amount（已筹本币）
        numeric_cols = ["goal", "pledged", "backers_count", "converted_pledged_amount"]
        for col in numeric_cols:
            if col in df.columns:
                # 剔除负数或0
                initial_count = len(df)
                df = df[df[col] > 0]
                print(
                    f"字段 {col} 剔除负数或0后，剩余行数: {len(df)} (剔除 {initial_count - len(df)})"
                )
        
                # 箱线图法剔除极端值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                print(
                    f"字段 {col} 剔除极端值（IQR法）: {outliers_count} 个，"
                    f"剩余行数: {len(df)}"
                )
        
        # 处理分类字段异常
        # state 字段合法值参考 Kickstarter 官方
        valid_states = ["successful", "failed", "canceled", "live", "suspended"]
        if "state" in df.columns:
            invalid_states = df[~df["state"].isin(valid_states)]
            invalid_count = len(invalid_states)
            print(f"异常分类值（state字段）数量: {invalid_count}")
            if invalid_count > 0:
                df = df[df["state"].isin(valid_states)]
                print("已删除异常分类记录。")
        print(f"异常值处理后数据行数: {df.shape[0]}")
        
        # 步骤 5：数据格式标准化与字段衍生
        print("\n=== 步骤 5：数据格式标准化与字段衍生 ===")
        
        # 保持Unix时间戳格式，不进行日期转换
        print("保持 launched_at 和 deadline 字段的原始Unix时间戳格式。")
        
        # 分类字段格式统一
        categorical_cols = ["category", "state", "country", "currency"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                print(f"字段 {col} 转为小写并去除空格。")
        
        # 字段命名标准化
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        print("所有列名转为小写，空格替换为下划线。")
        
        # 衍生新字段：项目持续天数
        # （基于Unix时间戳）
        if "deadline" in df.columns and "launched_at" in df.columns:
            df["duration_days"] = (df["deadline"] - df["launched_at"]) / 86400  # 86400秒 = 1天
            df["duration_days"] = df["duration_days"].round().astype(int)  # 四舍五入为整数天数
            print("衍生字段 'duration_days'（项目持续天数，基于Unix时间戳计算）。")
        
        print("格式标准化结果示例:")
        print(df.head())
        
        # 保存清洗后的数据
        os.makedirs(os.path.dirname(config.get_cleaned_data_path()), exist_ok=True)
        df.to_csv(config.get_cleaned_data_path(), index=False)
        print(f"\n清洗完成！数据集已保存到 {config.get_cleaned_data_path()}")
        
        # 保存日志
        log_capture.save_log()
        print(f"清洗日志已保存到 {log_capture.log_file}")
        
        # 恢复 stdout
        sys.stdout = log_capture.terminal
        
        # 2. 读取清洗后的数据
        cleaned_data_path = config.get_cleaned_data_path()
        print(f"\n正在从 {cleaned_data_path} 读取清洗后的数据...")
        full_data = pd.read_csv(cleaned_data_path)
        print(f"成功加载 {len(full_data)} 个项目")
        
        # 3. 初始化M2预处理器
        m2_preprocessor = M2Preprocessor()
        
        # 4. 对每个项目进行M2特征工程
        print("\n开始M2特征工程...")
        feature_vectors = []
        for idx, row in full_data.iterrows():
            try:
                feature_vector = m2_preprocessor.process(row)
                feature_vectors.append({
                    'original_data': row.to_dict(),
                    'feature_vector': feature_vector,
                    'project_id': row.get('id', idx)
                })
                if (idx + 1) % 100 == 0:  # 每100个项目显示进度
                    print(f"  已处理 {idx+1}/{len(full_data)} 个项目")
            except Exception as e:
                print(f"  处理项目 {row.get('id', idx)} 时发生错误: {str(e)}")
                # 添加额外的错误处理
                print(f"  项目数据详情: {row[['id', 'name', 'category', 'country']].to_dict()}")
                continue
        
        if not feature_vectors:
            raise ValueError("没有成功处理任何项目数据")
        
        print(f"\nM2特征工程完成，成功处理 {len(feature_vectors)} 个项目")
        
        # 5. 初始化M3仿真环境
        env = StartupEnv()
        
        # 6. 基于M2特征向量运行M3仿真
        print("\n开始M3仿真...")
        for i, project_data in enumerate(feature_vectors[:5]):  # 先处理前5个项目作为演示
            print(f"\n--- 仿真项目 {i+1}/{min(5, len(feature_vectors))} ---")
            print(f"项目名称: {project_data['original_data'].get('name', '未知')}")
            print(f"目标金额: {project_data['original_data'].get('goal', 0)}")
            
            # 重置环境，传入M2特征向量
            obs, info = env.reset(options={"feature_vector": project_data['feature_vector']})
            print(f"初始特征向量 (Observation): \n{obs}")
            print("-" * 30)
            
            done = False
            step_count = 0
            while not done:
                step_count += 1
                # 模拟Agent动作：这里简单地随机生成一个新状态向量
                action_new_state = env.action_space.sample()
                
                print(f"Step {step_count}: Agent 提交新状态...")
                obs, reward, terminated, truncated, info = env.step(action_new_state)
                env.render()
                
                done = terminated or truncated
            
            print(f"项目 {i+1} 仿真完成")
        
        print("\n--- M3仿真全部完成 ---")
        env.close()
        
        print("\n" + "="*60)
        print("M3仿真系统 - M2+M3完整流程执行完成")
        print("="*60)
        
    except Exception as e:
        print(f"\nM3仿真系统执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

# ==========================================
# 一键运行入口
# ==========================================
if __name__ == "__main__":
    run_m3_simulation()