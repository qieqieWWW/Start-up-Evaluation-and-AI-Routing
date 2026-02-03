#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#用于随机森林建模的全部特征：feature_cols = ["duration_days", "cat_enc", "country_enc", "goal_usd", "fund_sim", "time_ratio"]


# In[ ]:


#1. 工具函数（特征计算依赖的基础函数）
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
            return "OTHER"


# In[ ]:


#2. 训练阶段特征计算 + 清理（核心逻辑）
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
        # 训练前清理异常值
        X = df_train[feature_cols].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        X = X.clip(-1e18, 1e18)


# In[ ]:


#3. 预测阶段特征计算 + 清理（核心逻辑）
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


# In[ ]:


#用于规则提取的全部特征：goal_ratio，time_penalty，category_risk，combined_risk，country_factor，urgency_score


# In[ ]:


#1. 数据加载与基础清理（特征依赖的前置处理）


# In[ ]:


def load_processed_data(self):
    print(f"📥 加载预处理文件: {self.processed_file}")
    
    # 检查文件是否存在
    if not os.path.exists(self.processed_file):
        raise FileNotFoundError(
            f"❌ 预处理文件不存在！路径: {self.processed_file}\n"
            f"   提示：需先运行老代码生成 full_prediction_summary_20260201_174043.csv"
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


# In[ ]:


#2. 6 个核心特征构建（规则提取的核心数据来源）


# In[ ]:


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

