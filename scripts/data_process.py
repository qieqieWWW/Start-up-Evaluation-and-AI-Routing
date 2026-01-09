import pandas as pd
import os
import shutil
from io import StringIO
import sys

# 项目结构整理
dataset_folder_name = "Kickstarter_2025-12-18T03_20_24_296Z"
datasets_dir = "datasets"
output_dir = "Kickstarter_Clean"

# 创建datasets目录
os.makedirs(datasets_dir, exist_ok=True)

# 检查数据集文件夹是否在根目录，如果是则移动到datasets目录
dataset_path = os.path.join(datasets_dir, dataset_folder_name)
dataset_in_datasets = os.path.exists(dataset_path)
if os.path.exists(dataset_folder_name) and not dataset_in_datasets:
    shutil.move(dataset_folder_name, dataset_path)

# 数据集路径（相对路径）
data_path = os.path.join(datasets_dir, dataset_folder_name)
output_path = os.path.join(output_dir, "kickstarter_cleaned.csv")
log_path = os.path.join(output_dir, "cleaning_log.txt")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)


# 日志记录类
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
log_capture = LogCapture(log_path)
sys.stdout = log_capture

# 检查路径是否存在
if not os.path.exists(data_path):
    print(f"错误：数据集路径 {data_path} 不存在。请确保数据集已放置在正确位置。")
    exit(1)

# 读取数据集
if os.path.isfile(data_path) and data_path.endswith(".csv"):
    # 如果是单个 CSV 文件
    df = pd.read_csv(data_path)
    print(f"读取单个 CSV 文件: {data_path}")
elif os.path.isdir(data_path):
    # 如果是文件夹，读取所有 CSV 文件并合并
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"错误：文件夹 {data_path} 中没有找到 CSV 文件。")
        exit(1)
    df_list = []
    for file in csv_files:
        file_path = os.path.join(data_path, file)
        df_temp = pd.read_csv(file_path)
        df_list.append(df_temp)
        print(f"读取 CSV 文件: {file_path}")
    df = pd.concat(df_list, ignore_index=True)
    print(f"合并了 {len(csv_files)} 个 CSV 文件，总行数: {len(df)}")
else:
    print(f"错误：{data_path} 不是有效的 CSV 文件或文件夹。")
    exit(1)

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

# 清洗步骤（这里假设简单清洗：去除完全缺失的行，如果有的话。但根据用户描述，主要查看信息）
# 用户未指定具体清洗步骤，所以仅保存原数据作为“清洗完成”的数据集

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


# 处理非核心字符串字段缺失：填充“未知”
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
    print("衍生字段 'duration_days'（项目持续天数，" "基于Unix时间戳计算）。")

print("格式标准化结果示例:")
print(df.head())

# 保存清洗后的数据
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\n清洗完成！数据集已保存到 {output_path}")

# 保存日志
log_capture.save_log()
print(f"清洗日志已保存到 {log_path}")

# 恢复 stdout
sys.stdout = log_capture.terminal
