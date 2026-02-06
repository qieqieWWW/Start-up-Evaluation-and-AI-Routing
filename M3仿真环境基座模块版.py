"""
M3仿真系统 - 一键运行版本
整合了配置系统和主程序功能
只需运行此文件即可执行M3仿真
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

# 设置pandas选项以避免FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# 定义RISK_LABEL_MAP和EXTENDED_VALID_STATES
RISK_LABEL_MAP = {
    'successful': 0,
    'failed': 1,
    'canceled': 1,
    'suspended': 1,
    'live': 0,
    'undefined': 0
}

EXTENDED_VALID_STATES = ['successful', 'failed', 'canceled', 'suspended', 'live']


# ==========================================
# 配置系统
# ==========================================
class M3Config:
    def __init__(self):
        # M2处理数据输出目录
        self.output_dir = "Kickstarter_Clean"
        
        # 验证路径是否存在
        self._validate_paths()
        
        print(f"配置信息: 输出目录={self.output_dir}")
    
    def _validate_paths(self):
        """验证路径是否存在"""
        # 只验证和创建必要的目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"创建输出目录: {self.output_dir}")
    
    def update_data_path(self, new_path):
        """更新数据路径"""
        # 由于不再使用core_folder，此方法可以简化或移除
        print(f"✓ 数据路径更新功能已简化，当前仅使用输出目录: {self.output_dir}")
    
    def get_data_path(self):
        """获取当前数据路径"""
        return self.output_dir
    
    def get_cleaned_data_path(self):
        """获取清洗后数据的路径"""
        # 返回M2处理后的数据文件路径（基于最新生成的文件）
        return self.get_latest_processed_file()
    
    def get_latest_processed_file(self):
        """获取最新的M2处理文件路径"""
        output_dir = self.output_dir
        if os.path.exists(output_dir):
            # 查找目录中所有以"full_prediction_summary_"开头的CSV文件
            csv_files = [f for f in os.listdir(output_dir) if f.startswith("full_prediction_summary_") and f.endswith(".csv")]
            if csv_files:
                # 按修改时间排序，选择最新的文件
                csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
                return os.path.join(output_dir, csv_files[0])
        return None
    
    def __str__(self):
        return f"M3配置 - 清洗输出: {self.output_dir}"


# ==========================================
# M3 Core Class: 仿真环境基座
# ==========================================
class StartupEnv(gym.Env[Any, Any]):
    """
    符合 Gymnasium 标准的初创项目仿真环境。
    直接使用M2特征向量作为输入，完成M3仿真。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(StartupEnv, self).__init__()

        # 特征维度（基于M2特征工程 ）
        self.feature_dim = 6  # goal_ratio, time_penalty, category_risk, combined_risk, country_factor, urgency_score

        # ========== 修复：定义业务合理的状态/动作空间 ==========
        # 按每个特征的业务范围定义上下限（顺序与特征维度一致）
        self.feature_lows = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.feature_highs = np.array([50.0, 20.0, 1.0, 100.0, 1.0, 7.0], dtype=np.float32)
        
        # 定义状态空间 (Observation Space) - 贴合业务范围
        self.observation_space = spaces.Box(
            low=self.feature_lows, 
            high=self.feature_highs, 
            shape=(self.feature_dim,), 
            dtype=np.float32
        )

        # 定义动作空间 (Action Space) - 与状态空间范围一致
        self.action_space = spaces.Box(
            low=self.feature_lows, 
            high=self.feature_highs, 
            shape=(self.feature_dim,), 
            dtype=np.float32
        )

        self.current_state_vector = None
        self.steps_taken = 0
        self.max_steps = 12 # 模拟12个月的变化

    def reset(self, *, seed=None, options=None):
        """
        重置环境。
        关键：通过 options 传入M2特征向量，并强制约束在业务范围内。
        """
        super().reset(seed=seed)
        self.steps_taken = 0

        # 获取M2特征向量
        feature_vector = options.get("feature_vector") if options else None
        if feature_vector is None:
            print("⚠️ [Env] 未提供特征向量，使用默认 Mock 向量进行重置。")
            feature_vector = np.zeros(self.feature_dim, dtype=np.float32)

        # ========== 修复：清理异常值+强制约束业务范围 ==========
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
        feature_vector = np.clip(feature_vector, self.feature_lows, self.feature_highs)
        
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
        
        # ========== 修复：约束动作值在业务范围内 ==========
        action = np.nan_to_num(action, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
        action = np.clip(action, self.feature_lows, self.feature_highs)
        
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
# M3 主函数 - 使用M2处理好的数据直接运行仿真
# ==========================================
def save_simulation_results(simulation_results, save_path):
    """保存仿真结果到CSV文件（修复：拆解特征向量为单独列）"""
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将仿真结果转换为DataFrame
    results_data = []
    for project_result in simulation_results:
        # ========== 修复：拆解初始特征向量为单独列 ==========
        init_feature_vector = project_result["initial_feature_vector"]
        init_goal_ratio = init_feature_vector[0]
        init_time_penalty = init_feature_vector[1]
        init_category_risk = init_feature_vector[2]
        init_combined_risk = init_feature_vector[3]  # 核心风险
        init_country_factor = init_feature_vector[4]
        init_urgency_score = init_feature_vector[5]
        
        project_base = {
            "project_id": project_result["project_id"],
            "project_name": project_result["project_name"],
            "goal_amount": project_result["goal_amount"],
            # 初始特征（单独列）
            "init_goal_ratio": round(init_goal_ratio, 4),
            "init_time_penalty": round(init_time_penalty, 4),
            "init_category_risk": round(init_category_risk, 4),
            "init_combined_risk": round(init_combined_risk, 4),
            "init_country_factor": round(init_country_factor, 4),
            "init_urgency_score": round(init_urgency_score, 4),
            "total_steps": len(project_result["step_data"])
        }
        
        # 添加每步数据
        for step_data in project_result["step_data"]:
            row_data = project_base.copy()
            
            # ========== 修复：拆解每步特征向量为单独列 ==========
            new_state_vector = step_data["new_state_vector"]
            new_goal_ratio = new_state_vector[0]
            new_time_penalty = new_state_vector[1]
            new_category_risk = new_state_vector[2]
            new_combined_risk = new_state_vector[3]  # 核心风险
            new_country_factor = new_state_vector[4]
            new_urgency_score = new_state_vector[5]
            
            row_data.update({
                "step": step_data["step"],
                # 每步新特征（单独列）
                "new_goal_ratio": round(new_goal_ratio, 4),
                "new_time_penalty": round(new_time_penalty, 4),
                "new_category_risk": round(new_category_risk, 4),
                "new_combined_risk": round(new_combined_risk, 4),
                "new_country_factor": round(new_country_factor, 4),
                "new_urgency_score": round(new_urgency_score, 4),
                # 其他数据
                "reward": step_data["reward"],
                "terminated": step_data["terminated"],
                "truncated": step_data["truncated"]
            })
            results_data.append(row_data)
    
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 仿真结果已保存至: {save_path}")

def run_m3_simulation():
    """
    运行M3仿真系统 - 直接使用M2处理好的数据
    1. 读取M2处理好的数据文件（分批处理，避免内存问题）
    2. 基于M2特征向量完成M3仿真
    3. 保存仿真结果
    """
    print("="*60)
    print("M3仿真系统 - 使用M2处理好的数据直接运行（分批处理）")
    print("="*60)
    
    try:
        # 初始化配置
        config = M3Config()
        print(f"使用配置: {config}")
        
        # 1. 读取M2处理好的数据文件 - 查找最近生成的处理文件
        processed_data_path = config.get_latest_processed_file()
        
        if processed_data_path and os.path.exists(processed_data_path):
            print(f"\n找到最新的M2处理文件: {processed_data_path}")
            
            # 分批读取数据，避免内存问题
            batch_size = 1000  # 每批处理1000个项目
            # 重新计算总行数，减去表头行
            with open(processed_data_path, 'r', encoding="utf-8-sig") as f:
                # 跳过表头行
                next(f)
                total_rows = sum(1 for _ in f)
            print(f"数据总行数: {total_rows}")
            
            # 初始化仿真结果列表
            simulation_results = []
            
            # ========== 修复：全局初始化环境，批次复用 ==========
            # 环境初始化提到批次循环外（仅初始化一次）
            env = StartupEnv()
            
            # 使用chunksize分批读取，彻底避免skiprows问题
            chunk_iter = pd.read_csv(
                processed_data_path,
                encoding="utf-8-sig",
                chunksize=batch_size,
                dtype=str,
                na_values=['', 'NA', 'nan', 'N/A', 'null']
            )
            
            # 处理每个批次
            for batch_idx, chunks in enumerate(chunk_iter):
                batch_start = batch_idx * batch_size
                if batch_start >= total_rows:
                    break
                
                print(f"\n--- 处理数据批次: {batch_idx+1} ---")
                if chunks.empty:
                    continue
                
                # 检查特征列是否存在
                feature_columns = ["goal_ratio", "time_penalty", "category_risk", "combined_risk", "country_factor", "urgency_score"]
                missing_columns = [col for col in feature_columns if col not in chunks.columns]
                if missing_columns:
                    print(f"[警告] 当前批次缺少特征列: {missing_columns}")
                    continue
                
                print(f"当前批次数据行数: {len(chunks)}")
                print(f"当前批次字段名称: {list(chunks.columns)}")
                
                # 检查所需特征列是否存在
                missing_columns = [col for col in feature_columns if col not in chunks.columns]
                if missing_columns:
                    print(f"[警告] 当前批次数据中缺少特征列: {missing_columns}")
                    continue
                
                print(f"当前批次找到所有必需的特征列: {feature_columns}")
                
                # 处理当前批次的所有项目
                batch_simulation_results = []
                projects_to_process = min(len(chunks), total_rows - batch_start)
                
                for i in range(projects_to_process):
                    try:
                        row = chunks.iloc[i]
                        
                        # 提取特征向量并转换为浮点数
                        feature_vector = row[feature_columns].astype(float).values
                        
                        print(f"\n--- 仿真项目 {batch_start+i+1} ---")
                        project_name = row.get('name', '未知')[:50]  # 截断超长名称
                        goal_amount = row.get('goal', 0)
                        project_id = row.get('id', batch_start+i)
                        
                        # 重置环境
                        obs, info = env.reset(options={"feature_vector": feature_vector})
                        
                        # 收集单项目的仿真数据
                        project_simulation_data = {
                            "project_id": project_id,
                            "project_name": project_name,
                            "goal_amount": goal_amount,
                            "initial_feature_vector": obs.tolist(),
                            "step_data": []
                        }
                        
                        done = False
                        step_count = 0
                        
                        # 记录上一步的风险值，用于计算风险变化 - 移到循环开始处
                        prev_risk = None
                        while not done:
                            step_count += 1
                            
                            # 基于项目风险和融资可行性计算业务逻辑动作
                            # 锚定业务核心指标combined_risk（obs [3]）
                            current_risk = float(obs[3])  # combined_risk是第4个特征（索引3）
                            
# ========== 修复：有业务依据的动作生成 ==========
                            # 1. 提取核心风险指标（业务锚点）
                            combined_risk = obs[3]  # 核心风险：目标倍率×时间惩罚
                            current_risk_level = min(combined_risk / 20.0, 1.0)  # 风险等级（0~1）
    
                            # 2. 动态调整动作幅度：风险越高/步数越少，调整幅度越大
                            base_magnitude = 0.1 + (current_risk_level * 0.4)  # 风险高→幅度大（0.1~0.5）
                            step_decay = 1.0 - (step_count / env.max_steps)    # 步数多→幅度小（1.0~0.0）
                            action_magnitude = base_magnitude * step_decay

                            # 3. 定义每个特征的调整方向（贴合业务目标）
                            action_direction = np.array([
                                -0.8,  # goal_ratio：降低（权重0.8）
                                -0.7,  # time_penalty：降低（权重0.7）
                                -0.6,  # category_risk：降低（权重0.6）
                                -1.0,  # combined_risk：优先降低（核心指标，权重1.0）
                                -0.5,  # country_factor：降低（权重0.5）
                                +0.9   # urgency_score：提升（权重0.9）
                            ], dtype=np.float32)

                            # 4. 生成动作：基础状态 + 方向×幅度（确保在业务范围内）
                            action_new_state = obs + (action_direction * action_magnitude)
                            # 强制约束动作值在业务范围内（与环境空间一致）
                            action_new_state = np.clip(action_new_state, env.feature_lows, env.feature_highs)
                            
                            obs, reward, terminated, truncated, info = env.step(action_new_state)
                            
                            # 基于业务逻辑计算奖励
                            # 修复：确保使用正确的奖励变量
                            current_step_reward = 0.0
                            
                            # 1. 基础存活奖励：每运行1步 + 0.5分
                            current_step_reward += 0.5
                            
                            # 2. 风险优化奖励：当前步风险相比上一步降低越多，奖励越高
                            if step_count > 1 and prev_risk is not None:
                                risk_change = prev_risk - current_risk
                                # 风险降低越多，奖励越高，最高不超过2分
                                risk_improvement_reward = min(risk_change * 2.0, 2.0)
                                current_step_reward += risk_improvement_reward
                            
                            # 3. 终局奖励
                            if terminated:
                                if current_risk > 20.0:  # 高风险终止
                                    current_step_reward -= 5.0  # 项目失败扣5分
                                elif current_risk < 0.1:  # 低风险终止
                                    current_step_reward += 2.0  # 提前成功加2分
                            elif truncated:
                                current_step_reward += 2.0  # 跑满12步加2分
                            
                            # 使用当前步的奖励，避免与env.step返回的reward混淆
                            reward = current_step_reward
                            
                            # 添加终止条件（梯度化设计）
                            if current_risk > 20.0 and step_count >= 3:  # 高风险，至少运行3步
                                terminated = True
                            elif current_risk < 0.1 and step_count >= 3:  # 低风险，至少运行3步
                                terminated = True
                            elif step_count >= env.max_steps:  # 步数保护，最多env.max_steps步
                                truncated = True
                                terminated = False
                            
                            # 记录当前风险用于下一轮计算
                            prev_risk = current_risk
                            
                            # 打印调试信息（可选）
                            # print(f"Step {step_count}: Risk={current_risk:.2f}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")
                            
                            # 收集每步数据
                            step_data = {
                                "step": step_count,
                                "new_state_vector": obs.tolist(),
                                "reward": float(reward),
                                "terminated": bool(terminated),
                                "truncated": bool(truncated)
                            }
                            project_simulation_data["step_data"].append(step_data)
                            
                            # env.render()  # 注释掉渲染输出，避免测试代码
                            done = terminated or truncated
                        
                        # 将单项目数据加入批次列表
                        batch_simulation_results.append(project_simulation_data)  # type: ignore
                        print(f"项目 {batch_start+i+1} 仿真完成")
                        
                    except Exception as e:
                        print(f"  处理项目 {batch_start+i+1} 时发生错误: {str(e)}")
                        import traceback
                        traceback.print_exc()  # 打印详细错误栈
                        continue
                
                # 将当前批次的结果加入总列表
                simulation_results.extend(batch_simulation_results)
                batch_end = min(batch_start + batch_size, total_rows)
                print(f"批次 {batch_start+1}-{batch_end} 处理完成，共处理 {len(batch_simulation_results)} 个项目")
                
                # 关闭环境，为下一个批次重新初始化
                # env.close()  # 注释掉，全局环境在所有批次结束后关闭
                
        else:
            raise FileNotFoundError(f"错误：未找到M2处理好的数据文件。请运行M2.py生成处理文件。")
        
        if not simulation_results:
            raise ValueError("没有成功处理任何项目的仿真数据")
        
        # 所有批次处理完成后，关闭环境
        env.close()
        
        print("\n--- 所有批次M3仿真完成 ---")
        
        # 保存所有仿真结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(config.output_dir, f"m3_simulation_results_{timestamp}.csv")
        save_simulation_results(simulation_results, save_path)
        
        print("\n" + "="*60)
        print("M3仿真系统 - M2+M3完整流程执行完成（分批处理）")
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