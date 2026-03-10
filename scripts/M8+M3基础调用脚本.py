#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import importlib.util
from pathlib import Path
import m8_rule_adapter
from m8_rule_adapter import judge_project_risk_m8
print("M8文件路径：", m8_rule_adapter.__file__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
M3_MODULE_PATH = SCRIPT_DIR / "M3仿真环境基座模块版.py"
M3_MODULE_NAME = "m3_simulation_base"

spec = importlib.util.spec_from_file_location(M3_MODULE_NAME, M3_MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载M3模块：{M3_MODULE_PATH}")

M3仿真环境基座模块版 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M3仿真环境基座模块版)
StartupEnv = M3仿真环境基座模块版.StartupEnv  # M3仿真环境
print("M3文件路径：", M3_MODULE_PATH)

def test_m8_m3_integration():
    """端到端测试：M8输出 → M3仿真"""
    # 1. 定义测试项目（复用M8的测试用例）
    test_project = {
        'goal_usd': 15000,
        'duration_days': 60,
        'main_category': 'Technology',
        'country': 'US',
        'country_factor': 0.3,  # 已改为country_factor
        'actual_funding_usd': 5000,
        'planned_duration_days': 45
    }

    # 2. 调用M8，获取特征向量
    risk_level, reasons, intermediate = judge_project_risk_m8(test_project, verbose=False)
    print(f"📌 M8输出风险等级：{risk_level}")
    assert 'country_factor' in intermediate, "M8的intermediate中无country_factor字段"
    assert isinstance(intermediate['country_factor'], (int, float)), "country_factor不是数值类型"
    print(f"✅ 校验通过：country_factor = {intermediate['country_factor']}")
    

    # 3. 提取M3需要的6维特征向量（按M3顺序）
    m3_feature_order = ['goal_ratio', 'time_penalty', 'category_risk', 'combined_risk', 'country_factor', 'urgency_score']
    m3_feature_vector = np.array([intermediate[feat] for feat in m3_feature_order], dtype=np.float32)
    print(f"📌 传入M3的特征向量：{m3_feature_vector}")

    # 4. 初始化M3仿真环境
    env = StartupEnv()
    # 5. 重置M3环境，传入M8的特征向量
    obs, info = env.reset(options={"feature_vector": m3_feature_vector})
    print(f"✅ M3环境重置成功，初始状态：{obs}")

    # 6. 运行M3仿真（模拟12步）
    total_steps = 12
    for step in range(total_steps):
        # 模拟Agent动作：直接复用当前状态（也可修改测试）
        action = obs
        # 执行M3的step方法
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"🔄 M3仿真第{step+1}步：状态={obs.round(4)}, 步数={info['steps']}")

    # 7. 验证仿真完成
    assert truncated, "M3仿真未达到最大步数（truncated应为True）"
    print("\n🎉 M8 → M3 端到端适配测试通过！")

if __name__ == "__main__":
    test_m8_m3_integration()

