import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
from typing import Tuple, Dict, Any

class DriftGenerator:
    """分布偏移生成器 - 模拟突发风险事件"""
    
    @staticmethod
    def generate_market_crash(market_heat: float, intensity: float = 0.3) -> float:
        """模拟市场崩盘 - 市场热度突然下降"""
        if random.random() < 0.05:  # 5%概率发生市场崩盘
            crash_factor = random.uniform(0.3, 0.7) * intensity
            return max(0, market_heat - crash_factor)
        return market_heat
    
    @staticmethod
    def generate_funding_chain_break(supporters: int, intensity: float = 0.4) -> int:
        """模拟资金链断裂 - 支持者突然减少"""
        if random.random() < 0.03:  # 3%概率发生资金链断裂
            loss_factor = random.uniform(0.2, 0.6) * intensity
            return max(0, int(supporters * (1 - loss_factor)))
        return supporters
    
    @staticmethod
    def generate_negative_news(market_heat: float, intensity: float = 0.25) -> float:
        """模拟负面新闻 - 市场热度下降"""
        if random.random() < 0.08:  # 8%概率发生负面新闻
            news_impact = random.uniform(0.1, 0.4) * intensity
            return max(0, market_heat - news_impact)
        return market_heat
    
    @staticmethod
    def generate_success_boom(market_heat: float, intensity: float = 0.2) -> float:
        """模拟成功热潮 - 市场热度上升"""
        if random.random() < 0.1:  # 10%概率发生成功热潮
            boom_factor = random.uniform(0.1, 0.3) * intensity
            return min(1.0, market_heat + boom_factor)
        return market_heat
    
    @staticmethod
    def generate_supporter_growth(supporters: int, intensity: float = 0.15) -> int:
        """模拟支持者增长爆发"""
        if random.random() < 0.12:  # 12%概率发生支持者增长
            growth_factor = random.uniform(0.2, 0.5) * intensity
            return int(supporters * (1 + growth_factor))
        return supporters

class CrowdfundingEnv(gym.Env):
    """
    众筹环境类，模拟众筹项目的关键指标变化
    状态空间：[当前金额, 支持者数, 时间进度, 市场热度]
    动作空间：[营销投入, 产品改进投入]
    """
    def __init__(self):
        # 定义动作空间 - 营销投入和产品改进投入 (0-1范围)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 定义状态空间 - 众筹关键指标
        # [当前金额, 支持者数, 时间进度(0-1), 市场热度(0-1)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([100000, 1000, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        # 众筹参数
        self.target_amount = 50000  # 目标金额
        self.max_steps = 100       # 最大步数(天)
        self.current_step = 0
        self.current_amount = 0    # 当前筹集金额
        self.supporters = 0        # 支持者数
        self.market_heat = 0.5     # 市场热度
        self.success_threshold = 0.8  # 成功阈值(80%)
        
        # 风险参数
        self.risk_level = 0.5      # 风险等级(0-1)，影响风险事件发生概率
        self.event_history = []    # 事件历史记录
        
        # 记录历史数据用于可视化
        self.history = {
            'amount': [],
            'supporters': [],
            'market_heat': [],
            'steps': [],
            'events': []
        }
        
    def reset(self) -> np.ndarray:
        """重置环境状态"""
        self.current_step = 0
        self.current_amount = 0
        self.supporters = 0
        self.market_heat = 0.5
        self.event_history = []
        
        # 清空历史记录
        self.history = {
            'amount': [],
            'supporters': [],
            'market_heat': [],
            'steps': [],
            'events': []
        }
        
        initial_state = np.array([
            self.current_amount,
            self.supporters,
            0.0,  # 时间进度
            self.market_heat
        ])
        
        return initial_state
    
    def _apply_drift_events(self) -> None:
        """应用分布偏移事件"""
        # 市场崩盘
        old_market_heat = self.market_heat
        self.market_heat = DriftGenerator.generate_market_crash(self.market_heat, self.risk_level)
        if old_market_heat != self.market_heat:
            self.event_history.append(f"市场崩盘: 热度从{old_market_heat:.2f}降至{self.market_heat:.2f}")
        
        # 资金链断裂
        old_supporters = self.supporters
        self.supporters = DriftGenerator.generate_funding_chain_break(self.supporters, self.risk_level)
        if old_supporters != self.supporters:
            self.event_history.append(f"资金链断裂: 支持者从{old_supporters}降至{self.supporters}")
        
        # 负面新闻
        old_market_heat = self.market_heat
        self.market_heat = DriftGenerator.generate_negative_news(self.market_heat, self.risk_level)
        if old_market_heat != self.market_heat:
            self.event_history.append(f"负面新闻: 热度从{old_market_heat:.2f}降至{self.market_heat:.2f}")
        
        # 成功热潮
        old_market_heat = self.market_heat
        self.market_heat = DriftGenerator.generate_success_boom(self.market_heat, self.risk_level)
        if old_market_heat != self.market_heat:
            self.event_history.append(f"成功热潮: 热度从{old_market_heat:.2f}升至{self.market_heat:.2f}")
        
        # 支持者增长爆发
        old_supporters = self.supporters
        self.supporters = DriftGenerator.generate_supporter_growth(self.supporters, self.risk_level)
        if old_supporters != self.supporters:
            self.event_history.append(f"支持者增长: 从{old_supporters}增至{self.supporters}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步动作
        Args:
            action: [营销投入, 产品改进投入]
        """
        # 输入校验
        if not isinstance(action, np.ndarray) or action.shape != (2,):
            raise ValueError("Action must be a numpy array of shape (2,)")
        if not np.all((action >= 0) & (action <= 1)):
            raise ValueError("Action values must be between 0 and 1")
        
        # 更新时间
        self.current_step += 1
        time_progress = min(self.current_step / self.max_steps, 1.0)
        
        # 基于动作更新状态
        marketing_effort = action[0]
        product_improvement = action[1]
        
        # 营销投入增加支持者数和市场热度
        new_supporters = int(marketing_effort * 50 * (1 + self.market_heat))
        self.supporters += new_supporters
        
        # 产品改进影响市场热度和转化率
        self.market_heat = min(1.0, self.market_heat + product_improvement * 0.1)
        
        # 应用分布偏移事件
        self._apply_drift_events()
        
        # 计算新筹集金额 (基于支持者数和转化率)
        conversion_rate = 0.3 + self.market_heat * 0.4  # 转化率 30%-70%
        daily_amount = self.supporters * 100 * conversion_rate  # 每个支持者平均100元
        self.current_amount += daily_amount
        
        # 确保不超过目标金额
        self.current_amount = min(self.current_amount, self.target_amount)
        
        # 更新状态
        state = np.array([
            self.current_amount,
            self.supporters,
            time_progress,
            self.market_heat
        ])
        
        # 记录历史数据
        self.history['amount'].append(self.current_amount)
        self.history['supporters'].append(self.supporters)
        self.history['market_heat'].append(self.market_heat)
        self.history['steps'].append(self.current_step)
        self.history['events'].append(len(self.event_history))
        
        # 计算奖励 - 基于成功度和效率，考虑风险事件的影响
        success_rate = self.current_amount / self.target_amount
        reward = (success_rate * 100) + (marketing_effort * 10) + (product_improvement * 5)
        
        # 风险事件惩罚
        if len(self.event_history) > 0:
            reward -= len(self.event_history) * 5  # 每个事件扣5分
            
        # 检查是否成功或结束
        done = (success_rate >= self.success_threshold) or (self.current_step >= self.max_steps)
        
        return state, reward, done, {}
    
    def render(self, mode='human'):
        """渲染环境状态"""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"  目标金额: {self.target_amount}, 当前金额: {self.current_amount:.2f}")
        print(f"  支持者数: {self.supporters}")
        print(f"  市场热度: {self.market_heat:.2f}")
        print(f"  成功率: {self.current_amount / self.target_amount:.2%}")
        
        # 显示最近的事件
        if len(self.event_history) > 0:
            print("\n  最近事件:")
            for event in self.event_history[-3:]:  # 显示最近3个事件
                print(f"    - {event}")
        
        print("-" * 40)
    
    def visualize(self):
        """可视化众筹过程"""
        plt.figure(figsize=(18, 12))
        
        # 金额变化图
        plt.subplot(2, 3, 1)
        plt.plot(self.history['steps'], self.history['amount'], 'b-', label='筹集金额')
        plt.axhline(y=self.target_amount, color='r', linestyle='--', label='目标金额')
        plt.xlabel('时间(天)')
        plt.ylabel('金额')
        plt.title('众筹金额变化')
        plt.legend()
        plt.grid(True)
        
        # 支持者数变化图
        plt.subplot(2, 3, 2)
        plt.plot(self.history['steps'], self.history['supporters'], 'g-', label='支持者数')
        plt.xlabel('时间(天)')
        plt.ylabel('支持者数')
        plt.title('支持者数变化')
        plt.legend()
        plt.grid(True)
        
        # 市场热度变化图
        plt.subplot(2, 3, 3)
        plt.plot(self.history['steps'], self.history['market_heat'], 'r-', label='市场热度')
        plt.xlabel('时间(天)')
        plt.ylabel('市场热度')
        plt.title('市场热度变化')
        plt.legend()
        plt.grid(True)
        
        # 成功率变化图
        plt.subplot(2, 3, 4)
        success_rate = [a / self.target_amount for a in self.history['amount']]
        plt.plot(self.history['steps'], success_rate, 'm-', label='成功率')
        plt.axhline(y=self.success_threshold, color='orange', linestyle='--', label='成功阈值')
        plt.xlabel('时间(天)')
        plt.ylabel('成功率')
        plt.title('成功率变化')
        plt.legend()
        plt.grid(True)
        
        # 事件发生频率图
        plt.subplot(2, 3, 5)
        plt.plot(self.history['steps'], self.history['events'], 'k-', label='累计事件数')
        plt.xlabel('时间(天)')
        plt.ylabel('累计事件数')
        plt.title('风险事件发生频率')
        plt.legend()
        plt.grid(True)
        
        # 风险事件时间线
        plt.subplot(2, 3, 6)
        event_steps = []
        for i, step in enumerate(self.history['steps']):
            if i > 0 and self.history['events'][i] > self.history['events'][i-1]:
                event_steps.append(step)
        
        if event_steps:
            plt.eventplot(event_steps, lineoffsets=1, linelengths=0.5, colors='red')
            plt.title('风险事件时间线')
            plt.xlabel('时间(天)')
            plt.yticks([])
        else:
            plt.text(0.5, 0.5, '无风险事件发生', horizontalalignment='center', verticalalignment='center')
            plt.title('风险事件时间线')
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """关闭环境"""
        pass

class EnvAPI:
    """
    Env API MVP实现 - 众筹场景专用
    """
    def __init__(self):
        self.env = CrowdfundingEnv()
        self.observation = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        
    def get_observation(self) -> np.ndarray:
        """获取当前观察"""
        return self.observation
    
    def take_action(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        执行动作并返回新观察和奖励
        Args:
            action: [营销投入, 产品改进投入]
        """
        if not isinstance(action, np.ndarray) or action.shape != (2,):
            raise ValueError("Action must be a numpy array of shape (2,)")
        if not np.all((action >= 0) & (action <= 1)):
            raise ValueError("Action values must be between 0 and 1")
            
        self.observation, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        return self.observation, reward, done
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.observation = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        return self.observation
    
    def render(self):
        """渲染当前状态"""
        self.env.render()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.step_count),
            'step_count': self.step_count,
            'current_amount': self.observation[0],
            'supporters': self.observation[1],
            'success_rate': self.observation[0] / self.env.target_amount
        }
    
    def visualize(self):
        """可视化结果"""
        self.env.visualize()
    
    def close(self):
        """关闭环境"""
        self.env.close()

def crowdfunding_demo():
    """
    众筹Demo - 模拟不同策略的效果
    """
    print("=== 众筹Demo开始 ===")
    
    # 创建环境API实例
    api = EnvAPI()
    
    # 策略1: 均衡策略 (营销和产品各50%)
    print("\n策略1: 均衡策略")
    api.reset()
    for step in range(50):
        action = np.array([0.5, 0.5])  # 均衡投入
        observation, reward, done = api.take_action(action)
        api.render()
        if done:
            break
    
    stats1 = api.get_stats()
    print(f"策略1结果: 总奖励={stats1['total_reward']:.2f}, 平均奖励={stats1['average_reward']:.2f}")
    print(f"最终金额: {stats1['current_amount']:.2f}, 支持者数: {stats1['supporters']}")
    print(f"成功率: {stats1['success_rate']:.2%}")
    
    # 策略2: 营销优先策略 (营销80%, 产品20%)
    print("\n策略2: 营销优先策略")
    api.reset()
    for step in range(50):
        action = np.array([0.8, 0.2])  # 营销优先
        observation, reward, done = api.take_action(action)
        api.render()
        if done:
            break
    
    stats2 = api.get_stats()
    print(f"策略2结果: 总奖励={stats2['total_reward']:.2f}, 平均奖励={stats2['average_reward']:.2f}")
    print(f"最终金额: {stats2['current_amount']:.2f}, 支持者数: {stats2['supporters']}")
    print(f"成功率: {stats2['success_rate']:.2%}")
    
    # 策略3: 产品优先策略 (营销20%, 产品80%)
    print("\n策略3: 产品优先策略")
    api.reset()
    for step in range(50):
        action = np.array([0.2, 0.8])  # 产品优先
        observation, reward, done = api.take_action(action)
        api.render()
        if done:
            break
    
    stats3 = api.get_stats()
    print(f"策略3结果: 总奖励={stats3['total_reward']:.2f}, 平均奖励={stats3['average_reward']:.2f}")
    print(f"最终金额: {stats3['current_amount']:.2f}, 支持者数: {stats3['supporters']}")
    print(f"成功率: {stats3['success_rate']:.2%}")
    
    # 可视化策略1的结果
    print("\n可视化策略1的结果:")
    api.visualize()
    
    print("=== 众筹Demo结束 ===")
    api.close()

if __name__ == "__main__":
    # 运行众筹Demo
    crowdfunding_demo()