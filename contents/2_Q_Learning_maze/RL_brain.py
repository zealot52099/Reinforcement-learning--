"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    """
    Q-Learning 算法实现
    使用 Q-Table 存储状态-动作对应的 Q 值
    
    Q-learning 算法核心公式：
    Q(s, a) ← Q(s, a) + α[r + γ * max(Q(s', a')) - Q(s, a)]
    
    其中：
    - α (alpha): 学习率 (learning_rate)
    - γ (gamma): 折扣因子 (reward_decay)  
    - r: 奖励 (reward)
    - s: 当前状态 (state)
    - a: 动作 (action)
    - s': 下一状态 (next state)
    """
    
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        初始化 Q-Learning 表格
        
        Args:
            actions: 可用的动作列表 (action space)
            learning_rate (α): 学习率，控制 Q 值更新步长 [0, 1]
            reward_decay (γ): 折扣因子，考虑未来奖励的重要性 [0, 1]
            e_greedy (ε): epsilon-greedy 策略中的探索率
        """
        self.actions = actions  # 可用动作列表
        self.lr = learning_rate       # α (alpha) - 学习率
        self.gamma = reward_decay      # γ (gamma) - 折扣因子
        self.epsilon = e_greedy        # ε (epsilon) - 探索率
        # 创建空的 Q 表，列为动作，dtype=np.float64 确保数值为浮点数
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        """
        根据当前观察选择动作（ε-greedy 策略）
        
        ε-greedy 策略：
        - 以概率 ε 随机选择动作（探索）
        - 以概率 1-ε 选择当前最优动作（利用）
        
        Args:
            observation: 当前状态 s
            
        Returns:
            选择的动作 a
        """
        # 确保当前状态存在于 Q 表中（如果不存在则添加）
        self.check_state_exist(observation)
        
        # 动作选择：ε-greedy 策略
        if np.random.uniform() < self.epsilon:
            # 以概率 1-ε 选择最优动作（利用，exploitation）
            # a* = argmax_a Q(s, a)
            state_action = self.q_table.loc[observation, :]
            # 可能有多个动作具有相同的最大 Q 值，随机选择一个
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 以 ε 的概率随机选择动作（探索，exploration）
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """
        更新 Q 值（Q-learning 算法核心更新公式）
        
        TD (Temporal Difference) 更新公式：
        Q(s, a) ← Q(s, a) + α * TD_error
        
        其中 TD_error = r + γ * max(Q(s', a')) - Q(s, a)
        
        Args:
            s: 当前状态 (current state)
            a: 执行的动作 (action taken)
            r: 获得的奖励 (reward received)
            s_: 下一状态 (next state)
        """
        # 确保下一状态存在于 Q 表中
        self.check_state_exist(s_)
        
        # 获取当前状态-动作对的 Q 值预测 Q(s, a)
        q_predict = self.q_table.loc[s, a]
        
        if s_ != 'terminal':
            # 下一状态不是终止状态：计算 TD 目标值
            # Q(s', a*) = max_a Q(s', a)
            # Q_target = r + γ * Q(s', a*)
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            # 下一状态是终止状态：TD 目标就是即时奖励（没有未来奖励）
            # Q_target = r
            q_target = r
        
        # Q 值更新：Q(s,a) ← Q(s,a) + α * (Q_target - Q_predict)
        # 即：Q(s,a) ← Q(s,a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        """
        检查状态是否存在于 Q 表中，如果不存在则添加
        
        这个方法确保 Q 表能够动态扩展，以适应未知状态
        
        Args:
            state: 要检查/添加的状态 s
            
        Note:
            注意：pandas 2.0+ 版本移除了 DataFrame.append() 方法
            因此改用 pd.concat() 实现相同功能
        """
        if state not in self.q_table.index:
            # 状态不在 Q 表中，添加新行（初始化 Q 值为 0）
            # Q(s, a) 的初始值通常设为 0（也可以设为其他值如 0.5）
            # 创建新行数据，值为 0 表示初始 Q 值
            new_row = pd.DataFrame(
                [[0]*len(self.actions)],    # 初始化为 0 的 Q 值
                columns=self.q_table.columns,  # 与现有 Q 表列对齐
                index=[state],               # 新状态作为索引
            )
            # 使用 pd.concat 合并（pandas 2.0+ 不再支持 append 方法）
            self.q_table = pd.concat([self.q_table, new_row])