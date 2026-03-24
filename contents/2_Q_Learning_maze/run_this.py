"""
强化学习迷宫示例。

红色矩形：          探索者 (explorer)
黑色矩形：          地狱区域 (hell)    [奖励 = -1]
黄色圆形：          天堂区域 (paradise) [奖励 = +1]
其他所有状态：      地面 (ground)    [奖励 = 0]

本脚本是主控部分，控制示例的更新方法。
RL (Q-learning) 算法实现位于 RL_brain.py。

查看更多教程：https://morvanzhou.github.io/tutorials/

========================================
Q-Learning 算法流程：
1. 初始化 Q(s,a) = 0，∀s∈S, a∈A
2. 对于每个 episode：
   a. 初始化状态 s
   b. 对于 episode 的每一步：
      - 使用 ε-greedy 策略选择动作 a
      - 执行动作 a，获得奖励 r 和下一状态 s'
      - 更新 Q(s,a)：Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
      - s ← s'
   c. 如果 s 是终止状态，则结束
========================================
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    """
    Q-Learning 训练主循环
    
    运行 100 个 episode 的训练，每个 episode 是一个完整的迷宫探索过程
    """
    for episode in range(100):
        # ========== 步骤 1：初始化 ==========
        # 重置环境，获取初始状态 s
        observation = env.reset()

        # ========== 步骤 2-5：episode 循环 ==========
        while True:
            # 刷新环境，显示当前状态
            env.render()

            # ========== 步骤 2：选择动作 ==========
            # RL 根据当前状态 s 选择动作 a（ε-greedy 策略）
            # s = str(observation) 将状态转换为字符串
            action = RL.choose_action(str(observation))

            # ========== 步骤 3：执行动作 ==========
            # 执行动作 a，获取：
            # - s_: 下一状态 (next state)
            # - reward: 奖励 (reward)
            # - done: 是否结束 (terminal state flag)
            observation_, reward, done = env.step(action)

            # ========== 步骤 4：更新 Q 值 ==========
            # RL 学习这个转换 (s, a, r, s')
            # Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
            # 即：Q(s,a) ← (1-α)*Q(s,a) + α * [r + γ * max_a' Q(s',a')]
            RL.learn(str(observation), action, reward, str(observation_))

            # ========== 步骤 5：状态转移 ==========
            # s ← s'（更新当前状态）
            observation = observation_

            # ========== 检查终止条件 ==========
            # 如果到达终止状态（done=True），结束当前 episode
            if done:
                break

    # ========== 结束 ==========
    # 所有 episode 训练完成
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # ========== 初始化 ==========
    # 创建迷宫环境对象
    env = Maze()
    
    # 创建 Q-Learning 智能体
    # actions = 环境的所有可能动作（0, 1, 2, 3 对应上、下、左、右）
    # n_actions 由 maze_env.py 中的 Maze 类定义
    RL = QLearningTable(actions=list(range(env.n_actions)))

    # ========== 启动 ==========
    # 延迟 100ms 后调用 update 函数开始训练
    # 使用 tkinter 的 mainloop 启动 GUI
    env.after(100, update)
    env.mainloop()