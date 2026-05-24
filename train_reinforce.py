"""
REINFORCE 学生填空版。

目标:
    通过四个 TODO 实现 vanilla REINFORCE(带 baseline),让 CartPole-v1
    的平均回报稳定逼近 max_t。完成后用 evaluate.py / vis.py 指向本文件的
    Agent 类即可评估。

    python evaluate.py \\
        --agent-class train_reinforce:Agent \\
        --agent-init-kwargs '{"n_state":4,"hidden_c":16,"n_action":2}' \\
        --checkpoint checkpoints/reinforce_model.pt \\
        --seed-base 42 --seed-count 100

固定基座(不需要修改):
    Agent.decide           —— 训练时采样动作 + log_prob
    Agent.save_model/load_model
    Agent.predict / policy_info  —— evaluate / vis 接口
    Environment.__init__   —— env 和 agent 构造
    best-so-far 保存逻辑   —— 避免末尾一次坏更新覆盖好模型
"""

import os
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# =============================================================================
# 超参数(学生可以调,但建议先用默认值跑通 TODO)
# =============================================================================
LR = 1e-2
GAMMA = 1.0
HIDDEN_C = 16
MAX_T = 2000
EPOCHS = 3000
SOLVED_AVG = 1450.0      # 连续 100 轮平均分达到此阈值即认为收敛
SAVE_DIR = 'checkpoints'


class Agent(nn.Module):
    def __init__(self, n_state, hidden_c, n_action):
        super().__init__()
        self.fc1 = nn.Linear(n_state, hidden_c)
        self.fc2 = nn.Linear(hidden_c, n_action)

    # =========================================================================
    # TODO 1: 策略网络的前向传播 ✅ 已实现
    # =========================================================================
    def forward(self, x):
        # ------ 你的代码开始 ------
        # fc1 -> ReLU -> fc2 -> softmax(dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
        # ------ 你的代码结束 ------

    # ----- 固定基座:采样动作 + log_prob ------------------------------------
    def decide(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        act = m.sample()
        return act.item(), m.log_prob(act)

    # ----- 固定基座:evaluate / vis 接口 -----------------------------------
    def predict(self, state):
        state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(state)
        return int(probs.argmax(dim=1).item())

    def policy_info(self, state):
        state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
        with torch.no_grad():
            probs = self.forward(state).squeeze(0).tolist()
        return {"P(LEFT)": f"{probs[0]:.3f}", "P(RIGHT)": f"{probs[1]:.3f}"}

    # ----- 固定基座:checkpoint -------------------------------------------
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(self.state_dict(), filepath)
        print(f"REINFORCE 模型已保存到: {filepath}")

    def load_model(self, filepath, map_location=None):
        self.load_state_dict(torch.load(filepath, map_location=map_location))
        print(f"REINFORCE 模型已加载: {filepath}")


class Environment:
    def __init__(self, max_t=MAX_T, gamma=GAMMA):
        # max_episode_steps 对齐 max_t,保证训练分数上限等于 max_t。
        self.env = gym.make('CartPole-v1', max_episode_steps=max_t)
        self.max_t = max_t
        self.gamma = gamma
        n_state = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        self.agent = Agent(n_state, HIDDEN_C, n_action)

    def train(self, epoch=EPOCHS, save_path=None):
        optimizer = optim.Adam(self.agent.parameters(), lr=LR)
        scores_deque = deque(maxlen=100)
        scores = []
        best_avg = float('-inf')

        for i_episode in range(1, epoch + 1):
            saved_log_probs = []
            rewards = []
            state = self.env.reset()[0]
            for _ in range(self.max_t):
                action, log_prob = self.agent.decide(state)
                saved_log_probs.append(log_prob)
                state, reward, t1, t2, _ = self.env.step(action)
                rewards.append(reward)
                if t1 or t2:
                    break
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            # =================================================================
            # TODO 2: 计算逐步折扣回报 G_t ✅ 已实现
            # =================================================================
            # ------ 你的代码开始 ------
            # 从后向前累积折扣回报 G_t = r_t + gamma * r_{t+1} + ...
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            # ------ 你的代码结束 ------
            returns = torch.tensor(returns, dtype=torch.float32)

            # =================================================================
            # TODO 3: returns 标准化(减均值 / 除标准差) ✅ 已实现
            # =================================================================
            if returns.numel() > 1:
                # ------ 你的代码开始 ------
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                # ------ 你的代码结束 ------

            # =================================================================
            # TODO 4: 构造 policy loss 并反传 ✅ 已实现
            # =================================================================
            # ------ 你的代码开始 ------
            log_probs = torch.cat(saved_log_probs)      # (T,)
            policy_loss = -(log_probs * returns).sum()  # 最小化负目标 = 最大化 J(θ)
            # ------ 你的代码结束 ------

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if i_episode % 100 == 0:
                print(f'Episode {i_episode}\t Average Score: {np.mean(scores_deque):.2f}')

            # ----- 固定基座:best-so-far 保存 ----------------------------------
            if save_path is not None and len(scores_deque) == scores_deque.maxlen:
                current_avg = float(np.mean(scores_deque))
                if current_avg > best_avg:
                    best_avg = current_avg
                    self.agent.save_model(save_path)

            if np.mean(scores_deque) >= SOLVED_AVG:
                print(f'Environment solved in {i_episode - 100} episode!\t'
                      f'Average Score: {np.mean(scores_deque):.2f}')
                break

        if save_path is not None and best_avg == float('-inf'):
            self.agent.save_model(save_path)

        return scores


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    test_env = Environment()
    record = test_env.train(epoch=EPOCHS, save_path=os.path.join(SAVE_DIR, 'reinforce_model.pt'))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(record)), record)
    plt.title('REINFORCE Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plot_path = os.path.join(SAVE_DIR, 'reinforce_training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")
    print(f"最后 20 轮平均分: {np.mean(record[-20:]):.2f}")


if __name__ == '__main__':
    main()
