"""
Q-Learning Bonus 2 优化版。

主要修改 vs 原版（详见文末说明）：
1. MAX_DIGITIZE_NUM: 6 → 10   离散化精度大幅提升（状态数 1296 → 10000）
2. ε 指数衰减: 0.5 → 0.005   充分探索后过渡到贪心利用
3. 状态 clip 范围优化：pole_angle 精确覆盖终止边界 ±0.42 rad；pole_v 放宽到 ±3
4. LR: 0.04 → 0.08；EPOCHS: 2000 → 6000；TARGET: 1000 → 2000
5. 周期性贪心评估 + best-so-far checkpoint 保存（关键！）

使用方式：
    python train_qlearning_bonus2.py
    python evaluate.py \\
        --agent-class train_qlearning_bonus2:Agent \\
        --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.08,"gamma":0.99,"epsilon":0.0}' \\
        --checkpoint checkpoints/q_learning_bonus2_best.pkl \\
        --seed-base 42 --seed-count 100 --max-episode-steps 2000

固定基座（不修改原始签名）：
    predict / policy_info / save_model / load_model   —— evaluate / vis 接口
    make_bins_state / bins                            —— 状态离散化
"""

import os
import pickle
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 超参数（Bonus 2 调优）
# =============================================================================
LR            = 0.08     # 学习率（原 0.04；更多状态需要稍快的学习速率）
GAMMA         = 0.99     # 折扣因子
EPSILON       = 0.5      # 初始 ε（高探索起步）
EPSILON_MIN   = 0.005    # 最小 ε（保留微量探索避免完全固化）
EPSILON_DECAY = 0.9993   # 每轮乘以此系数（6000 轮后 0.5 * 0.9993^6000 ≈ 0.012）
EPOCHS        = 6000     # 训练轮数（原 2000；需要足够轮次让 Q 表收敛）
TARGET        = 2000     # 单局最大步数（与 evaluate.py 对齐）
MAX_DIGITIZE_NUM = 10    # 每维 bin 数（原 6；10^4 = 10000 个离散状态）
SAVE_DIR      = 'checkpoints'
EVAL_INTERVAL = 300      # 每 N 轮做一次贪心评估（寻找 best-so-far）
EVAL_EPISODES = 20       # 每次贪心评估跑多少局（取均值，减少噪声）


class Agent:
    def __init__(self, n_state, n_action, lr, gamma, epsilon):
        self.n_action = n_action
        self.n_state  = n_state
        # Q 表初始化：大小 = MAX_DIGITIZE_NUM^n_state × n_action
        self.q_table  = np.random.uniform(
            low=0, high=1,
            size=(MAX_DIGITIZE_NUM ** n_state, n_action)
        )
        self.lr      = lr
        self.gamma   = gamma
        self.epsilon = epsilon

    # ----- 固定基座：状态离散化 -----------------------------------------------
    def bins(self, clip_min, clip_max):
        return np.linspace(clip_min, clip_max, MAX_DIGITIZE_NUM + 1)

    def make_bins_state(self, current_state):
        """
        把 4 维连续观测映射到 [0, MAX_DIGITIZE_NUM^4 - 1] 的离散 state_id。

        clip 范围选择原则：
          cart_pos  : ±2.4  正好是终止边界，全范围覆盖
          cart_v    : ±3.5  覆盖高速运动状态
          pole_angle: ±0.42 精确覆盖终止边界 0.418 rad（原来 ±0.5 浪费了末端分辨率）
          pole_v    : ±3.0  覆盖典型角速度范围（原来 ±2.0 太窄）
        """
        cart_pos, cart_v, pole_angle, pole_v = current_state

        cart_pos    = np.clip(cart_pos,    -2.4,   2.4)
        cart_v      = np.clip(cart_v,      -3.5,   3.5)
        pole_angle  = np.clip(pole_angle,  -0.42,  0.42)  # 精确覆盖终止边界
        pole_v      = np.clip(pole_v,      -3.0,   3.0)   # 放宽（原 ±2）

        s1 = np.argwhere((cart_pos   - self.bins(-2.4001,  2.4))  > 0)[-1, 0]
        s2 = np.argwhere((cart_v     - self.bins(-3.5001,  3.5))  > 0)[-1, 0]
        s3 = np.argwhere((pole_angle - self.bins(-0.4201,  0.42)) > 0)[-1, 0]
        s4 = np.argwhere((pole_v     - self.bins(-3.0001,  3.0))  > 0)[-1, 0]

        return (s1 * MAX_DIGITIZE_NUM ** 0
                + s2 * MAX_DIGITIZE_NUM ** 1
                + s3 * MAX_DIGITIZE_NUM ** 2
                + s4 * MAX_DIGITIZE_NUM ** 3)

    # =========================================================================
    # TODO 1: Q-table 更新（Bellman TD）✅
    # =========================================================================
    def update_q_table(self, current_state, action, reward, next_state):
        current_id  = self.make_bins_state(current_state)
        next_id     = self.make_bins_state(next_state)

        best_next_q = np.max(self.q_table[next_id, :])
        td_target   = reward + self.gamma * best_next_q
        td_error    = td_target - self.q_table[current_id, action]
        self.q_table[current_id, action] += self.lr * td_error

    # =========================================================================
    # TODO 2: epsilon-greedy 动作选择 ✅
    # =========================================================================
    def decide_action(self, current_state):
        current_id = self.make_bins_state(current_state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_action)
        else:
            action = np.argmax(self.q_table[current_id, :])
        return int(action)

    # ----- 固定基座：evaluate / vis 的统一接口 --------------------------------
    def predict(self, state):
        """贪心动作，用于 evaluate.py / vis.py。"""
        current_id = self.make_bins_state(state)
        return int(np.argmax(self.q_table[current_id, :]))

    def policy_info(self, state):
        """vis.py 右上角叠加显示 Q 值。"""
        current_id = self.make_bins_state(state)
        q_vals = self.q_table[current_id, :]
        return {"Q(LEFT)": f"{q_vals[0]:+.3f}", "Q(RIGHT)": f"{q_vals[1]:+.3f}"}

    # ----- 固定基座：checkpoint -------------------------------------------
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table':   self.q_table,
                'lr':        self.lr,
                'gamma':     self.gamma,
                'epsilon':   self.epsilon,
                'n_state':   self.n_state,
                'n_action':  self.n_action,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }, f)
        print(f"  [保存] → {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table  = data['q_table']
        self.lr       = data['lr']
        self.gamma    = data['gamma']
        self.epsilon  = data['epsilon']
        self.n_state  = data['n_state']
        self.n_action = data['n_action']
        print(f"Q-learning 模型已加载: {filepath}  (训练时间 {data['timestamp']})")


class Environment:
    def __init__(self):
        self.env = gym.make('CartPole-v1', max_episode_steps=TARGET)
        n_state  = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        self.agent = Agent(n_state, n_action, LR, GAMMA, EPSILON)

    # -------------------------------------------------------------------------
    def _evaluate_greedy(self, n_episodes=EVAL_EPISODES):
        """
        用当前 Q 表的纯贪心策略评估（ε=0），返回平均步数。
        使用 max_episode_steps=2000 与 evaluate.py 一致。
        """
        eval_env   = gym.make('CartPole-v1', max_episode_steps=2000)
        steps_list = []
        for _ in range(n_episodes):
            state = eval_env.reset()[0]
            steps = 0
            while True:
                action = self.agent.predict(state)
                state, _, terminated, truncated, _ = eval_env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            steps_list.append(steps)
        eval_env.close()
        return float(np.mean(steps_list))

    # -------------------------------------------------------------------------
    def train(self, epoch=EPOCHS, save_path=None):
        step_record = []
        best_mean   = -float('inf')
        best_path   = None
        if save_path is not None:
            best_path = save_path.replace('.pkl', '_best.pkl')

        for i in range(epoch):
            current_state = self.env.reset()[0]

            for step in range(TARGET):
                action     = self.agent.decide_action(current_state)
                next_state, _, terminated, truncated, _ = self.env.step(action)

                # =============================================================
                # TODO 3: reward shaping ✅
                # 终止时：step - TARGET（死得越早惩罚越大，满 2000 步惩罚为 0）
                # 存活中：0（不给正向信号，让 Q 值完全由终端惩罚反向传播驱动）
                # =============================================================
                if terminated or truncated:
                    reward = step - TARGET
                else:
                    reward = 0

                self.agent.update_q_table(current_state, action, reward, next_state)
                current_state = next_state

                if terminated or truncated:
                    step_record.append(step + 1)
                    break

            # ε 指数衰减（每轮结束后执行）
            self.agent.epsilon = max(EPSILON_MIN, self.agent.epsilon * EPSILON_DECAY)

            # 进度打印
            if (i + 1) % 500 == 0:
                recent = np.mean(step_record[-200:]) if len(step_record) >= 200 else np.mean(step_record)
                print(f"Episode {i+1:5d}  recent_200_mean={recent:.1f}  ε={self.agent.epsilon:.4f}")

            # 周期性贪心评估 + best-so-far 保存
            if best_path is not None and (i + 1) % EVAL_INTERVAL == 0:
                mean_steps = self._evaluate_greedy()
                tag = " ← 新最优！" if mean_steps > best_mean else ""
                print(f"  [Eval] ep={i+1}  greedy_mean={mean_steps:.1f}"
                      f"  best={best_mean:.1f}{tag}  ε={self.agent.epsilon:.4f}")
                if mean_steps > best_mean:
                    best_mean = mean_steps
                    self.agent.save_model(best_path)

        self.env.close()

        # 训练结束也保存最终版本
        if save_path is not None:
            self.agent.save_model(save_path)

        print(f"\n训练完成。best-so-far 贪心均值 = {best_mean:.1f}")
        print(f"最后 200 轮训练步数均值 = {np.mean(step_record[-200:]):.1f}")
        return step_record


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    env = Environment()
    print("=" * 60)
    print("Q-Learning Bonus 2 优化版  开始训练")
    print(f"  MAX_DIGITIZE_NUM = {MAX_DIGITIZE_NUM}  ({MAX_DIGITIZE_NUM**4} states)")
    print(f"  ε: {EPSILON} → {EPSILON_MIN}  decay={EPSILON_DECAY}  epochs={EPOCHS}")
    print(f"  LR={LR}  GAMMA={GAMMA}  TARGET={TARGET}")
    print("=" * 60)

    save_path = os.path.join(SAVE_DIR, 'q_learning_bonus2.pkl')
    record    = env.train(epoch=EPOCHS, save_path=save_path)

    # 绘制训练曲线（原始 + 50 轮滑动均值）
    plt.figure(figsize=(13, 5))
    plt.plot(record, alpha=0.25, color='steelblue', label='raw steps')
    if len(record) >= 50:
        ma50 = np.convolve(record, np.ones(50) / 50, mode='valid')
        plt.plot(np.arange(49, len(record)), ma50,
                 color='steelblue', linewidth=2, label='50-ep MA')
    plt.axhline(500, color='red', linestyle='--', linewidth=1.2, label='Bonus 2 target (500)')
    plt.title('Q-Learning Bonus 2 Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, 'q_learning_bonus2_curve.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"训练曲线已保存到: {plot_path}")

    print("\n评测命令（加载 best-so-far checkpoint）：")
    print("python evaluate.py \\")
    print("    --agent-class train_qlearning_bonus2:Agent \\")
    print("    --agent-init-kwargs "
          "'{\"n_state\":4,\"n_action\":2,\"lr\":0.08,\"gamma\":0.99,\"epsilon\":0.0}' \\")
    print(f"    --checkpoint {SAVE_DIR}/q_learning_bonus2_best.pkl \\")
    print("    --seed-base 42 --seed-count 100 --max-episode-steps 2000")


if __name__ == "__main__":
    main()


# =============================================================================
# 【Bonus 2 说明】我改了什么、为什么（~200 字）
#
# ① 离散化精度：MAX_DIGITIZE_NUM 从 6 提升到 10，状态数从 1296 → 10000。
#   更细的网格让 Q 表能区分更细微的状态差异（如杆子角度差 0.08 rad），
#   避免原来"不同物理状态映射到同一格"造成的 Q 值混淆。
#
# ② ε 指数衰减：ε 从 0.5 线性衰减到 0.005（6000 轮 decay=0.9993）。
#   前期高 ε 保证充分探索 10000 个状态；后期低 ε 让智能体专注利用已学策略，
#   避免原版固定 ε=0.1 导致"训练后期还在随机破坏已收敛策略"的问题。
#
# ③ best-so-far 保存：每 300 轮做一次纯贪心评估（20 局 × 2000 步），
#   只在均值创历史新高时保存 checkpoint。这样最终提交的是训练过程中
#   性能最峰值的 Q 表，而非训练结束时可能已在震荡下行的版本。
#
# ④ 状态 clip 范围调整：pole_angle 从 ±0.5 收窄到 ±0.42（精确覆盖终止
#   边界 0.418 rad），把 10 个 bin 的分辨率都用在有效区域；pole_v 从
#   ±2 放宽到 ±3 避免高速状态被截断到同一个格。
# =============================================================================