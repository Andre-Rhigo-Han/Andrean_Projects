"""
MCTS 学生填空版。

目标:
    通过四个 TODO 实现 UCT 版 MCTS,在 CartPole-v1 上做在线规划并保持平衡。
    MCTS 不需要 checkpoint,可以直接跑本文件看分数,或用 vis_mcts.py 录视频:

    python train_mcts.py                         # 跑几轮看分数
    python vis_mcts.py --episodes 1              # 录一段视频(默认指向 backup,
                                                 # 学生可改成 train_mcts 验证)

固定基座(不需要修改):
    MCTSNode               —— 树节点结构
    Agent.__init__ / act   —— 入口和状态管理
    _tree_policy           —— 外层搜索控制
    _expand                —— 节点扩展
    Environment            —— 训练/演示壳子
"""

import itertools
import math
import random

import gymnasium as gym
import numpy as np


# =============================================================================
# 超参数(学生可以调)
# =============================================================================
ITERATION_BUDGET = 100
C_P_INIT = 200
EPOCHS = 5
LOOKAHEAD_TARGET = 200


class MCTSNode:
    id_iter = itertools.count()

    def __init__(self, params, done, depth):
        self.params = params
        self.children = {}
        self.parent = None
        self.Q = 0
        self.N = 0
        self.id = next(MCTSNode.id_iter)
        self.done = done
        self.depth = depth
        self.action = None


class Agent:
    def __init__(self, iteration_budget, env_id):
        self.env_id = env_id
        self.iteration_budget = int(iteration_budget)
        self.n_actions = None

    # ----- 固定基座:外层入口 ----------------------------------------------
    def act(self, state, n_actions, node=None, C_p=C_P_INIT, lookahead_target=LOOKAHEAD_TARGET):
        self.n_actions = n_actions
        return self._uct_search(state, n_actions, node=node, C_p=C_p,
                                lookahead_target=lookahead_target)

    def _uct_search(self, state, n_actions, node=None, C_p=C_P_INIT, lookahead_target=LOOKAHEAD_TARGET):
        root_node = node if node is not None else MCTSNode(state, False, 0)
        # 复用 best_child 做新根时切断父链,避免老树挂着 + backward 越界。
        root_node.parent = None
        max_depth = 0

        for _ in range(self.iteration_budget):
            c_node = self._tree_policy(root_node, n_actions, C_p)
            max_depth = max(c_node.depth - root_node.depth, max_depth)
            reward = self._default_policy(c_node)
            self._backward(c_node, reward, root_node)

        # =====================================================================
        # TODO 4: C_p 自适应调整 ✅ 已实现
        # 如果搜索深度不够,减小 C_p(减少探索宽度,逼着往深走);
        # 如果深度已经充足,适当加大 C_p(鼓励探索更多分支)。
        # =====================================================================
        # ------ 你的代码开始 ------
        if max_depth < lookahead_target:
            C_p = max(1, C_p - 1)   # 深度不足 -> 减小 C_p
        else:
            C_p = C_p + 1           # 深度充足 -> 加大 C_p
        # ------ 你的代码结束 ------

        best_child_node = max(root_node.children.values(), key=lambda x: x.N)
        return best_child_node.action, best_child_node, C_p

    # ----- 固定基座:selection -> expansion 的外层循环 --------------------
    def _tree_policy(self, node, n_actions, C_p):
        while not node.done:
            if len(node.children) < n_actions:
                return self._expand(node, n_actions)
            node = self._bestchild(node, C_p)
        return node

    def _expand(self, node, n_actions):
        exp_env = gym.make(self.env_id)
        exp_env.reset()
        exp_env.unwrapped.state = np.array(node.params)

        unchosen_actions = [a for a in range(n_actions) if a not in node.children]
        a = random.choice(unchosen_actions)
        params, _, terminated, truncated, _ = exp_env.step(a)
        done = terminated or truncated
        child_node = MCTSNode(params, done, node.depth + 1)
        child_node.parent = node
        child_node.action = a
        node.children[a] = child_node
        exp_env.close()
        return child_node

    # =========================================================================
    # TODO 1: UCT 打分公式(bestchild 的核心) ✅ 已实现
    # UCT = Q/N + C_p * sqrt(2 * ln(N_parent) / N_child)
    # =========================================================================
    def _bestchild(self, node, C_p):
        # ------ 你的代码开始 ------
        def uct_score(child):
            exploitation = child.Q / child.N
            exploration = C_p * math.sqrt(2 * math.log(node.N) / child.N)
            return exploitation + exploration

        return max(node.children.values(), key=uct_score)
        # ------ 你的代码结束 ------

    # =========================================================================
    # TODO 2: 默认策略 rollout ✅ 已实现
    # 从 node 的状态出发随机走到 episode 终止,累积 reward。
    # =========================================================================
    def _default_policy(self, node):
        # ------ 你的代码开始 ------
        new_env = gym.make(self.env_id)
        new_env.reset()
        new_env.unwrapped.state = np.array(node.params)
        done = node.done
        reward = node.depth    # 用深度作初值,鼓励深搜索
        while not done:
            a = random.randrange(self.n_actions)
            _, step_reward, terminated, truncated, _ = new_env.step(a)
            done = terminated or truncated
            reward += step_reward
        new_env.close()
        return reward
        # ------ 你的代码结束 ------

    # =========================================================================
    # TODO 3: 回溯更新(backward) ✅ 已实现
    # 从叶子向上爬到新根,路径上每个节点 N+=1, Q+=reward。
    # =========================================================================
    def _backward(self, node, reward, root_node):
        stop = root_node.parent  # 已被设为 None
        # ------ 你的代码开始 ------
        while node is not stop:
            node.N += 1
            node.Q += reward
            node = node.parent
        # ------ 你的代码结束 ------


class Environment:
    def __init__(self, env_id="CartPole-v1", iteration_budget=ITERATION_BUDGET,
                 C_p=C_P_INIT, epochs=EPOCHS, lookahead_target=LOOKAHEAD_TARGET):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.agent = Agent(iteration_budget, env_id)
        self.epochs = epochs
        self.C_p = C_p
        self.lookahead_target = lookahead_target
        self.n_action = self.env.action_space.n

    def train(self):
        record = []
        for i in range(self.epochs):
            self.env.reset()
            sum_reward = 0
            node = None
            C_p = self.C_p
            while True:
                action, node, C_p = self.agent.act(
                    self.env.unwrapped.state,
                    n_actions=self.n_action,
                    node=node,
                    C_p=C_p,
                    lookahead_target=self.lookahead_target,
                )
                _, reward, terminated, truncated, _ = self.env.step(action)
                sum_reward += reward
                if terminated or truncated:
                    break
            record.append(sum_reward)
            print(f"Epoch {i}: Score = {sum_reward}")
        self.env.close()
        return np.mean(record)


if __name__ == "__main__":
    exp_env = Environment()
    final = exp_env.train()
    print(f"Average score: {final}")
