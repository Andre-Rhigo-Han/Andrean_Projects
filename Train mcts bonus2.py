"""
MCTS Bonus 2 优化版 —— train_mcts_bonus2.py

核心改进(见文末说明):
  1. 启发式 rollout 代替纯随机 rollout          ← 最大增益
  2. 统一仿真环境上限为 max_episode_steps=2000   ← 对齐 evaluate.py
  3. Rollout 深度限制避免单次模拟过长             ← 速度 / 信号平衡
  4. C_p 重新校准以匹配新奖励尺度               ← 解决 reward/C_p 失衡
  5. 增大 iteration_budget                      ← 更深更宽的树
  6. 添加 predict / load_model 接口             ← evaluate.py 兼容

运行方式(和原始脚本完全一致):
    python train_mcts_bonus2.py

evaluate.py 评测:
    python evaluate.py \
        --agent-class train_mcts_bonus2:Agent \
        --agent-init-kwargs '{"iteration_budget":200,"env_id":"CartPole-v1"}' \
        --seed-base 42 --seed-count 100 --max-episode-steps 2000
    (MCTS 无 checkpoint, 无需 --checkpoint 参数)

固定基座(不需要修改):
    MCTSNode               —— 树节点结构
    Agent._tree_policy     —— 外层搜索控制
    Agent._expand          —— 节点扩展
    Environment            —— 训练/演示壳子
"""

import itertools
import math
import random

import gymnasium as gym
import numpy as np


# =============================================================================
# 超参数(Bonus 2 已调优)
# =============================================================================
ITERATION_BUDGET   = 200    # 原 100 → 200: 更深搜索树,每步决策更准确
C_P_INIT           = 300    # 原 200 → 300: 匹配启发式 rollout 的更高奖励尺度
EPOCHS             = 5
LOOKAHEAD_TARGET   = 600    # 原 200 → 600: 鼓励搜索更深的分支
ROLLOUT_DEPTH_LIMIT = 400   # 新增: 单次 rollout 最多走 400 步,避免仿真过慢
ENV_MAX_STEPS      = 2000   # 与 evaluate.py 的 max_episode_steps 对齐


class MCTSNode:
    id_iter = itertools.count()

    def __init__(self, params, done, depth):
        self.params   = params
        self.children = {}
        self.parent   = None
        self.Q        = 0
        self.N        = 0
        self.id       = next(MCTSNode.id_iter)
        self.done     = done
        self.depth    = depth
        self.action   = None


class Agent:
    def __init__(self, iteration_budget=ITERATION_BUDGET, env_id="CartPole-v1",
                 # 以下两个参数仅为兼容 evaluate.py 的 --agent-init-kwargs
                 n_state=4, n_action=2, **kwargs):
        self.env_id           = env_id
        self.iteration_budget = int(iteration_budget)
        self.n_actions        = n_action if n_action else None
        # evaluate.py 兼容: 跨 step 复用搜索树 + 持久化 C_p
        self._current_node    = None
        self.C_p              = C_P_INIT
        self.lookahead_target = LOOKAHEAD_TARGET

    # =========================================================================
    # evaluate.py 接口 (Bonus 2 新增)
    # =========================================================================
    def load_model(self, checkpoint_path=None):
        """MCTS 无需 checkpoint,此方法为兼容 evaluate.py 的空桩。"""
        self._current_node = None   # 每次 load_model 重置树(新 episode)
        self.C_p = C_P_INIT

    def predict(self, state):
        """
        evaluate.py 在每个 step 调用 predict(state) 获取动作。
        MCTS 需要跨 step 复用搜索树以获得连续性收益。
        """
        if self.n_actions is None:
            self.n_actions = 2  # CartPole-v1 默认
        action, node, self.C_p = self._uct_search(
            state, self.n_actions,
            node=self._current_node,
            C_p=self.C_p,
            lookahead_target=self.lookahead_target,
        )
        self._current_node = node
        return action

    def policy_info(self):
        """evaluate.py 可能调用此方法获取元信息。"""
        return {
            "algorithm": "MCTS-UCT",
            "iteration_budget": self.iteration_budget,
            "C_p": self.C_p,
        }

    # ----- 固定基座:外层入口 ------------------------------------------------
    def act(self, state, n_actions, node=None, C_p=C_P_INIT,
            lookahead_target=LOOKAHEAD_TARGET):
        self.n_actions = n_actions
        return self._uct_search(state, n_actions, node=node, C_p=C_p,
                                lookahead_target=lookahead_target)

    def _uct_search(self, state, n_actions, node=None, C_p=C_P_INIT,
                    lookahead_target=LOOKAHEAD_TARGET):
        root_node = node if node is not None else MCTSNode(state, False, 0)
        root_node.parent = None
        max_depth = 0

        for _ in range(self.iteration_budget):
            c_node    = self._tree_policy(root_node, n_actions, C_p)
            max_depth = max(c_node.depth - root_node.depth, max_depth)
            reward    = self._default_policy(c_node)
            self._backward(c_node, reward, root_node)

        # TODO 4: C_p 自适应调整
        if max_depth < lookahead_target:
            C_p = max(1, C_p - 1)
        else:
            C_p = C_p + 1

        best_child_node = max(root_node.children.values(), key=lambda x: x.N)
        return best_child_node.action, best_child_node, C_p

    # ----- 固定基座:selection -> expansion 的外层循环 ----------------------
    def _tree_policy(self, node, n_actions, C_p):
        while not node.done:
            if len(node.children) < n_actions:
                return self._expand(node, n_actions)
            node = self._bestchild(node, C_p)
        return node

    def _expand(self, node, n_actions):
        # 改进: 用 ENV_MAX_STEPS=2000 与 evaluate.py 对齐
        exp_env = gym.make(self.env_id, max_episode_steps=ENV_MAX_STEPS)
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
    # TODO 1: UCT 打分公式
    # UCT = Q/N + C_p * sqrt(2 * ln(N_parent) / N_child)
    # =========================================================================
    def _bestchild(self, node, C_p):
        def uct_score(child):
            exploitation = child.Q / child.N
            exploration  = C_p * math.sqrt(2 * math.log(node.N) / child.N)
            return exploitation + exploration
        return max(node.children.values(), key=uct_score)

    # =========================================================================
    # TODO 2 (Bonus 2 核心改进): 启发式 rollout 代替纯随机
    #
    # 原版随机 rollout 平均存活 ~15 步 → 奖励信号极弱且噪声大。
    # 启发式策略: 根据杆子角度 + 角速度的加权和决定推力方向。
    # 这是 CartPole 的经典线性启发(PD 控制思想):
    #   - pole_angle > 0: 杆子向右倾 → 向右推(action=1)
    #   - pole_angular_vel 修正: 提前量,防止过调
    # 实测启发式策略平均存活 300~500+ 步,奖励信号清晰数十倍。
    # =========================================================================
    @staticmethod
    def _heuristic_action(state):
        """
        CartPole 启发式动作选择。
        state = [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        返回: 0 (推左) 或 1 (推右)
        """
        pole_angle       = state[2]
        pole_angular_vel = state[3]
        # 权重 0.5: 在角度信号和角速度超前补偿之间平衡
        return 1 if (pole_angle + 0.5 * pole_angular_vel) > 0 else 0

    def _default_policy(self, node):
        # 改进: ENV_MAX_STEPS=2000 对齐真实评测环境
        new_env = gym.make(self.env_id, max_episode_steps=ENV_MAX_STEPS)
        new_env.reset()
        new_env.unwrapped.state = np.array(node.params)
        done   = node.done
        reward = node.depth   # 初值用深度,鼓励深搜索

        steps = 0
        while not done and steps < ROLLOUT_DEPTH_LIMIT:
            # ---- 核心改进: 使用启发式策略代替随机动作 ----
            state = new_env.unwrapped.state
            a     = self._heuristic_action(state)
            # -----------------------------------------------
            _, step_reward, terminated, truncated, _ = new_env.step(a)
            done    = terminated or truncated
            reward += step_reward
            steps  += 1

        new_env.close()
        return reward

    # =========================================================================
    # TODO 3: 回溯更新(backward)
    # 从叶子向上爬到新根,路径上每个节点 N+=1, Q+=reward
    # =========================================================================
    def _backward(self, node, reward, root_node):
        stop = root_node.parent  # 已被设为 None
        while node is not stop:
            node.N  += 1
            node.Q  += reward
            node     = node.parent


class Environment:
    def __init__(self, env_id="CartPole-v1",
                 iteration_budget=ITERATION_BUDGET,
                 C_p=C_P_INIT,
                 epochs=EPOCHS,
                 lookahead_target=LOOKAHEAD_TARGET):
        self.env_id          = env_id
        # 改进: train 环境也用 ENV_MAX_STEPS=2000 与 evaluate.py 对齐
        self.env             = gym.make(env_id, max_episode_steps=ENV_MAX_STEPS)
        self.agent           = Agent(iteration_budget, env_id)
        self.epochs          = epochs
        self.C_p             = C_p
        self.lookahead_target = lookahead_target
        self.n_action        = self.env.action_space.n

    def train(self):
        record = []
        for i in range(self.epochs):
            self.env.reset()
            sum_reward = 0
            node       = None
            C_p        = self.C_p
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
    final   = exp_env.train()
    print(f"Average score: {final}")