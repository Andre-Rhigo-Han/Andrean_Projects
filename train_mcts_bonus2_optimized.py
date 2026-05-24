"""
MCTS Bonus 2 高阶优化版 —— train_mcts_bonus2_optimized.py

核心改进 (完美解决老师提到的 "UCT reward 尺度与 C_p 探索项全局失衡" 痛点):
  1. UCT 归一化: 将 Q/N 除以 ROLLOUT_DEPTH_LIMIT，将 Exploitation 项严格约束在 [0, 1] 附近。
  2. C_p 尺度重构与平滑自适应: 配合归一化，将初始 C_p 设为 0.5，自适应步长改为 0.05。
  3. Lookahead Target 合理化: 原版 600 对于 200 budget 是不可能达到的，修正为符合逻辑的深度 15。
  4. 改进的带噪 PD 控制启发式: 加入 x 和 x_dot 考量防止出界，同时引入 10% 噪声确保 Rollout 多样性。
  5. 状态校验树复用 (State-Verified Tree Reuse): predict 接口增加 np.allclose 校验，完美兼容 evaluate.py 并行重置。
"""

import itertools
import math
import random
import gymnasium as gym
import numpy as np

# =============================================================================
# 超参数 (Bonus 2 高阶调优版)
# =============================================================================
ITERATION_BUDGET   = 200    # 每次决策构建 200 个节点
C_P_INIT           = 0.5    # UCT 归一化后的经典探索系数
EPOCHS             = 5
LOOKAHEAD_TARGET   = 15     # 对应 200 budget 下的合理期望深度
ROLLOUT_DEPTH_LIMIT = 400   # 单次 rollout 最多走 400 步
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
                 n_state=4, n_action=2, **kwargs):
        self.env_id           = env_id
        self.iteration_budget = int(iteration_budget)
        self.n_actions        = n_action if n_action else 2
        
        # evaluate.py 兼容: 跨 step 复用搜索树
        self._current_node    = None
        self.C_p              = C_P_INIT
        self.lookahead_target = LOOKAHEAD_TARGET

    def load_model(self, checkpoint_path=None):
        self._current_node = None
        self.C_p = C_P_INIT

    def predict(self, state):
        if self.n_actions is None:
            self.n_actions = 2
            
        # [核心改进 5]: 状态校验，防止 evaluate.py 重置环境后继续使用上一个 episode 的残留树
        if self._current_node is not None:
            if not np.allclose(self._current_node.params, state, atol=1e-3):
                self._current_node = None
                self.C_p = C_P_INIT

        action, node, self.C_p = self._uct_search(
            state, self.n_actions,
            node=self._current_node,
            C_p=self.C_p,
            lookahead_target=self.lookahead_target,
        )
        self._current_node = node
        return action

    def policy_info(self):
        return {
            "algorithm": "MCTS-UCT-Normalized",
            "iteration_budget": self.iteration_budget,
            "C_p": self.C_p,
        }

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

        # [核心改进 3]: C_p 连续自适应调整，步长缩减以匹配归一化尺度
        if max_depth < lookahead_target:
            C_p = max(0.1, C_p - 0.05)
        else:
            C_p = min(2.0, C_p + 0.05)

        best_child_node = max(root_node.children.values(), key=lambda x: x.N)
        return best_child_node.action, best_child_node, C_p

    def _tree_policy(self, node, n_actions, C_p):
        while not node.done:
            if len(node.children) < n_actions:
                return self._expand(node, n_actions)
            node = self._bestchild(node, C_p)
        return node

    def _expand(self, node, n_actions):
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

    def _bestchild(self, node, C_p):
        def uct_score(child):
            # [核心改进 1]: UCT 归一化。通过除以 ROLLOUT_DEPTH_LIMIT 消除 Reward 尺度暴胀
            exploitation = (child.Q / child.N) / ROLLOUT_DEPTH_LIMIT
            exploration  = C_p * math.sqrt(2 * math.log(node.N) / child.N)
            return exploitation + exploration
        return max(node.children.values(), key=uct_score)

    @staticmethod
    def _heuristic_action(state):
        # [核心改进 4]: 完整的带噪 PD 控制。加入 x 和 x_dot 考量防止冲出边界
        cart_pos, cart_vel, pole_angle, pole_angular_vel = state
        score = pole_angle + 0.5 * pole_angular_vel - 0.01 * cart_pos - 0.05 * cart_vel
        
        # 10% 的随机性：极大幅度增加 MCTS 探索不同路径的能力
        if random.random() < 0.1:
            return random.choice([0, 1])
        return 1 if score > 0 else 0

    def _default_policy(self, node):
        new_env = gym.make(self.env_id, max_episode_steps=ENV_MAX_STEPS)
        new_env.reset()
        new_env.unwrapped.state = np.array(node.params)
        done   = node.done
        reward = node.depth   

        steps = 0
        while not done and steps < ROLLOUT_DEPTH_LIMIT:
            state = new_env.unwrapped.state
            a     = self._heuristic_action(state)
            _, step_reward, terminated, truncated, _ = new_env.step(a)
            done    = terminated or truncated
            reward += step_reward
            steps  += 1

        new_env.close()
        return reward

    def _backward(self, node, reward, root_node):
        stop = root_node.parent
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