"""
MCTS Bonus 2 强化优化版 —— train_mcts_bonus2_optimized.py

目标：在不修改 evaluate.py / vis.py / vis_mcts.py 的前提下，把 MCTS 路线从
“随机 rollout + 大 C_p”改成“模型内仿真 + 归一化 UCT + 控制器引导 rollout”。

运行：
    python train_mcts_bonus2_optimized.py

若 evaluate.py 支持动态加载 MCTS Agent，可用：
    python evaluate.py \
        --agent-class train_mcts_bonus2_optimized:Agent \
        --agent-init-kwargs '{"iteration_budget":96,"env_id":"CartPole-v1"}' \
        --seed-base 42 --seed-count 100 --max-episode-steps 2000

说明：
    MCTS 无需 checkpoint；load_model 是为了兼容统一评测接口的空桩。
"""

import itertools
import math
import random
from typing import Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np


# =============================================================================
# Bonus 2 超参数
# =============================================================================
ENV_ID = "CartPole-v1"
ENV_MAX_STEPS = 2000
EPOCHS = 5

# 不再盲目把 budget 拉很大。内部动力学仿真很快，96 通常已经比原版稳定。
# 若本机性能允许，可以在命令行或代码中改成 128 / 160。
ITERATION_BUDGET = 96

# 关键修正：rollout value 已归一化到约 [0, 1]，C_p 不应再用 200/300 量级。
C_P_INIT = 1.25
C_P_MIN = 0.15
C_P_MAX = 3.00
LOOKAHEAD_TARGET = 260

# rollout 越长，信号越准但越慢；有内部动力学后 420 仍可接受。
ROLLOUT_DEPTH_LIMIT = 420

# 根节点最终动作用短程安全复核，解决“按访问次数 N 选动作”带来的偏差。
ROOT_VERIFY_DEPTH = 520
ROOT_VERIFY_WEIGHT = 0.55
TREE_VALUE_WEIGHT = 0.45

# 根节点增加一个很短的 beam-MPC 复核。它仍然使用同一套 CartPole 内部模型，
# 只是在最终动作选择时补足普通 rollout 对“小车慢慢漂移”的感知不足。
USE_BEAM_MPC_ROOT = True
MPC_HORIZON = 18
MPC_BEAM_WIDTH = 64

# 是否启用一个很轻的安全控制器直通。
# True：更容易跑出亮眼成绩；False：更接近纯 MCTS，但成绩通常低一些。
USE_SAFE_CONTROLLER_FAST_PATH = False
FAST_PATH_SAFE_X = 1.80
FAST_PATH_SAFE_THETA = 0.30

# 失败惩罚与稳定性 shaping 只用于树内评分，不改变真实环境 reward。
FAILURE_PENALTY = 0.35
STABILITY_BONUS_WEIGHT = 0.08


class MCTSNode:
    id_iter = itertools.count()

    def __init__(self, params, done: bool, depth: int):
        self.params = np.asarray(params, dtype=np.float64)
        self.children = {}
        self.parent = None
        self.Q = 0.0          # 累计归一化 value
        self.N = 0
        self.id = next(MCTSNode.id_iter)
        self.done = bool(done)
        self.depth = int(depth)
        self.action = None


class Agent:
    def __init__(self, iteration_budget=ITERATION_BUDGET, env_id=ENV_ID,
                 n_state=4, n_action=2, **kwargs):
        self.env_id = env_id
        self.iteration_budget = int(iteration_budget)
        self.n_actions = int(n_action) if n_action else 2
        self._current_node: Optional[MCTSNode] = None
        self.C_p = float(kwargs.get("C_p", C_P_INIT))
        self.lookahead_target = int(kwargs.get("lookahead_target", LOOKAHEAD_TARGET))

        # CartPole 线性控制器组合。符号含义：
        #   score > 0 -> action 1(向右推)；score <= 0 -> action 0(向左推)
        # x / x_dot 用负权重防止小车漂到边界；theta / theta_dot 用正权重扶杆。
        self.policy_weights = [
            np.array([-0.35, -0.65, 7.5, 1.10], dtype=np.float64),
            np.array([-0.50, -0.85, 9.0, 1.35], dtype=np.float64),
            np.array([-0.70, -1.05, 11.0, 1.65], dtype=np.float64),
            np.array([-0.95, -1.35, 13.0, 2.00], dtype=np.float64),
            np.array([-1.20, -1.60, 15.0, 2.35], dtype=np.float64),
            np.array([0.00, 0.00, 1.0, 0.50], dtype=np.float64),  # 原启发式兜底
        ]

    # =========================================================================
    # evaluate.py 兼容接口
    # =========================================================================
    def load_model(self, checkpoint_path=None):
        """MCTS 不需要 checkpoint；这里只重置跨 step 复用树。"""
        self._current_node = None
        self.C_p = C_P_INIT

    def predict(self, state):
        """统一评测接口：输入连续状态，输出离散动作。"""
        state = np.asarray(state, dtype=np.float64)

        # 新 episode 或外部状态与树根不一致时，丢弃旧树，避免跨局污染。
        if self._current_node is not None:
            mismatch = np.linalg.norm(state - self._current_node.params) > 1e-6
            if self._current_node.done or mismatch:
                self._current_node = None

        action, node, self.C_p = self._uct_search(
            state,
            self.n_actions,
            node=self._current_node,
            C_p=self.C_p,
            lookahead_target=self.lookahead_target,
        )
        self._current_node = node
        return int(action)

    def policy_info(self):
        return {
            "algorithm": "MCTS-UCT + normalized value + guided rollout",
            "iteration_budget": self.iteration_budget,
            "C_p": round(float(self.C_p), 4),
            "rollout_depth_limit": ROLLOUT_DEPTH_LIMIT,
            "root_verify_depth": ROOT_VERIFY_DEPTH,
            "fast_path": USE_SAFE_CONTROLLER_FAST_PATH,
            "beam_mpc_root": USE_BEAM_MPC_ROOT,
        }

    # =========================================================================
    # 原 train_mcts.py 入口兼容
    # =========================================================================
    def act(self, state, n_actions, node=None, C_p=C_P_INIT,
            lookahead_target=LOOKAHEAD_TARGET):
        self.n_actions = int(n_actions)
        state = np.asarray(state, dtype=np.float64)
        if node is not None and np.linalg.norm(state - node.params) > 1e-6:
            node = None
        return self._uct_search(state, n_actions, node=node, C_p=C_p,
                                lookahead_target=lookahead_target)

    def _uct_search(self, state, n_actions, node=None, C_p=C_P_INIT,
                    lookahead_target=LOOKAHEAD_TARGET):
        root_node = node if node is not None else MCTSNode(state, False, 0)
        root_node.parent = None
        max_depth = 0

        # 可选：在安全区域内直接使用“经过根节点复核的控制器动作”。
        # 这样可避免每一个稳定状态都做大量树搜索，实际 2000 步评测会快很多。
        if USE_SAFE_CONTROLLER_FAST_PATH and self._is_safe_region(root_node.params):
            action = self._verified_controller_action(root_node.params, root_node.depth)
            child = self._expand_with_action(root_node, action, n_actions)
            return child.action, child, C_p

        # 确保根节点两个动作都至少被扩展一次，避免随机扩展顺序影响第一轮选择。
        for a in range(n_actions):
            if a not in root_node.children:
                self._expand_with_action(root_node, a, n_actions)

        for _ in range(self.iteration_budget):
            c_node = self._tree_policy(root_node, n_actions, C_p)
            max_depth = max(max_depth, c_node.depth - root_node.depth)
            value = self._default_policy(c_node)
            self._backward(c_node, value, root_node)

        # C_p 自适应仍保留，但改为乘法调节 + clamp，适配归一化 value。
        if max_depth < lookahead_target:
            C_p *= 0.97
        else:
            C_p *= 1.015
        C_p = float(np.clip(C_p, C_P_MIN, C_P_MAX))

        # 原版按 N 选 child 容易把“被探索多”的分支误当成“好分支”。
        # 这里用：树内平均 value + 根节点安全复核 value。
        best_child_node = max(
            root_node.children.values(),
            key=lambda child: self._final_child_score(root_node, child),
        )
        return best_child_node.action, best_child_node, C_p

    # =========================================================================
    # MCTS: selection / expansion / rollout / backup
    # =========================================================================
    def _tree_policy(self, node, n_actions, C_p):
        while not node.done:
            if len(node.children) < n_actions:
                return self._expand(node, n_actions)
            node = self._bestchild(node, C_p)
        return node

    def _expand(self, node, n_actions):
        unchosen_actions = [a for a in range(n_actions) if a not in node.children]
        a = random.choice(unchosen_actions)
        return self._expand_with_action(node, a, n_actions)

    def _expand_with_action(self, node, action, n_actions):
        if action in node.children:
            return node.children[action]

        next_state, done = self._model_step(node.params, action, node.depth)
        child_node = MCTSNode(next_state, done, node.depth + 1)
        child_node.parent = node
        child_node.action = int(action)
        node.children[int(action)] = child_node
        return child_node

    def _bestchild(self, node, C_p):
        # PUCT 风格：UCT 探索项乘以先验，避免在明显危险动作上浪费太多 budget。
        # Q/N 已经是归一化 value，C_p 使用 1 左右才合理。
        parent_N = max(1, node.N)

        def score(child):
            if child.N == 0:
                mean_value = 0.0
            else:
                mean_value = child.Q / child.N
            prior = self._policy_prior(node.params, child.action)
            exploration = C_p * prior * math.sqrt(parent_N) / (1 + child.N)
            return mean_value + exploration

        return max(node.children.values(), key=score)

    def _default_policy(self, node):
        """
        从 node 状态出发，用“控制器组合”做 rollout，并返回归一化 value。
        归一化后 UCT 的 exploitation 与 exploration 在同一数量级，避免 Bonus 提示中
        提到的 reward 尺度 / C_p 全局失衡。
        """
        if node.done:
            return self._terminal_value(node.depth)

        # 用 node.id 和访问次数切换 rollout 控制器，使估值不是单一启发式。
        policy_index = (node.id + node.N) % len(self.policy_weights)
        weights = self.policy_weights[policy_index]

        state = np.asarray(node.params, dtype=np.float64)
        depth = node.depth
        survived = 0
        stability_acc = 0.0
        done = False

        while (not done and survived < ROLLOUT_DEPTH_LIMIT
               and depth < ENV_MAX_STEPS):
            action = self._linear_policy_action(state, weights)
            state, done = self._model_step(state, action, depth)
            depth += 1
            survived += 1
            stability_acc += self._stability_score(state)

        # 核心 value：预计已走到的深度占 2000 上限的比例。
        value = depth / ENV_MAX_STEPS

        if survived > 0:
            value += STABILITY_BONUS_WEIGHT * (stability_acc / survived)

        # 若在 rollout 内失败，额外惩罚“很快倒”的分支。
        if done and depth < ENV_MAX_STEPS:
            value -= FAILURE_PENALTY * (1.0 - survived / max(1, ROLLOUT_DEPTH_LIMIT))
        elif depth >= ENV_MAX_STEPS:
            value += 0.20  # 到达 2000 上限，给满分分支额外奖励

        return float(value)

    def _backward(self, node, value, root_node):
        stop = root_node.parent  # root_node.parent 已被置为 None
        while node is not stop:
            node.N += 1
            node.Q += value
            node = node.parent

    # =========================================================================
    # 根节点最终选择与控制器引导
    # =========================================================================
    def _final_child_score(self, root, child):
        tree_value = child.Q / child.N if child.N > 0 else 0.0
        verify_value = self._portfolio_rollout_value(
            child.params,
            child.depth,
            child.done,
            depth_limit=ROOT_VERIFY_DEPTH,
        )
        prior = self._policy_prior(root.params, child.action)
        return TREE_VALUE_WEIGHT * tree_value + ROOT_VERIFY_WEIGHT * verify_value + 0.02 * prior

    def _verified_controller_action(self, state, depth):
        """直接比较左右两个根动作后接控制器组合的短程生存价值。"""
        best_action = 0
        best_value = -1e9
        for action in range(self.n_actions):
            next_state, done = self._model_step(state, action, depth)
            value = self._portfolio_rollout_value(
                next_state,
                depth + 1,
                done,
                depth_limit=ROOT_VERIFY_DEPTH,
            )
            # 轻微加入控制器先验，平局时选择更自然的动作。
            value += 0.01 * self._policy_prior(state, action)
            if value > best_value:
                best_value = value
                best_action = action

        if USE_BEAM_MPC_ROOT:
            beam_action, beam_value = self._beam_search_action_value(state, depth)
            # beam_value 是局部稳定性积分，量纲与 portfolio value 不同；这里不直接相加，
            # 只在 beam 明显能给出动作时作为根节点 tie-break / rescue。
            beam_next, beam_done = self._model_step(state, beam_action, depth)
            beam_verify = self._portfolio_rollout_value(
                beam_next, depth + 1, beam_done, depth_limit=ROOT_VERIFY_DEPTH
            )
            if beam_verify >= best_value - 0.015:
                best_action = beam_action

        return int(best_action)

    def _beam_search_action_value(self, state, depth):
        """
        短视野 beam-MPC：保留若干条局部最稳的动作序列，返回最佳序列的首动作。
        这个函数只用于根节点复核，不替代 MCTS 的 selection / rollout / backup。
        """
        init_state = np.asarray(state, dtype=np.float64)
        # item = (score, state, done, first_action, depth)
        beam = [(0.0, init_state, False, None, int(depth))]

        for _ in range(MPC_HORIZON):
            candidates = []
            for score, s, done, first_action, d in beam:
                if done:
                    candidates.append((score - 5.0, s, done, first_action, d))
                    continue
                for action in range(self.n_actions):
                    ns, nd = self._model_step(s, action, d)
                    fa = int(action) if first_action is None else first_action
                    # 局部目标：杆直、小车居中、速度不过大。
                    local = self._stability_score(ns)
                    # 对接近边界与终止动作强惩罚。
                    x, x_dot, theta, theta_dot = ns
                    margin_penalty = 0.20 * (abs(x) / 2.4) + 0.25 * (abs(theta) / (12 * 2 * math.pi / 360))
                    terminal_penalty = 6.0 if (nd and d + 1 < ENV_MAX_STEPS) else 0.0
                    candidates.append((
                        score + local - margin_penalty - terminal_penalty,
                        ns, nd, fa, d + 1,
                    ))

            # 额外用当前状态稳定性排序，避免只看累计分而留下速度很危险的状态。
            candidates.sort(
                key=lambda item: item[0] + 0.8 * self._stability_score(item[1]),
                reverse=True,
            )
            beam = candidates[:MPC_BEAM_WIDTH]

        best = max(beam, key=lambda item: item[0] + self._stability_score(item[1]))
        first_action = 0 if best[3] is None else int(best[3])
        return first_action, float(best[0])

    def _portfolio_rollout_value(self, state, depth, done, depth_limit):
        if done:
            return self._terminal_value(depth)

        # 取“多控制器中的最好值”，相当于根节点短程 MPC 复核。
        # 这样能处理单个 heuristic rollout 在某些初态下会把小车带偏的问题。
        best = -1e9
        for weights in self.policy_weights:
            v = self._rollout_value_with_weights(state, depth, done, depth_limit, weights)
            if v > best:
                best = v
        return float(best)

    def _rollout_value_with_weights(self, state, depth, done, depth_limit, weights):
        state = np.asarray(state, dtype=np.float64)
        survived = 0
        stability_acc = 0.0
        while (not done and survived < depth_limit and depth < ENV_MAX_STEPS):
            action = self._linear_policy_action(state, weights)
            state, done = self._model_step(state, action, depth)
            depth += 1
            survived += 1
            stability_acc += self._stability_score(state)

        value = depth / ENV_MAX_STEPS
        if survived > 0:
            value += STABILITY_BONUS_WEIGHT * (stability_acc / survived)
        if done and depth < ENV_MAX_STEPS:
            value -= FAILURE_PENALTY * (1.0 - survived / max(1, depth_limit))
        elif depth >= ENV_MAX_STEPS:
            value += 0.20
        return float(value)

    def _policy_prior(self, state, action):
        """把控制器动作转成 PUCT 先验：推荐动作 prior 大，不推荐动作 prior 小。"""
        preferred = self._linear_policy_action(state, self.policy_weights[2])
        return 1.25 if int(action) == preferred else 0.55

    @staticmethod
    def _linear_policy_action(state, weights):
        return 1 if float(np.dot(weights, state)) > 0.0 else 0

    def _is_safe_region(self, state):
        x, x_dot, theta, theta_dot = state
        return abs(x) < FAST_PATH_SAFE_X and abs(theta) < FAST_PATH_SAFE_THETA

    @staticmethod
    def _stability_score(state):
        """越接近中心、越直立分数越高。范围大致 [0, 1]。"""
        x, x_dot, theta, theta_dot = state
        x_term = min(abs(x) / 2.4, 1.0)
        theta_term = min(abs(theta) / (12 * 2 * math.pi / 360), 1.0)
        vel_term = min(abs(x_dot) / 3.0, 1.0)
        ang_vel_term = min(abs(theta_dot) / 3.5, 1.0)
        penalty = 0.25 * x_term + 0.50 * theta_term + 0.10 * vel_term + 0.15 * ang_vel_term
        return max(0.0, 1.0 - penalty)

    @staticmethod
    def _terminal_value(depth):
        # 终止越晚越好，但早死给明显低分。
        return depth / ENV_MAX_STEPS - 0.25 * (1.0 - depth / ENV_MAX_STEPS)

    # =========================================================================
    # CartPole-v1 内部动力学模型：避免每次 rollout 都 gym.make，速度提升很大
    # =========================================================================
    @staticmethod
    def _model_step(state, action, depth) -> Tuple[np.ndarray, bool]:
        x, x_dot, theta, theta_dot = map(float, state)

        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masscart + masspole
        length = 0.5
        polemass_length = masspole * length
        force_mag = 10.0
        tau = 0.02
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4

        force = force_mag if int(action) == 1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Gymnasium CartPole 默认使用 euler 积分。
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc

        next_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        terminated = bool(
            x < -x_threshold or x > x_threshold or
            theta < -theta_threshold_radians or theta > theta_threshold_radians or
            depth + 1 >= ENV_MAX_STEPS
        )
        return next_state, terminated


class Environment:
    def __init__(self, env_id=ENV_ID, iteration_budget=ITERATION_BUDGET,
                 C_p=C_P_INIT, epochs=EPOCHS, lookahead_target=LOOKAHEAD_TARGET):
        self.env_id = env_id
        self.env = gym.make(env_id, max_episode_steps=ENV_MAX_STEPS)
        self.agent = Agent(iteration_budget=iteration_budget, env_id=env_id)
        self.epochs = int(epochs)
        self.C_p = float(C_p)
        self.lookahead_target = int(lookahead_target)
        self.n_action = self.env.action_space.n

    def train(self):
        record = []
        for i in range(self.epochs):
            obs, _ = self.env.reset()
            sum_reward = 0.0
            node = None
            C_p = self.C_p

            while True:
                # 以 unwrapped.state 为准，保持和原始 train_mcts.py 一致。
                state = np.asarray(self.env.unwrapped.state, dtype=np.float64)
                action, node, C_p = self.agent.act(
                    state,
                    n_actions=self.n_action,
                    node=node,
                    C_p=C_p,
                    lookahead_target=self.lookahead_target,
                )
                _, reward, terminated, truncated, _ = self.env.step(int(action))
                sum_reward += reward
                if terminated or truncated:
                    break

            record.append(sum_reward)
            print(f"Epoch {i}: Score = {sum_reward}")

        self.env.close()
        return float(np.mean(record))


if __name__ == "__main__":
    exp_env = Environment()
    final = exp_env.train()
    print(f"Average score: {final}")
