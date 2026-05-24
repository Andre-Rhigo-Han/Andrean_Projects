"""
train_mcts_bonus2_improved.py

MCTS Bonus 2 tuned version for CartPole-v1.

Design goal:
    Keep the required MCTS structure, but fix the main bottleneck of vanilla
    MCTS on CartPole: random rollouts and raw UCT rewards are too noisy.  This
    version uses MCTS + heuristic rollout + one-step safety planning.

How to run:
    python train_mcts_bonus2_improved.py

Suggested evaluation command:
    python evaluate.py \
        --agent-class train_mcts_bonus2_improved:Agent \
        --agent-init-kwargs '{"iteration_budget":240,"env_id":"CartPole-v1"}' \
        --seed-base 42 --seed-count 100 --max-episode-steps 2000

MCTS has no checkpoint.  If your evaluate.py requires a checkpoint argument,
pass any existing path, because load_model() is intentionally a no-op.
"""

import itertools
import math
import random

import gymnasium as gym
import numpy as np


# =============================================================================
# Bonus 2 hyperparameters
# =============================================================================
ENV_MAX_STEPS = 2000
ITERATION_BUDGET = 240
EPOCHS = 5

# The reward returned by the rollout is now roughly in the 0-350 range, so C_p
# should be much smaller than the original 200/300 scale.
C_P_INIT = 18.0
C_P_MIN = 2.0
C_P_MAX = 40.0
LOOKAHEAD_TARGET = 80

ROLLOUT_DEPTH_LIMIT = 320
PROGRESSIVE_BIAS = 18.0
STATE_SYNC_TOL = 1e-6

# Linear controller used as rollout policy and progressive prior.
# Score > 0 means push right; score <= 0 means push left.
# The pole-angle terms dominate; the cart-position terms gently pull the cart
# back toward the center so that the policy does not drift to the side boundary.
CONTROL_WEIGHTS = np.array([-0.08, -0.28, 1.15, 0.78], dtype=np.float64)

# Terminal-risk evaluator for one-step safety planning.
VALUE_WEIGHTS = np.array([3.0, 1.0, 90.0, 6.0], dtype=np.float64)


class MCTSNode:
    id_iter = itertools.count()

    def __init__(self, params, done, depth):
        self.params = np.array(params, dtype=np.float64)
        self.children = {}
        self.parent = None
        self.Q = 0.0
        self.N = 0
        self.id = next(MCTSNode.id_iter)
        self.done = bool(done)
        self.depth = int(depth)
        self.action = None


class Agent:
    def __init__(self, iteration_budget=ITERATION_BUDGET, env_id="CartPole-v1",
                 n_state=4, n_action=2, **kwargs):
        self.env_id = env_id
        self.iteration_budget = int(iteration_budget)
        self.n_actions = int(n_action) if n_action is not None else 2
        self.C_p = float(kwargs.get("C_p", C_P_INIT))
        self.lookahead_target = int(kwargs.get("lookahead_target", LOOKAHEAD_TARGET))
        self._current_node = None

    # -------------------------------------------------------------------------
    # evaluate.py-compatible methods
    # -------------------------------------------------------------------------
    def load_model(self, checkpoint_path=None):
        self._current_node = None
        self.C_p = C_P_INIT

    def policy_info(self):
        return {
            "algorithm": "MCTS with heuristic rollout and safety planning",
            "iteration_budget": self.iteration_budget,
            "C_p": self.C_p,
            "rollout_depth_limit": ROLLOUT_DEPTH_LIMIT,
        }

    def predict(self, state):
        state = np.array(state, dtype=np.float64)

        # evaluate.py may reuse the same Agent object across episodes.  If the
        # cached tree root is not the actual current state, discard the tree.
        if self._current_node is not None:
            if np.max(np.abs(self._current_node.params - state)) > STATE_SYNC_TOL:
                self._current_node = None

        action, node, self.C_p = self._uct_search(
            state=state,
            n_actions=self.n_actions,
            node=self._current_node,
            C_p=self.C_p,
            lookahead_target=self.lookahead_target,
        )
        self._current_node = node
        return int(action)

    # -------------------------------------------------------------------------
    # Public MCTS entry used by the original training wrapper
    # -------------------------------------------------------------------------
    def act(self, state, n_actions, node=None, C_p=C_P_INIT,
            lookahead_target=LOOKAHEAD_TARGET):
        self.n_actions = int(n_actions)
        return self._uct_search(state, n_actions, node=node, C_p=C_p,
                                lookahead_target=lookahead_target)

    def _uct_search(self, state, n_actions, node=None, C_p=C_P_INIT,
                    lookahead_target=LOOKAHEAD_TARGET):
        state = np.array(state, dtype=np.float64)
        if node is None or np.max(np.abs(node.params - state)) > STATE_SYNC_TOL:
            root_node = MCTSNode(state, False, 0)
        else:
            root_node = node
        root_node.parent = None

        max_depth = 0
        for _ in range(self.iteration_budget):
            leaf = self._tree_policy(root_node, n_actions, C_p)
            max_depth = max(max_depth, leaf.depth - root_node.depth)
            reward = self._default_policy(leaf)
            self._backward(leaf, reward, root_node)

        if len(root_node.children) == 0:
            safe_action = self._safety_action(root_node.params)
            child = MCTSNode(root_node.params, False, root_node.depth + 1)
            child.action = safe_action
            child.parent = root_node
            root_node.children[safe_action] = child
            return safe_action, child, C_p

        # Adaptive exploration.  Vanilla C_p=200 is too large for the shaped
        # reward scale here, so keep it in a bounded interval.
        if max_depth < lookahead_target:
            C_p = max(C_P_MIN, C_p * 0.96)
        else:
            C_p = min(C_P_MAX, C_p * 1.03)

        # Final root decision: combine MCTS mean value with a one-step safety
        # score.  This preserves the tree-search statistics but prevents the
        # controller from choosing a branch that is already locally dangerous.
        best_child_node = self._root_best_child(root_node)
        return int(best_child_node.action), best_child_node, C_p

    # -------------------------------------------------------------------------
    # Selection and expansion
    # -------------------------------------------------------------------------
    def _tree_policy(self, node, n_actions, C_p):
        while not node.done:
            if len(node.children) < n_actions:
                return self._expand(node, n_actions)
            node = self._bestchild(node, C_p)
        return node

    def _expand(self, node, n_actions):
        exp_env = gym.make(self.env_id, max_episode_steps=ENV_MAX_STEPS)
        exp_env.reset()
        exp_env.unwrapped.state = np.array(node.params, dtype=np.float64)

        # Expand the heuristic-preferred action first.  This is a progressive
        # widening/prior trick for a binary action space: the early tree already
        # contains the action that is most likely to be useful.
        preferred = self._linear_action(node.params)
        unchosen_actions = [a for a in range(n_actions) if a not in node.children]
        if preferred in unchosen_actions:
            action = preferred
        else:
            action = random.choice(unchosen_actions)

        params, _, terminated, truncated, _ = exp_env.step(action)
        done = terminated or truncated
        exp_env.close()

        child = MCTSNode(params, done, node.depth + 1)
        child.parent = node
        child.action = int(action)
        node.children[int(action)] = child
        return child

    def _bestchild(self, node, C_p):
        parent_visits = max(1, node.N)
        preferred = self._linear_action(node.params)

        best_score = -float("inf")
        best_child = None
        for child in node.children.values():
            if child.N == 0:
                return child
            mean_value = child.Q / child.N
            exploration = C_p * math.sqrt(2.0 * math.log(parent_visits + 1.0) / child.N)
            prior = PROGRESSIVE_BIAS if child.action == preferred else -PROGRESSIVE_BIAS
            prior = prior / (1.0 + child.N)
            score = mean_value + exploration + prior
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    # -------------------------------------------------------------------------
    # Rollout and value shaping
    # -------------------------------------------------------------------------
    @staticmethod
    def _linear_action(state):
        score = float(np.dot(CONTROL_WEIGHTS, np.array(state, dtype=np.float64)))
        return 1 if score > 0.0 else 0

    def _default_policy(self, node):
        rollout_env = gym.make(self.env_id, max_episode_steps=ENV_MAX_STEPS)
        rollout_env.reset()
        rollout_env.unwrapped.state = np.array(node.params, dtype=np.float64)

        done = node.done
        reward = float(node.depth)
        steps = 0

        while not done and steps < ROLLOUT_DEPTH_LIMIT:
            state = np.array(rollout_env.unwrapped.state, dtype=np.float64)
            action = self._linear_action(state)
            _, step_reward, terminated, truncated, _ = rollout_env.step(action)
            reward += step_reward
            done = terminated or truncated
            steps += 1

        final_state = np.array(rollout_env.unwrapped.state, dtype=np.float64)
        rollout_env.close()

        # Add a small continuous-state value term.  This breaks ties between two
        # rollouts that both survive to the depth limit but leave the cart/pole in
        # very different safety margins.
        reward += 0.10 * self._state_value(final_state)
        if done and steps < ROLLOUT_DEPTH_LIMIT:
            reward -= 25.0
        return reward

    @staticmethod
    def _state_value(state):
        state = np.array(state, dtype=np.float64)
        penalty = float(np.dot(VALUE_WEIGHTS, state * state))
        # Larger is better; the constant keeps the scale positive and intuitive.
        return 100.0 - penalty

    def _backward(self, node, reward, root_node):
        stop = root_node.parent
        while node is not stop:
            node.N += 1
            node.Q += reward
            node = node.parent

    # -------------------------------------------------------------------------
    # One-step safety planner for final root choice
    # -------------------------------------------------------------------------
    def _safety_action(self, state):
        best_action = 0
        best_score = -float("inf")
        for action in range(self.n_actions):
            score = self._one_step_score(state, action)
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def _one_step_score(self, state, action):
        test_env = gym.make(self.env_id, max_episode_steps=ENV_MAX_STEPS)
        test_env.reset()
        test_env.unwrapped.state = np.array(state, dtype=np.float64)
        next_state, _, terminated, truncated, _ = test_env.step(int(action))
        test_env.close()

        if terminated or truncated:
            return -1e9

        next_state = np.array(next_state, dtype=np.float64)
        score = self._state_value(next_state)

        # Reward agreement with the linear controller, but only mildly; the
        # continuous safety score remains dominant near boundaries.
        if int(action) == self._linear_action(state):
            score += 2.0
        return score

    def _root_best_child(self, root_node):
        safe_action = self._safety_action(root_node.params)
        best_child = None
        best_score = -float("inf")

        for child in root_node.children.values():
            if child.N == 0:
                mean_value = -float("inf")
            else:
                mean_value = child.Q / child.N

            one_step = self._one_step_score(root_node.params, child.action)
            score = mean_value + 0.45 * one_step

            # Strong but not absolute safety prior.  This is what prevents the
            # UCT visit-count bias from choosing an obviously unsafe root action.
            if child.action == safe_action:
                score += 35.0

            if score > best_score:
                best_score = score
                best_child = child

        # If the safe action was not expanded for some unexpected reason, create
        # a compatible child so tree reuse still works after this action.
        if best_child is None or safe_action not in root_node.children:
            if best_child is None:
                child = MCTSNode(root_node.params, False, root_node.depth + 1)
                child.action = safe_action
                child.parent = root_node
                root_node.children[safe_action] = child
                return child
        return best_child


class Environment:
    def __init__(self, env_id="CartPole-v1", iteration_budget=ITERATION_BUDGET,
                 C_p=C_P_INIT, epochs=EPOCHS, lookahead_target=LOOKAHEAD_TARGET):
        self.env_id = env_id
        self.env = gym.make(env_id, max_episode_steps=ENV_MAX_STEPS)
        self.agent = Agent(iteration_budget=iteration_budget, env_id=env_id,
                           C_p=C_p, lookahead_target=lookahead_target)
        self.epochs = int(epochs)
        self.C_p = float(C_p)
        self.lookahead_target = int(lookahead_target)
        self.n_action = self.env.action_space.n

    def train(self):
        record = []
        for i in range(self.epochs):
            self.env.reset()
            score = 0.0
            node = None
            C_p = self.C_p
            while True:
                state = np.array(self.env.unwrapped.state, dtype=np.float64)
                action, node, C_p = self.agent.act(
                    state,
                    n_actions=self.n_action,
                    node=node,
                    C_p=C_p,
                    lookahead_target=self.lookahead_target,
                )
                _, reward, terminated, truncated, _ = self.env.step(action)
                score += reward
                if terminated or truncated:
                    break
            record.append(score)
            print(f"Epoch {i}: Score = {score}")
        self.env.close()
        return float(np.mean(record))


if __name__ == "__main__":
    exp_env = Environment()
    final = exp_env.train()
    print(f"Average score: {final}")
