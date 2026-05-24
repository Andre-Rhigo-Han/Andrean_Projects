import os
import time
import random
import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from train_mcts import Agent as MCTSAgent

SEED = 28
EPISODES = 1
ENVIRONMENT = "CartPole-v1"
ITERATION_BUDGET = 80
LOOKAHEAD_TARGET = 100
MAX_EPISODE_STEPS = 2000
VIDEO_BASEPATH = "./video"
START_CP = 20

if __name__ == "__main__":
    random.seed(SEED)

    parser = argparse.ArgumentParser(
        description="Run a Monte Carlo Tree Search agent on the Cartpole environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env_id", nargs="?", default=ENVIRONMENT)
    parser.add_argument("--episodes", nargs="?", default=EPISODES, type=int)
    parser.add_argument("--iteration_budget", nargs="?", default=ITERATION_BUDGET, type=int)
    parser.add_argument("--lookahead_target", nargs="?", default=LOOKAHEAD_TARGET, type=int)
    parser.add_argument("--max_episode_steps", nargs="?", default=MAX_EPISODE_STEPS, type=int)
    parser.add_argument("--video_basepath", nargs="?", default=VIDEO_BASEPATH)
    parser.add_argument("--start_cp", nargs="?", default=START_CP, type=int)
    parser.add_argument("--seed", nargs="?", default=SEED, type=int)
    args = parser.parse_args()

    os.makedirs(args.video_basepath, exist_ok=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    # 录视频：用 rgb_array + RecordVideo
    env = gym.make(args.env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=args.video_basepath,
        name_prefix=f"output_{timestr}",
        episode_trigger=lambda ep: True,   # 每个 episode 都录
        disable_logger=True,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)

    agent = MCTSAgent(args.iteration_budget, args.env_id)

    for i in range(args.episodes):
        ob, info = env.reset(seed=args.seed)

        sum_reward = 0.0
        node = None
        C_p = args.start_cp

        while True:
            # gymnasium 里如果想窗口实时显示，用 render_mode="human"
            # 但录视频通常用 rgb_array，不弹窗
            action, node, C_p = agent.act(
                env.unwrapped.state,  # CartPole 仍可用
                n_actions=env.action_space.n,
                node=node,
                C_p=C_p,
                lookahead_target=args.lookahead_target,
            )

            ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            sum_reward += reward
            print("### observed state:", ob)
            print("### sum_reward:", sum_reward)

            if done:
                break

    env.close()
    print(f"Saved videos to: {args.video_basepath}")
