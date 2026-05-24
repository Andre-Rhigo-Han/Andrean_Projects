"""
通用 CartPole 可视化入口 —— 单 seed 跑一个 episode,渲染时在右上角叠加
状态参数和模型决策,默认存成 gif。

MCTS 不走这里(见 vis_mcts.py)。

学生的 Agent 类需要提供:
    __init__(**init_kwargs)
    load_model(path)
    predict(state) -> int
    policy_info(state) -> dict   # 可选。提供则在信息框里额外展示

典型用法:
    python vis.py \\
        --agent-class train_qlearning_backup:Agent \\
        --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}' \\
        --checkpoint checkpoints/q_learning_model.pkl \\
        --seed 42 \\
        --output video/q_learning_seed42.gif
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from typing import List, Optional

import cv2
import gymnasium as gym
import numpy as np
from PIL import Image


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _load_agent_class(agent_class_spec: str):
    if ":" not in agent_class_spec:
        raise ValueError(
            f"--agent-class 应为 'module:Class' 格式,收到: {agent_class_spec!r}"
        )
    module_name, class_name = agent_class_spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_agent(agent_class_spec: str, init_kwargs: dict, checkpoint: str):
    AgentCls = _load_agent_class(agent_class_spec)
    agent = AgentCls(**init_kwargs)
    agent.load_model(checkpoint)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent


def _overlay_info_topright(frame: np.ndarray, lines: List[str], margin: int = 10) -> np.ndarray:
    """在帧右上角画半透明黑底 + 白字信息块。frame 为 RGB uint8。"""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_h = 22

    max_tw = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines
    )
    box_w = max_tw + 2 * margin
    box_h = len(lines) * line_h + margin

    x0 = w - box_w - margin
    y0 = margin

    # 半透明黑底
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    alpha = 0.55
    frame_out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # 白字
    for i, line in enumerate(lines):
        y = y0 + (i + 1) * line_h - 6
        cv2.putText(
            frame_out, line, (x0 + margin, y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return frame_out


def _save_gif(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    duration = int(1000 / fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    print(f"GIF 已保存到: {path}")


def visualize_one_episode(
    agent,
    seed: int,
    max_steps: int,
    output: Optional[str],
    perturb_scale: float = 0.0,
    fps: int = 30,
):
    _seed_everything(seed)
    env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=max_steps)
    state, _ = env.reset(seed=seed)
    state = np.array(state, dtype=np.float32)

    if perturb_scale > 0:
        rng = np.random.default_rng(seed)
        state = state + rng.normal(0.0, perturb_scale, size=state.shape).astype(np.float32)
        try:
            env.unwrapped.state = state
        except Exception:
            pass

    frames: List[np.ndarray] = []
    step = 0
    total_reward = 0.0

    while True:
        raw_frame = env.render()
        cart_pos, cart_vel, pole_angle, pole_vel = state

        action = int(agent.predict(state))
        action_name = "LEFT <-- " if action == 0 else " --> RIGHT"

        lines = [
            f"Seed: {seed}",
            f"Step: {step}",
            f"Action:    {action_name}",
            f"CartPos:   {cart_pos:+.3f}",
            f"CartVel:   {cart_vel:+.3f}",
            f"PoleAngle: {pole_angle:+.3f}",
            f"PoleVel:   {pole_vel:+.3f}",
        ]
        if hasattr(agent, "policy_info"):
            try:
                pi = agent.policy_info(state)
                if isinstance(pi, dict) and pi:
                    lines.append("-- policy --")
                    for k, v in pi.items():
                        lines.append(f"{k}: {v}")
            except Exception:
                pass

        frames.append(_overlay_info_topright(raw_frame, lines))

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        step += 1
        state = np.array(next_state, dtype=np.float32)

        if terminated or truncated:
            final = env.render()
            if final is not None:
                lines[2] = "Action:    TERMINATED"
                frames.append(_overlay_info_topright(final, lines))
            break

    env.close()
    print(f"Episode 结束: steps={step}  reward={total_reward:.2f}")

    if output:
        _save_gif(frames, output, fps=fps)

    return step, total_reward


def main():
    parser = argparse.ArgumentParser(
        description="CartPole 通用可视化入口 —— 单 seed gif",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent-class", required=True,
                        help="Agent 类定位符,格式 'module:ClassName'")
    parser.add_argument("--agent-init-kwargs", default="{}",
                        help="Agent 构造函数 kwargs(JSON 字符串)")
    parser.add_argument("--checkpoint", required=True,
                        help="模型 checkpoint 文件路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="可视化使用的随机种子")
    parser.add_argument("--max-episode-steps", type=int, default=2000,
                        help="单个 episode 的 TimeLimit(覆盖 CartPole-v1 默认 500)")
    parser.add_argument("--perturb-scale", type=float, default=0.0,
                        help="初始状态高斯扰动尺度(0 关闭)")
    parser.add_argument("--output", type=str, default=None,
                        help="GIF 输出路径;不传则不保存,只打印统计")
    parser.add_argument("--fps", type=int, default=30,
                        help="GIF 帧率")
    args = parser.parse_args()

    try:
        init_kwargs = json.loads(args.agent_init_kwargs)
    except json.JSONDecodeError as e:
        print(f"--agent-init-kwargs 不是合法 JSON: {e}", file=sys.stderr)
        sys.exit(2)
    if not isinstance(init_kwargs, dict):
        print("--agent-init-kwargs 必须解析为 JSON 对象(dict)", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.checkpoint):
        print(f"checkpoint 不存在: {args.checkpoint}", file=sys.stderr)
        sys.exit(2)

    agent = _build_agent(args.agent_class, init_kwargs, args.checkpoint)
    visualize_one_episode(
        agent=agent,
        seed=args.seed,
        max_steps=args.max_episode_steps,
        output=args.output,
        perturb_scale=args.perturb_scale,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
