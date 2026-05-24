"""
通用 CartPole 模型评估入口。

支持动态加载学生自己实现的 Agent 类,对一组 seed(默认 100 个)并行跑完整
episode,输出性能统计。MCTS 不走这里(它没有 checkpoint 概念),用
vis_mcts.py 单独录制。

学生的 Agent 类需要提供三个最小方法:
    __init__(**init_kwargs)
    load_model(checkpoint_path)
    predict(state) -> int

典型用法:
    python evaluate.py \\
        --agent-class train_qlearning_backup:Agent \\
        --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}' \\
        --checkpoint checkpoints/q_learning_model.pkl \\
        --seed-base 42 --seed-count 100 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np


def _seed_everything(seed: int) -> None:
    """统一播种 Python random / numpy / torch,让同一 seed 下评估可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _load_agent_class(agent_class_spec: str):
    """'module.submodule:ClassName' 字符串 → Agent 类对象。"""
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
    # 常见 greedy 设定:若 agent 暴露 epsilon 则强制 0(避免评估混入随机探索)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent


def _run_one_episode(
    agent_class_spec: str,
    init_kwargs: dict,
    checkpoint: str,
    seed: int,
    max_episode_steps: int,
    perturb_scale: float,
) -> Tuple[int, int, float]:
    """
    独立进程里跑一个 episode:构建 agent → reset(seed) → rollout → 返回 (seed, steps, reward)。
    """
    _seed_everything(seed)
    agent = _build_agent(agent_class_spec, init_kwargs, checkpoint)

    # max_episode_steps 通过 gym.make 传入,覆盖 CartPole-v1 默认的 500 TimeLimit。
    env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    state, _ = env.reset(seed=seed)
    state = np.array(state, dtype=np.float32)

    if perturb_scale > 0:
        rng = np.random.default_rng(seed)
        state = state + rng.normal(0.0, perturb_scale, size=state.shape).astype(np.float32)
        try:
            env.unwrapped.state = state
        except Exception:
            pass

    steps = 0
    total_reward = 0.0
    while True:
        action = int(agent.predict(state))
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return seed, steps, total_reward


def evaluate(
    agent_class_spec: str,
    init_kwargs: dict,
    checkpoint: str,
    seeds: List[int],
    max_episode_steps: int,
    workers: int,
    perturb_scale: float,
) -> List[Tuple[int, int, float]]:
    """返回每个 seed 的 (seed, steps, reward) 结果列表,按 seed 排序。"""
    results: List[Tuple[int, int, float]] = []
    if workers <= 1:
        for s in seeds:
            results.append(
                _run_one_episode(
                    agent_class_spec, init_kwargs, checkpoint, s, max_episode_steps, perturb_scale
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    _run_one_episode,
                    agent_class_spec, init_kwargs, checkpoint, s, max_episode_steps, perturb_scale,
                )
                for s in seeds
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    results.sort(key=lambda r: r[0])
    return results


def _print_report(results: List[Tuple[int, int, float]], max_steps: int) -> dict:
    steps = np.array([r[1] for r in results], dtype=np.float64)
    rewards = np.array([r[2] for r in results], dtype=np.float64)
    report = {
        "n_episodes": len(results),
        "max_episode_steps": max_steps,
        "steps": {
            "mean": float(np.mean(steps)),
            "median": float(np.median(steps)),
            "std": float(np.std(steps)),
            "min": float(np.min(steps)),
            "max": float(np.max(steps)),
            "p25": float(np.percentile(steps, 25)),
            "p75": float(np.percentile(steps, 75)),
            "reached_max_ratio": float(np.mean(steps >= max_steps)),
        },
        "rewards": {
            "mean": float(np.mean(rewards)),
            "median": float(np.median(rewards)),
            "std": float(np.std(rewards)),
        },
    }
    print(f"\n=== 评估结果({report['n_episodes']} episodes,TimeLimit={max_steps}) ===")
    s = report["steps"]
    print(f"步数:mean={s['mean']:.1f}  median={s['median']:.0f}  std={s['std']:.1f}  "
          f"min={s['min']:.0f}  max={s['max']:.0f}  (p25={s['p25']:.0f}/p75={s['p75']:.0f})")
    print(f"      达到 TimeLimit 上限的比例: {s['reached_max_ratio']*100:.1f}%")
    r = report["rewards"]
    print(f"回报:mean={r['mean']:.2f}  median={r['median']:.2f}  std={r['std']:.2f}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="CartPole 通用评估入口 —— 支持动态加载学生 Agent 类 + 多 seed 并行",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent-class", required=True,
                        help="Agent 类定位符,格式 'module:ClassName'")
    parser.add_argument("--agent-init-kwargs", default="{}",
                        help="Agent 构造函数 kwargs(JSON 字符串)")
    parser.add_argument("--checkpoint", required=True,
                        help="模型 checkpoint 文件路径")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="seed 起点")
    parser.add_argument("--seed-count", type=int, default=100,
                        help="seed 数量")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="并行进程数(默认 CPU 一半)")
    parser.add_argument("--max-episode-steps", type=int, default=2000,
                        help="单个 episode 的 TimeLimit(覆盖 CartPole-v1 默认 500)")
    parser.add_argument("--perturb-scale", type=float, default=0.0,
                        help="初始状态高斯扰动尺度(0 关闭)")
    parser.add_argument("--report-json", type=str, default=None,
                        help="可选:把统计和 per-seed 结果存成 JSON")
    parser.add_argument("--print-per-seed", action="store_true",
                        help="额外打印每个 seed 的步数/回报")

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

    seeds = list(range(args.seed_base, args.seed_base + args.seed_count))
    print(f"Agent:    {args.agent_class}")
    print(f"Kwargs:   {init_kwargs}")
    print(f"Ckpt:     {args.checkpoint}")
    print(f"Seeds:    [{seeds[0]}..{seeds[-1]}] ({len(seeds)} 个)")
    print(f"Workers:  {args.workers}")

    results = evaluate(
        agent_class_spec=args.agent_class,
        init_kwargs=init_kwargs,
        checkpoint=args.checkpoint,
        seeds=seeds,
        max_episode_steps=args.max_episode_steps,
        workers=args.workers,
        perturb_scale=args.perturb_scale,
    )

    if args.print_per_seed:
        print("\nper-seed:")
        for s, steps, rew in results:
            print(f"  seed={s:>5d}  steps={steps:>5d}  reward={rew:>8.2f}")

    report = _print_report(results, args.max_episode_steps)

    if args.report_json:
        payload = {
            "agent_class": args.agent_class,
            "init_kwargs": init_kwargs,
            "checkpoint": args.checkpoint,
            "seeds": seeds,
            "per_episode": [
                {"seed": s, "steps": int(steps), "reward": float(rew)}
                for s, steps, rew in results
            ],
            "summary": report,
        }
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n详细报告已写入: {args.report_json}")


if __name__ == "__main__":
    main()
