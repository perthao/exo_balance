#!/usr/bin/env python3
"""训练 exo_balance 强化学习策略。"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import tyro
import mjlab

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


def patch_warp_context_for_mjlab() -> None:
    """给新版 Warp 补上 mjlab 旧版本还在读取的 CUDA driver 字段。"""
    import warp as wp

    if hasattr(wp, "context"):
        return
    try:
        driver_version = wp.get_cuda_driver_version() if wp.is_cuda_available() else None
    except Exception:
        driver_version = None
    wp.context = SimpleNamespace(runtime=SimpleNamespace(driver_version=driver_version))


@dataclass(frozen=True)
class TrainConfig:
    """训练入口参数，结构保持和宇树 train.py 接近。"""

    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
    enable_nan_guard: bool = False

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        """根据任务名加载环境配置和 PPO 配置。"""
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path, resume_path: Path | None = None) -> None:
    """创建 MuJoCo 并行环境，并交给 RSL-RL 的 PPO runner 训练。"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device = "cpu" if cuda_visible == "" else "cuda:0"

    configure_torch_backends()
    patch_warp_context_for_mjlab()
    cfg.env.seed = cfg.agent.seed

    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(f"[INFO] 已开启 NaN guard: {cfg.env.sim.nan_guard.output_dir}")

    print(f"[INFO] task={task_id} device={device} log_dir={log_dir}")
    env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg.agent), str(log_dir), device)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO] 从 checkpoint 继续训练: {resume_path}")
        runner.load(str(resume_path), map_location=device)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))
    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
    """准备日志目录、选择设备，然后启动训练。"""
    args = args or TrainConfig.from_task(task_id)
    log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
    log_dir = log_root_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resume_path: Path | None = None

    if args.agent.resume:
        # 按宇树 train.py 的习惯：load_run/load_checkpoint 可以是正则，默认取最新。
        resume_path = get_checkpoint_path(
            log_root_path,
            args.agent.load_run,
            args.agent.load_checkpoint,
        )

    selected_gpus, _num_gpus = select_gpus(args.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if selected_gpus is None else ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    run_train(task_id, args, log_dir, resume_path=resume_path)


def main() -> None:
    """命令行入口：先注册任务，再让用户选择任务名。"""
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
