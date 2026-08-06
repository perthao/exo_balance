#!/usr/bin/env python3
"""加载训练好的策略，在 MuJoCo 里观察 exo_balance。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import tyro
import mjlab

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


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
class PlayConfig:
    """回放入口参数。"""

    agent: str = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = 1
    device: str | None = None
    viewer: str = "auto"
    no_terminations: bool = False


def run_play(task_id: str, cfg: PlayConfig) -> None:
    """创建环境，加载 checkpoint，然后启动 viewer。"""
    configure_torch_backends()
    patch_warp_context_for_mjlab()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.no_terminations:
        env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if cfg.agent == "zero":
        action_shape = env.unwrapped.action_space.shape

        class ZeroPolicy:
            """输出全零动作，用来检查默认站姿是否能撑住。"""

            def __call__(self, obs):
                del obs
                return torch.zeros(action_shape, device=env.unwrapped.device)

        policy = ZeroPolicy()
    else:
        if cfg.checkpoint_file is None:
            raise ValueError("trained 模式必须提供 --checkpoint-file")
        checkpoint = Path(cfg.checkpoint_file)
        if not checkpoint.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint}")
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)

    resolved_viewer = cfg.viewer
    if resolved_viewer == "auto":
        resolved_viewer = "native"
    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"不支持 viewer: {cfg.viewer}")
    env.close()


def main() -> None:
    """命令行入口：注册任务并解析任务名。"""
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(PlayConfig, args=remaining_args, default=PlayConfig(), config=mjlab.TYRO_FLAGS)
    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
