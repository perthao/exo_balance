"""注册 exo_balance 抗扰动站立任务。"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import exo_balance_stand_push_flat_env_cfg
from .rl_cfg import exo_balance_stand_push_ppo_runner_cfg

register_mjlab_task(
    task_id="ExoBalance-Stand-Push-Flat",
    env_cfg=exo_balance_stand_push_flat_env_cfg(),
    play_env_cfg=exo_balance_stand_push_flat_env_cfg(play=True),
    rl_cfg=exo_balance_stand_push_ppo_runner_cfg(),
)
