"""exo_balance 无绳站立 PPO 配置。"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def exo_balance_stand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """创建 PPO runner 配置，网络结构先参考宇树 G1。"""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.45,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.003,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=7.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="exo_balance_stand",
        save_interval=500,
        num_steps_per_env=32,
        max_iterations=6000,
    )


def exo_balance_stand_push_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """创建抗扰动站立 PPO 配置，默认按约一小时训练量设置。"""
    cfg = exo_balance_stand_ppo_runner_cfg()
    cfg.experiment_name = "exo_balance_stand_push"
    cfg.max_iterations = 6000
    cfg.save_interval = 500
    return cfg
