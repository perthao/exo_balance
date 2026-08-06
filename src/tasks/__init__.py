"""训练任务注册入口。"""

from mjlab.utils.lab_api.tasks.importer import import_packages

# 和宇树保持一致：导入 tasks 包时自动导入子任务，让任务注册代码执行。
_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
