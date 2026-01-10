本项目聚焦于使用Kickstarter平台的项目数据建立初创项目的成功率预测模型并输出决策建议，同时模型将由独特的路由机制进行切换，保证资源的合理充分利用。
在docs里查看仓库基本使用流程
由于github不允许过大文件上传，我们需要从https://webrobots.io/kickstarter-datasets/ 下载原始数据集放到项目目录（目前需选择2025-12-18的csv，或自助修改data_process.py中的dataset_folder_name字段为原始集文件夹），然后用scripts/data_process.py对其进行清洗

添加pre-commit规则，配置文件在config/pre-commit-config.yaml，首次使用需（pip）安装pre-commit工具
