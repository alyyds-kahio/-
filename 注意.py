"""

依赖安装方法1（占用空间过大）：
Win + R

ms-settings:network-proxy

梯子换虚拟网卡
设置关代理

终端运行
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt   或者  python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

依赖安装方法2：
自行安装，跟着豆包

#
首次训练和续训
使用：
第一次：
RESUME = False
重训练。
以后中断：
改：
RESUME = True
继续。
生成：
checkpoints
│
├── last_model.pth
└── best_score_model.pth
"""
