import os
import json

def load_config():
    # 获取当前脚本所在目录（即 utils/ 的父目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上一级到项目根目录
    project_root = os.path.dirname(current_dir)
    # 构建 config.json 的绝对路径
    path = os.path.join(project_root, 'configs', 'config.json')
    
    with open(path, 'r') as f:
        return json.load(f)