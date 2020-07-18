#### 快速开始
python online.py
#### 发布与应用
##### 发布
1. python setup.py bdist_wheel
2. cd dist
##### 应用
1. 安装
pip install  ****.whl
2. 使用
from transformer import Decode
output = Decode.decoding(arg...)
3. 卸载
pip uninstall transformer
#### 单元测试
1. cd test
2. ./build_test.sh
3. ./test
