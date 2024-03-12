# PublicOpinion_Flask
舆情项目flask后端接口

1. 安装flask
```
pip install flask // Version: 3.0.0
pip install flask_cors
pip install pymysql
pip install mysql-connector-python
pip install open_clip_torch
```
search()函数中的Python解释器需要换为自己电脑的路径
2. 启动服务
```
from flask import Flask
app = Flask(__name__)
app.run()
```

3.所有的接口我们都以/api开头

4.所有模型文件都用绝对路径，都需要传在服务器上！