# PublicOpinion_Flask
舆情项目flask后端接口

## 安装依赖
```
pip install flask // Version: 3.0.0
pip install flask_cors
pip install pymysql
pip install mysql-connector-python
pip install open_clip_torch
```
目前torch版本为1.13.0，一般torch会向低版本兼容，需要更换可以查看 https://pytorch.org/get-started/previous-versions/
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

```
torchvision==0.14.0
torchaudio==0.13.0
```


## 启动服务
```
from flask import Flask
app = Flask(__name__)
app.run()
```

## 所有的接口我们都以/api开头

## 所有模型文件都用绝对路径，都需要传在服务器上！

git上无法上传太大的文件，建议超过10M的都放在本地，使用绝对路径进行访问。

例如`/media/dell/xiehou/project/opinion/PublicOpinion_Flask/oneie/logs/cn/best.role.mdl`

最终的python编译器会选择root用户下的统一编译器，所以可以不用考虑文件权限问题。

但是，如果你的python第三方库需要指定按照某些版本，请在下方进行填写：

```
transformers==4.12.1 //示例

```
**再添加自己模块的时候，尽量创建一个属于自己的目录，便于后期自己的管理**