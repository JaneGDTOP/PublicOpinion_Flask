from unicodedata import name
from flask import Flask
# 解决跨域问题
from flask_cors import CORS

# 创建Flask实例
app = Flask(__name__)
# 解决跨域
cors = CORS()
cors.init_app(app=app, resources={r"/api/*": {"origins": "*"}})

# 定义接口
@app.route('/api')
def index():
  return {
    "msg": "success",
    "data": "welcome to use flask."
  }

  
if __name__ == '__main__':
  # 启动服务 默认开在5000端口
  app.run()