import json
from unicodedata import name
from flask import Flask, jsonify
# 解决跨域问题
from flask_cors import CORS
import mysql.connector
# 创建Flask实例
app = Flask(__name__)
# 解决跨域
cors = CORS()
cors.init_app(app=app, resources={r"/api/*": {"origins": "*"}})

# 连接数据库
# 创建MySQL连接
conn = mysql.connector.connect(
    host='172.20.137.141',
    user='root',
    password='123456',
    database='PublicOpinion'
)
cursor = conn.cursor()
# 定义接口
@app.route('/api')
def index():
  return {
    "msg": "success",
    "data": "welcome to use flask."
  }

# 监听数据集获取端口
@app.route('/api/getDataList', methods=['GET'])
def get_data_list():
  try:
    # 执行SQL查询
    with conn.cursor() as cursor:
        sql = 'SELECT * FROM my_data'
        cursor.execute(sql)
        results = cursor.fetchall()
        # print(json.dumps(results))
        # 将查询结果转换为列表
        payload = []
        content = {}
        for result in results:
            content = {'id': result[0], 'num': result[1], 'content': result[2]}
            payload.append(content)
            content = {}
        return jsonify({'code': 200, 'message': payload})

  except Exception as e:
    return jsonify({'code': 10001, 'message': str(e)})

if __name__ == '__main__':
  # 启动服务 默认开在5000端口

  app.run()