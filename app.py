import json
from unicodedata import name
from flask import Flask, request, jsonify
# 解决跨域问题
from flask_cors import CORS
import subprocess
import mysql.connector
from oneie.predict_text import predict
# 创建Flask实例
app = Flask(__name__)
# 解决跨域
CORS(app, resources={r'/*': {'origins': '*'}}, supports_credentials=True)

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

# 搜索接口
@app.route('/api/search', methods=['POST', 'OPTIONS'])
def search():
    if request.method == 'OPTIONS':
        # 处理预检请求
        response_headers = {
            'Access-Control-Allow-Origin': '*',  # 允许所有来源的请求
            'Access-Control-Allow-Methods': 'POST',  # 允许的方法
            'Access-Control-Allow-Headers': 'Content-Type',  # 允许的头部信息
        }
        return '', 200, response_headers

    query = request.json.get('query')
    script_path = 'test.py'  # Python 脚本的路径
    python_path = '/root/anaconda3/envs/opinion/bin/python'  # Python 解释器的路径

    cmd = [python_path, script_path, query]  # 构建执行脚本的命令

    try:
        result = subprocess.check_output(cmd)  # 执行命令并获取输出结果
        result = result.decode('utf-8')  # 将字节流转换为字符串
        return jsonify(result=result)
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e))
    

    # 返回搜索结果作为 JSON 响应
    # return jsonify(query)

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

@app.route('/api/sendEvent',methods=['POST'])
def textEEPredict():
  text=request.json['text']
  language=request.json["language"]
  res=predict(language,text)
  return jsonify({'code': 200, 'result':res})
  
if __name__ == '__main__':
  # 启动服务 默认开在5000端口
  app.run(debug=True)