import os
import sys
import json
import random
import configparser
import pandas as pd
from diet_recom.src.dish_recommendation import PregnancyDietRecommender, PregnancyInfo, weekly_diet_plan_to_json, \
    diet_parsing

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sqlalchemy import create_engine
from sport_recom.src import sport_recommendation
from flask import Flask, request, abort, Response, jsonify
from text_and_audio.src.text_and_audio_url import chat_and_generate_audio
from dify_api.api import DifyApiRouter
from third_api.api import ThirdApiRouter

app = Flask(__name__)
try:
    from flask_cors import CORS
    CORS(app, resources=r'/*')
except ImportError:
    print("当前flask不支持跨请求，若要支持跨域请求，需执行下面命令后再启动：pip install flask-cors")
# 定义一个白名单
allowed_ips = []


def ip_whitelist(f):
    def wrapper(*args, **kwargs):
        if request.remote_addr in allowed_ips:
            return f(*args, **kwargs)
        else:
            # 在网页中主动抛出错误
            abort(403)

    return wrapper


def get_pregnancy(week):
    if week <= 12:
        return '孕早期'
    elif week < 28:
        return '孕中期'
    else:
        return '孕晚期'


def safe_read_by_ids(select_query, ids):
    config = read_ini("conf/app.ini")
    host = config.get("mysql", "host")
    port = config.get("mysql", "port")
    username = config.get("mysql", "username")
    password = config.get("mysql", "password")
    database = config.get("mysql", "database")
    # print(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    if ids == 0:
        df = pd.read_sql(select_query, engine)
    else:
        df = pd.read_sql(select_query, engine, params=tuple(ids))
    return df


def read_ini(filename: str = "conf/app.ini"):
    """
    Read configuration from ini file.
    :param filename: filename of the ini file
    """
    config = configparser.ConfigParser()
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    config.read(filename, encoding="utf-8")
    return config


def sample_80_percent(my_list):
    """
    从列表中随机抽取约80%的元素

    参数:
        my_list: 要抽取元素的列表

    返回:
        包含约80%元素的随机子列表
    """
    # 处理空列表情况
    if not my_list:
        return []

    # 计算需要抽取的元素数量，确保至少抽取1个元素（当列表长度较小时）
    list_length = len(my_list)
    sample_size = round(list_length * 0.8)
    # 确保抽取数量在合理范围内（1到列表长度之间）
    sample_size = max(1, min(sample_size, list_length))

    # 随机抽取元素
    return random.sample(my_list, sample_size)


@app.route('/sport', methods=['GET', 'POST'])
# @ip_whitelist
def sport():
    if request.method == 'GET':
        return '欢迎来到主页！'
    elif request.method == 'POST':
        usermes = request.json.get('userMes')
        try:
            select_query = "select * from my_h_exercise_intensity"
            df = safe_read_by_ids(select_query, 0)
            # df = pd.read_excel('./data/运动强度.xlsx')
        except Exception as e:
            err = {"code": 1045,
                   "status": "error",
                   "error": f"{e}"}
            return Response(json.dumps(err, ensure_ascii=False, sort_keys=False), mimetype='application/json')
            #  孕周
        week = usermes.get('week')
        pregnancy = get_pregnancy(week)

        trimesters = {
            "孕早期": 1,
            "孕中期": 2,
            "孕晚期": 3
        }
        # 示例：为不同孕期生成运动计划 [1,2,3]
        trimester = trimesters.get(pregnancy)
        weekly_calorie_deficits = usermes.get('weekly_calorie_deficits')
        weight_kg = usermes.get('weight_kg')
        # 用户偏好的运动列表
        user_preferences = usermes.get('user_preferences')
        # 休息日
        rest_days = usermes.get('rest_days')
        planned_days = usermes.get('planned_days') if usermes.get('planned_days') else 7
        plan = {}
        try:
            # 创建运动计划生成器
            planner = sport_recommendation.ExercisePlanner(df)
            plan = planner.generate_weekly_plan(
                trimester=trimester,
                weekly_calorie_deficit=weekly_calorie_deficits,
                weight_kg=weight_kg,
                rest_days=rest_days,
                planned_days=planned_days,
                preferred_exercises=user_preferences
            )
            plan["pregnancy"] = pregnancy
            plan["code"] = 200
            plan["status"] = "success"
        except Exception as e:
            plan = {"code": 400,
                    "status": "error",
                    "error": f"{e}"}
        finally:
            return Response(json.dumps(plan, ensure_ascii=False, sort_keys=False), mimetype='application/json')
    else:
        abort(405)


@app.route('/diet', methods=['GET', 'POST'])
# @ip_whitelist
def diet():
    if request.method == 'GET':
        return '欢迎来到主页！'
    elif request.method == 'POST':
        usermes = request.json.get('userMes')
        try:
            select_query = "select * from my_h_dish_classify_meal"
            dishes = safe_read_by_ids(select_query, 0)
            # dishes = pd.read_excel("./get_nutrition_data/dish_classify_meal_all.xlsx")
        except Exception as e:
            err = {"code": 1045,
                   "status": "error",
                   "error": f"{e}"}
            return Response(json.dumps(err, ensure_ascii=False, sort_keys=False), mimetype='application/json')
            #  孕周
        week = usermes.get('week')
        bmi = usermes.get('bmi')
        pre_pregnancy_weight = usermes.get('pre_pregnancy_weight')
        pregnancy_weight = usermes.get('pregnancy_weight')
        activity_level = usermes.get('activity_level')
        pregnancy = get_pregnancy(week)

        trimesters = {
            "孕早期": 1,
            "孕中期": 2,
            "孕晚期": 3
        }
        # 示例：为不同孕期生成运动计划 [1,2,3]
        trimester = trimesters.get(pregnancy)
        weekly_plan = {}
        try:
            dishes = diet_parsing(dishes)
            #  增加数据随机性
            sample_80_dishes = sample_80_percent(dishes)
            # 初始化推荐器
            recommender = PregnancyDietRecommender(sample_80_dishes)
            # 输入孕妇信息（示例：BMI22.5，孕中期，孕前60kg，轻度活动）
            pregnant_info = PregnancyInfo(
                bmi=bmi,
                week=week,
                trimester=trimester,
                pre_pregnancy_weight=pre_pregnancy_weight,
                pregnancy_weight=pregnancy_weight,
                activity_level=activity_level
            )

            planned_days = usermes.get('planned_days') if usermes.get('planned_days') else 7
            # 生成7天计划（可修改days参数调整天数）
            weekly_plan = recommender.recommend_multi_day_plan(pregnant_info, days=planned_days)
            weekly_plan = weekly_diet_plan_to_json(weekly_plan)

            weekly_plan["code"] = 200
            weekly_plan["status"] = "success"
        # except ReferenceError as refe:
        #     weekly_plan = {"code": 300,
        #                    "status": "error",
        #                    "error": f"{refe}"}
        except Exception as e:
            weekly_plan = {"code": 400,
                           "status": "error",
                           "error": f"{e}"}
        finally:
            return Response(json.dumps(weekly_plan, ensure_ascii=False, sort_keys=False), mimetype='application/json')
    else:
        abort(405)


@app.route('/health')
def health():
    return Response(json.dumps({'status': 'UP'}), mimetype='application/json')


@app.route('/chat-audio', methods=['POST'])
def chat_audio_api():
    # 1. 校验请求格式
    if not request.is_json:
        return jsonify({
            "code": 400,
            "msg": "请求格式错误，必须提交JSON数据",
            "data": None
        }), 400

    # 2. 提取并校验请求参数
    auth_token = request.headers.get('Authorization')
    req_data = request.get_json()
    # auth_token = req_data.get("auth_token", "").strip()
    query_json = req_data.get("query_json", {})
    oss_sign_base_url = req_data.get("oss_sign_base_url", "").strip()
    uuid = req_data.get("uuid", "").strip()

    # 校验必填参数
    if not auth_token:
        return jsonify({
            "code": 401,
            "msg": "令牌无效",
            "data": None
        }), 401

    if not req_data or not isinstance(req_data, dict):
        return jsonify({
            "code": 400,
            "msg": "缺少必填参数 或格式错误（必须为字典）",
            "data": None
        }), 400

    # 3. 调用核心业务函数
    try:
        result = chat_and_generate_audio(
            auth_token=auth_token,
            query_json=query_json,
            oss_sign_base_url=oss_sign_base_url,
            uuid=uuid
        )
        if result is None:
            return jsonify({
                "code": 500,
                "msg": "接口调用失败（聊天/合成/上传环节出错）",
                "data": None
            }), 500
        else:
            return jsonify({
                "code": 200,
                "msg": "接口调用成功",
                "data": result
            }), 200

    except Exception as e:
        return jsonify({
            "code": 500,
            "msg": "接口调用失败（聊天/合成/上传环节出错）",
            "data": None
        }), 500

# 注册蓝图
app.register_blueprint(DifyApiRouter,url_prefix='/api')
app.register_blueprint(ThirdApiRouter,url_prefix='/thirdapi')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
