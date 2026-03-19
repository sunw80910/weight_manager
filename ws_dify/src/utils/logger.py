import json
import traceback
import re
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import httpx
import asyncio

from enum import Enum
from utils.logger import get_logger
from typing import Union
from pydantic import BaseModel
# 加载.env文件
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

logger = get_logger(__name__)

# 配置心跳参数 - 修复问题2：大小写统一为小写ping，和客户端一致
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", 20))
HEARTBEAT_PING = "ping" # 心跳请求标识 小写 ✅ 匹配客户端{"type": "ping"}
HEARTBEAT_PONG = "PONG" # 心跳响应标识（保持连接的核心标识）
# CONNECT_SUCCESS = "CONNECT_SUCCESS"  # 连接成功的标识
# 句子切分的标点符号（中文常用句末标点）
SENTENCE_END_MARKS = r'[。！？；～]'
# 只处理event为message/message_end/tts_message的事件
TARGET_EVENTS = ["message","message_end","tts_message"]
# 返回的文本最短的长度
MSG_MIN_LENGTH = int(os.getenv("MSG_MIN_LENGTH", 10))
DIFY_API_BASE_URL = os.getenv("DIFY_API_BASE_URL","https://myaitest.miyingbl.com/v1/")

# ws返回状态的type
class BaseType(Enum):
  HEARTBEAT = 1 # 心跳
  MESSAGE_STR = 2 # 文本
  ERROR = 3 # 异常状态

# 返回数据
class LLMDataWSResult(BaseModel):
    tag:str = ""
    type: int # 对应BaseType的值
    data:Union[str,dict,list]

# ✅ 核心新增：带时间戳的通用日志打印函数
def unicode_escape_to_chinese(escape_str):
    try:
        # 处理不同格式的转义字符串（单反斜杠/双反斜杠）
        if isinstance(escape_str, str):
            # 确保最终编码是 utf-8，避免乱码
            return escape_str.encode('raw_unicode_escape').decode('unicode_escape')
        else:
            return "输入内容不是字符串格式！"
    except Exception as e:
        return f"转换失败：{str(e)}"

# ✅ 安全发送消息的通用函数（封装状态判断，避免重复写）
async def safe_send_text(websocket, msg):
    """安全发送文本消息，仅当连接存活时发送"""
    if websocket.client_state.CONNECTED:
        await websocket.send_text(msg)

def split_text_to_sentences(text):
    """
    将文本按中文句末标点切分成句子列表
    :param text: 完整文本
    :return: 句子列表（过滤空字符串）
    """
    # 按句末标点切分，保留标点在句子末尾
    sentences = re.split(f'({SENTENCE_END_MARKS})', text)
    # 合并分割后的标点和句子（解决"你好。世界"切分成["你好","。","世界"]的问题）
    merged_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if sentences[i]:  # 跳过空内容
            merged_sentences.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ""))
    # 处理最后一段无标点的内容
    if len(sentences) % 2 != 0 and sentences[-1]:
        merged_sentences.append(sentences[-1])
    # 合并小于20个字的文本字符串
    return merge_short_strings_simple(merged_sentences, MSG_MIN_LENGTH)

async def get_dify_data(websocket, result_data, resm, headers, row_data):
    """ 获取dify数据. """
    # Dify基础地址
    base_url = DIFY_API_BASE_URL
    try:
        # 异步调用Dify接口 - 修复问题3：必须加 stream=True 开启流式 ✅ 核心！
        full_answer = ""  # 缓存所有answer内容
        # base_response_json = {}  # 缓存原始JSON结构（取第一条有效数据的结构）
        sentence_cache = [] # 记录已发送的数据
        sentences=[]
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                    method="POST",
                    url=base_url + resm,
                    headers=headers,
                    json=row_data
            ) as response:
                # 异步迭代Dify的流式返回数据
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    if "[DONE]" in line:
                        break
                    # 1. 解析原始流式数据
                    data_str = line[5:].strip()  # 去掉data:前缀
                    try:
                        data_json = json.loads(data_str)
                        # ========== 核心修改：只处理event为message的数据 ==========
                        event_str = data_json.get("event")
                        if event_str not in TARGET_EVENTS:
                            logger.info(f"过滤非message事件：{event_str}")
                            # if event_str=="tts_message":
                            #     print(data_json)
                            continue
                        # logger.info(f"过滤非message事件：{time.time()}.{event_str}")
                        # 1. 提取answer内容并累加
                        answer = data_json.get("answer", "")
                        if answer:
                            full_answer += answer
                        
                        _result_data = result_data.model_copy()
                        _result_data.type = BaseType.MESSAGE_STR.value
                        _result_data.data = {}
                        # 语音数据处理，暂不返回
                        if event_str=="tts_message":
                        #     send_json = data_json.copy()
                        #     base_res1ult_json["type"] = BaseType.MESSAGE_STR
                        #     base_resu1lt_json["data"] = json.dumps(send_json, ensure_ascii=False)
                        #     await safe_send_text(websocket, json.dumps(base_resul1t_json, ensure_ascii=False))
                            continue
                        # # bak start 以防后续不再要句子，要返回原始数据
                        # # 返回原始文本数据
                        # send_json = data_json.copy()
                        # # 按原始格式返回（data: + JSON字符串）
                        # send_data = f"data: {json.dumps(send_json, ensure_ascii=False)}"
                        # await safe_send_text(websocket, json.dumps(base_res1ult_json, ensure_ascii=False))
                        # continue
                        # # bak end 以防后续不再要句子，要返回原始数据

                        # 2. 切分完整answer为句子
                        sentences = split_text_to_sentences(full_answer)
                        if not sentences:
                            logger.error("无法将回答切分为有效句子")
                            continue

                        # 3. 非结束时过滤掉最后一条
                        if event_str!="message_end":
                            sentences = sentences[:-1]
                        # 4. 循环句子进行回复
                        for index,sentence in enumerate(sentences):
                            # 判断当前句子是否在缓存中
                            is_exist = next((True for se in sentence_cache if se["index"]==index and se["sentence"]==sentence ), False)
                            
                            if sentence.strip() and not is_exist:
                                # 句子添加进缓存
                                sentence_cache.append({"index":index,"sentence":sentence})
                                # 替换answer为当前句子
                                send_json = data_json.copy()
                                send_json["answer"] = sentence.strip()
                                # 按原始格式返回（data: + JSON字符串）
                                
                                _result_data.data = send_json
                                await safe_send_text(websocket, _result_data.model_dump_json(ensure_ascii=False))
                                
                                # 句子间短暂延迟，模拟自然回复
                                await asyncio.sleep(0.001)
                    except json.JSONDecodeError:
                        continue
        logger.info(f"📝 共解析到 {len(sentences)} 句回答，按原始结构逐句推送,原文--->\n{full_answer}")
        # 增加多层边界判断
        if not full_answer.strip():
            _log_data = "Dify返回的answer内容为空"
            _result_data = result_data.model_copy()
            _result_data.data = _log_data
            await safe_send_text(websocket, _result_data.model_dump_json(ensure_ascii=False))
            logger.warn(_log_data)
        # 推送完成标识
        # await safe_send_text(websocket, f"[STATUS] AI回答按句推送完成 ✔️")
    except Exception as req_err:
        # 捕获请求Dify的异常，友好提示
        traceback.print_exc()
        _log_data = f"请求Dify接口失败：{str(req_err)}"
        logger.error(_log_data)
        _result_data = result_data.model_copy()
        _result_data.data = _log_data
        await safe_send_text(websocket, _result_data.model_dump_json(ensure_ascii=False))

def merge_short_strings_simple(arr, min_length=20):
    """整理string数组，合并小于N的字符串"""
    result = []
    i = 0
    while i < len(arr):
        result.append(arr[i] if len(arr[i]) >= min_length or i == len(arr)-1 else arr[i] + arr[i+1])
        i += 1 if len(arr[i]) >= min_length or i == len(arr)-1 else 2
    return result

@app.websocket("/ws/dify")
async def websocket_endpoint(websocket: WebSocket):
    # ✅ 新增：连接关闭开关（核心，标记后心跳任务立即停止）
    is_closed = asyncio.Event()
    heartbeat_task = None
    try:
        # 1. WebSocket握手建立连接，必须第一行执行，正确无误
        await websocket.accept()
        # await safe_send_text(websocket, f"[STATUS] {CONNECT_SUCCESS}")
        logger.info("✅ 客户端WebSocket连接成功，已发送连接成功标识")

        # 定义心跳任务：定时给客户端发心跳响应，维持连接
        async def heartbeat():
            while True:
                # ✅ 优先判断关闭开关，一旦标记立即终止（比CONNECTED更优先）
                if is_closed.is_set():
                    logger.info("心跳任务：检测到关闭开关，终止循环")
                    break
                # ✅ 核心：连接断开则立即终止心跳任务
                if websocket.client_state != websocket.client_state.CONNECTED:
                    logger.info(f"心跳任务检测到连接已关闭，终止心跳,{websocket.client_state}")
                    break
                try:
                    await safe_send_text(websocket, LLMDataWSResult(type=BaseType.HEARTBEAT.value, data=HEARTBEAT_PONG).model_dump_json(ensure_ascii=False))
                    await asyncio.sleep(HEARTBEAT_INTERVAL)
                except Exception as e:
                    traceback.print_exc()
                    logger.warn(f"心跳任务发送失败：{str(e)}")
                    break

        # 启动心跳后台任务，不阻塞正常消息收发
        heartbeat_task = asyncio.create_task(heartbeat())
        logger.info("心跳任务启动")

        while True:
            if websocket.client_state != websocket.client_state.CONNECTED:
                logger.info(f"检测到连接已关闭，终止，{websocket.client_state}")
                break

            try:
                # 接收客户端的JSON格式消息（心跳/业务请求）
                json_data = await asyncio.wait_for(websocket.receive_json(), timeout=35)
            except asyncio.TimeoutError:
                # 超时无消息，继续循环，心跳正常推送
                logger.error("超时无消息，继续循环，心跳正常推送")
                continue
            # 修复问题8：新增捕获【JSON格式错误】异常，友好提示，不会断开连接
            except Exception as e:
                traceback.print_exc()
                logger.error("收到非标准的JSON格式数据！")
                await safe_send_text(websocket, LLMDataWSResult(type=BaseType.ERROR.value,data="收到非标准的JSON格式数据！").model_dump_json(ensure_ascii=False))
                continue

            # ========== 心跳逻辑 ==========
            if json_data.get("type") == HEARTBEAT_PING:
                await safe_send_text(websocket, LLMDataWSResult(type=BaseType.HEARTBEAT.value, data=HEARTBEAT_PONG).model_dump_json(ensure_ascii=False))
                logger.info("📌 收到客户端心跳包，已回复PONG心跳标识")

            # ========== 正常业务请求逻辑 ==========
            else:
                # 统一获取参数+赋值正确的默认值+变量命名规范
                resm = json_data.get("resm", "")  # Dify接口后缀 如：chat-messages
                headers = json_data.get("headers", {})  # 请求头 默认空字典 ✅ 修复问题5
                row_data = json_data.get("data", {})  # 请求体参数 默认空字典
                req_tag = json_data.get("tag", "")  # tag,fanhuishi

                result_data = LLMDataWSResult(type=BaseType.ERROR.value, data="",tag=req_tag)
                
                # 基础参数校验，防止无效请求
                if not resm or not row_data:
                    _log_data = f"客户端参数错误：resm={resm}，data={row_data}"
                    logger.warn(_log_data)
                    _result_data = result_data.model_copy()
                    _result_data.data = _log_data
                    await safe_send_text(websocket, _result_data.model_dump_json(ensure_ascii=False))
                    continue

                # 发送请求中状态
                logger.info(f"正在请求AI回答，请稍候...")

                # 解决同一客户端同时发送多次会变成顺序执行，导致阻塞问题，当需要前端根据tag来进行筛选数据
                asyncio.create_task(get_dify_data(websocket, result_data, resm, headers, row_data))
                # 原阻塞版本：
                # await get_dify_data(websocket, result_data, resm, headers, row_data)
                
    # 捕获客户端主动断开连接
    except WebSocketDisconnect:
        logger.error("❌ 客户端主动断开WebSocket连接")
        # ✅ 立即标记关闭开关，阻断所有发送操作
        is_closed.set()
    # 捕获其他所有异常
    except Exception as e:
        _log_data = f"[ERROR] 服务端异常：{str(e)}"
        logger.error(_log_data)
        is_closed.set()  # ✅ 标记关闭
        await safe_send_text(websocket, LLMDataWSResult(type=BaseType.ERROR.value,data=_log_data).model_dump_json(ensure_ascii=False))
    # 最终收尾：关闭连接+取消心跳任务
    finally:
        # 标记关闭开关（双重保险）
        is_closed.set()
        # 终止心跳任务（避免异步残留）
        if heartbeat_task and not heartbeat_task.done():
            try:
                heartbeat_task.cancel()
                await heartbeat_task  # 等待任务终止
                logger.info("心跳任务已终止")
            except asyncio.CancelledError:
                logger.info("心跳任务正常取消")
            except Exception as e:
                logger.warn(f"终止心跳任务失败：{str(e)}")
        # 仅当连接存活时关闭（避免重复关闭）
        if websocket.client_state.CONNECTED:
            try:
                await websocket.close()
                logger.info("🔚 连接已正常关闭")
            except Exception as e:
                # ✅ 过滤掉「关闭后发送」的无效报错，仅记录其他异常
                if "Cannot call 'send' once a close message has been sent" not in str(e):
                    logger.warn(f"关闭连接失败：{str(e)}")
                else:
                    logger.info("连接已关闭，忽略发送报错")
        else:
            logger.info("🔚 连接已关闭，无需重复操作")


if __name__ == "__main__":
    import uvicorn
    logger = get_logger(__name__)    

    logger.info("🚀 服务启动中：0.0.0.0:8000")
    uvicorn.run(
        app="dify_websocket:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=None  # 关闭 uvicorn 的日志，
    )
