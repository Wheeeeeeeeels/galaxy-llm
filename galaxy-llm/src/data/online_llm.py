import json
import logging
import traceback
from typing import Dict

import requests
from requests import RequestException
from tenacity import retry, retry_if_exception_type, wait_fixed, stop_after_attempt, before_log
import time

logger = logging.getLogger()


class AiGenerationException(Exception):
    """ AI生成异常 """
    def __init__(self, message: str = "AI生成错误"):
        self.message = message

    def __str__(self):
        return f'AiGenerationException: {self.message}'


class online_llm_streaming(object):
    """ AI生成 """

    base_url = "https://test-copilot.galaxy-immi.com/v1"

    def __init__(self, input_query, api_key: str = "app-Y3l0oHk6c80SqTrO1Ep0UMVk", route: str = "/workflows/run", response_mode: str = "streaming"):
        self.route = route
        self.response_mode = response_mode
        self.api_key = api_key
        self.inputs = {"input_query":input_query}

    @retry(retry=retry_if_exception_type((RequestException, AiGenerationException, )), reraise=True,
           wait=wait_fixed(1), stop=stop_after_attempt(2),
           before=before_log(logger, logging.INFO))
    def ai_generate(self, url: str, headers: Dict, data: Dict, timeout: int, response_mode: str) -> str:
        """
            AI生成
        :param url: 请求的url
        :param headers: headers
        :param data: body
        :param timeout: 超时时间
        :param response_mode: 响应模型，详见constants.py的RESPONSE_MODE
        :return:
        """
        if response_mode == "blocking":
            resp = requests.post(
                url=url,
                headers=headers,
                json=data,
                timeout=timeout
            )

            if resp.status_code != 200:
                error_msg = f'AI生成失败，http-status_code：{resp.status_code}\nresponse.text：\n=====\n{resp.text}\n=====\n'
                logger.error(error_msg)
                raise AiGenerationException(message=error_msg)

            res_json = resp.json()
            logger.info(f"AI生成返回：\n=====\n{json.dumps(res_json, indent=4, ensure_ascii=False)}\n=====\n")

            if res_json["status"] == "failed":
                error_msg = f'AI生成失败，res_json：\n=====\n{json.dumps(res_json, indent=4, ensure_ascii=False)}\n=====\n'
                logger.error(error_msg)
                raise AiGenerationException(message=error_msg)

            return res_json["data"]["outputs"]["output"]
        elif response_mode == "streaming":
            resp = requests.post(
                url=url,
                headers=headers,
                json=data,
                timeout=1200,
                stream=True
            )

            if resp.status_code != 200:
                error_msg = f'AI生成失败，http-status_code：{resp.status_code}\nresponse.text：\n=====\n{resp.text}\n=====\n'
                logger.error(error_msg)
                raise AiGenerationException(message=error_msg)

            result = ""
            for chunk in resp.iter_lines():  # chunk examples: b''、 b'event: ping'、 b'data: {"event": "node_started...
                if not chunk:
                    continue

                _, data = chunk.decode('utf-8').split(':', maxsplit=1)
                data = data.strip()
                # print(data)
                if data == "ping":
                    chunk_event = "ping"
                else:
                    chunk_event = json.loads(data.strip())["event"]

                if chunk_event != "workflow_finished":
                    continue

                chunk_data = json.loads(data)["data"]
                logger.info(f"AI生成返回：\n=====\n{json.dumps(chunk_data, indent=4, ensure_ascii=False)}\n=====\n")

                if chunk_data["status"] == "failed":
                    error_msg = f'AI生成失败，chunk_data：\n=====\n{chunk_data}\n=====\n'
                    logger.error(error_msg)
                    raise AiGenerationException(message=error_msg)

                result += chunk_data["outputs"]["output"]

            return result
        else:
            raise AiGenerationException(message=f"不支持的response_mode：{response_mode}")

    def run(self, timeout: int = 600) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "inputs": self.inputs,
            "response_mode": self.response_mode,
            "user": "fadsf"
        }
        response_output = None
        # 如果响应成功则返回，如果响应不成功，则继续尝试，最多尝试3次
        is_continue = True
        try_i = 1
        while is_continue:
            try:
                response_output = self.ai_generate(
                    url=f"{self.base_url}{self.route}",
                    headers=headers,
                    data=data,
                    timeout=timeout,
                    response_mode=self.response_mode
                )
                is_continue = False
            except:
                time.sleep(2)
                try_i += 1
            if try_i > 3:
                is_continue = False
        return response_output 