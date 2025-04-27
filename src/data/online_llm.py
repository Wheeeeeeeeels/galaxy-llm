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

    base_url = "http://test-copilot.galaxy-immi.com"

    def __init__(self, input_query, api_key: str = "app-X5tnXYzmlEwvNBOGj8Y0yaad", route: str = "/workflow/PYtMF2GJrI76I9Nu", response_mode: str = "streaming"):
        self.route = route
        self.response_mode = response_mode
        self.api_key = api_key
        self.inputs = {"input_query":input_query}

    @retry(retry=retry_if_exception_type((RequestException, AiGenerationException, )), reraise=True,
           wait=wait_fixed(2), stop=stop_after_attempt(3),
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
        resp = requests.post(
            url=url,
            headers=headers,
            json=data,
            timeout=60,
            stream=True
        )

        if resp.status_code != 200:
            error_msg = f'AI生成失败，http-status_code：{resp.status_code}\nresponse.text：\n=====\n{resp.text}\n=====\n'
            logger.error(error_msg)
            raise AiGenerationException(message=error_msg)

        try:
            result = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if 'choices' in data and len(data['choices']) > 0:
                        if 'delta' in data['choices'][0]:
                            content = data['choices'][0]['delta'].get('content', '')
                            if content:
                                result += content
                        elif 'message' in data['choices'][0]:
                            content = data['choices'][0]['message'].get('content', '')
                            if content:
                                result += content
            return result
        except Exception as e:
            error_msg = f'处理响应失败：{str(e)}\nresponse.text：\n=====\n{resp.text}\n=====\n'
            logger.error(error_msg)
            raise AiGenerationException(message=error_msg)

    def run(self, timeout: int = 600) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "prompt": self.inputs["input_query"],
            "model": "gpt-3.5-turbo",
            "stream": True
        }
        logger.info(f"准备调用API，URL: {self.base_url}{self.route}")
        logger.info(f"请求头: {json.dumps(headers, indent=2, ensure_ascii=False)}")
        logger.info(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        response_output = None
        # 如果响应成功则返回，如果响应不成功，则继续尝试，最多尝试2次
        is_continue = True
        try_i = 1
        last_error = None
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
            except Exception as e:
                last_error = e
                logger.error(f"API调用失败 (尝试 {try_i}/2): {str(e)}")
                logger.error(f"错误类型: {e.__class__.__name__}")
                if hasattr(e, 'response'):
                    logger.error(f"响应状态码: {e.response.status_code}")
                    logger.error(f"响应内容: {e.response.text}")
                time.sleep(0.2)
                try_i += 1
            if try_i > 2:
                is_continue = False
        
        if response_output is None and last_error is not None:
            raise last_error
            
        return response_output 