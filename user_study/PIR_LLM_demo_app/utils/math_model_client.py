import requests
import json
import re
import time

class ModelClient:
    def __init__(self, model_path, base_url="http://localhost:8716", stop_tokens=["<｜end▁of▁sentence｜>"],reasoning_model = True, api_key="none", temperature=1, top_p=1,  timeout=120):
        """
        初始化 ModelClient

        :param api_key: 调用 API 的密钥
        :param model_path: 模型路径
        :param base_url: API 地址
        :param timeout: 请求超时时间
        """
        self.api_key = api_key
        self.model_path = model_path
        self.base_url = base_url
        self.chat_completions_url = base_url + "/v1/chat/completions"
        self.tokenizer_url = base_url + "/tokenize"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.stop_reason = None
        if reasoning_model == True:
            self.completion = "<think>\n"
        else:
            self.completion = ""
        self.max_model_len = 4096
        self.stop_tokens = stop_tokens
        self._first_request = True
        self.api_key = "sk-EAR9W8gMRCM2JIMJ2e5b8b095c494718B5CeC1C17004577d"

    def _token_count(self, messages):
        total_token_count = 0
        payload = {
            "messages": messages
        }
        results = self._post_with_retry(
            url=self.tokenizer_url,
            headers={"Content-Type": "application/json"},
            payload=payload)
        if isinstance(results, dict):
            total_token_count = results.get("count", 0)
        return total_token_count


    def _handle_error(self, error, response=None):
        """
        专门负责错误处理
        """
        if response is not None:
            return f"HTTP Error {response.status_code}: {response.text}"
        elif isinstance(error, requests.exceptions.RequestException):
            return f"Request failed: {str(error)}"
        elif isinstance(error, Exception):
            return f"Unexpected error: {str(error)}"
        else:
            return f"Error: {str(error)}"

    def _post_with_retry(self, url, headers, payload, max_retries=3, retry_wait=10):
        """
        负责带重试的 POST 请求

        :param url: 请求地址
        :param headers: 请求头
        :param payload: 请求数据
        :param max_retries: 最大重试次数
        :param retry_wait: 每次重试间隔秒数
        :return: 成功时返回 response.json()，失败时返回错误信息字符串
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    url=url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    last_error = self._handle_error(None, response=response)
                    print(f"[Attempt {attempt}] 调用模型失败: {last_error}")
                    
            except Exception as e:
                last_error = self._handle_error(e)
                print(f"[Attempt {attempt}] 调用模型失败: {last_error}")
                

            if attempt < max_retries:
                print(f"[Retry {attempt}/{max_retries}] 请求失败，{retry_wait} 秒后重试...")
                time.sleep(retry_wait)

        return last_error

    def chat(self, messages):
        """
        发起一次聊天请求（调用带重试的请求）
        """
        if self.max_model_len <= 0:
            raise ValueError("Out of tokens for this session.")
        
        token_count = self._token_count(messages)
        if token_count:
            self.max_model_len = self.max_model_len - token_count
            # self._first_request = False
        
        print("当前剩余可用 completion_tokens:", self.max_model_len)
        if self.max_model_len <= 0:
            raise ValueError("Out of tokens for this session.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_tokens,
            "max_tokens": self.max_model_len,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "include_stop_str_in_output": True,
            "echo": False
        }

        results = self._post_with_retry(
            url=self.chat_completions_url,
            headers=headers,
            payload=payload
        )
        # print("模型返回结果:", results)
        if isinstance(results, dict):  # 成功
            self.stop_reason = results['choices'][0].get('stop_reason', None)
            self.completion += results['choices'][0]['message']['content']
            self.max_model_len -= results['usage']['total_tokens']
            # total_tokens = results['usage']['total_tokens']
            self.max_model_len = max(1, self.max_model_len)
            # print(results)
            return results['choices'][0]['message']['content']
        else:
            raise ValueError(results)  # 错误信息字符串


if __name__ == "__main__":
    model_client = ModelClient(model_path="Proactive-Interactive-R1-Math-7B-Max")
    messages = [
        {"role": "system", "content": "You are a helpful assistant for solving math problems."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    response = model_client.chat(messages)
    print("模型回答:", response)