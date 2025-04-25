import logging
from src.data.online_llm import online_llm_streaming

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_api():
    # 创建LLM实例，使用非streaming模式
    llm = online_llm_streaming("你好，请介绍一下你自己。", response_mode="blocking")
    
    try:
        # 调用API
        response = llm.run()
        print("API响应:", response)
    except Exception as e:
        print("API调用失败:", str(e))
        print("错误详情:", str(e.__class__.__name__))

if __name__ == "__main__":
    test_api() 