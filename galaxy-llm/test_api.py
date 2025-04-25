import logging
from src.data.online_llm import online_llm_streaming

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_api():
    # 测试阻塞模式
    print("测试阻塞模式:")
    llm = online_llm_streaming("你好，请介绍一下你自己。", response_mode="blocking")
    try:
        response = llm.run()
        if isinstance(response, str):
        print("API响应:", response)
        else:
            print("API响应类型错误:", type(response))
    except Exception as e:
        print("API调用失败:", str(e))
        print("错误详情:", str(e.__class__.__name__))
    
    print("\n" + "="*50 + "\n")
    
    # 测试流式模式
    print("测试流式模式:")
    llm = online_llm_streaming("你好，请介绍一下你自己。", response_mode="streaming")
    try:
        for chunk in llm.run():
            print(chunk, end="", flush=True)
    except Exception as e:
        print("API调用失败:", str(e))
        print("错误详情:", str(e.__class__.__name__))

if __name__ == "__main__":
    test_api() 