from src.models.stream_llm import stream_chat
from src.models.online_stream import online_stream

def test_stream():
    # 测试普通流式输出
    print("测试普通流式输出:")
    for chunk in stream_chat("你好，请介绍一下自己"):
        print(chunk, end="", flush=True)
    print("\n")

    # 测试在线流式输出
    print("测试在线流式输出:")
    for chunk in online_stream("你好，请介绍一下自己"):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    test_stream() 