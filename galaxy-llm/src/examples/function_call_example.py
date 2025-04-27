from model.moe_model import MoEModel
from model.function_call import Function, FunctionCaller
from model.config import ModelConfig

def main():
    # 初始化模型
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        num_experts=8,
        expert_capacity=64,
        dropout=0.1
    )
    model = MoEModel(config)
    
    # 创建函数调用器
    caller = FunctionCaller(model)
    
    # 注册一些示例函数
    def search_weather(city: str) -> dict:
        """模拟天气查询"""
        return {
            "city": city,
            "temperature": "25°C",
            "weather": "晴天",
            "humidity": "60%"
        }
    
    def calculate_math(expression: str) -> float:
        """计算数学表达式"""
        try:
            return eval(expression)
        except:
            return None
    
    # 注册函数
    caller.register_function(Function(
        name="search_weather",
        description="查询指定城市的天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            }
        },
        required=["city"],
        handler=search_weather
    ))
    
    caller.register_function(Function(
        name="calculate_math",
        description="计算数学表达式",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式"
                }
            }
        },
        required=["expression"],
        handler=calculate_math
    ))
    
    # 测试函数调用
    prompts = [
        "北京今天天气怎么样？",
        "计算 1 + 2 * 3 的结果",
        "上海和北京的天气对比"
    ]
    
    for prompt in prompts:
        print(f"\n用户: {prompt}")
        response = caller.generate_with_functions(prompt)
        print(f"助手: {response}")

if __name__ == "__main__":
    main() 