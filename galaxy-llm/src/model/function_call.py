import json
import re
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from .moe_model import MoEModel
from .config import ModelConfig

@dataclass
class Function:
    """函数定义"""
    name: str
    description: str
    parameters: Dict
    required: List[str]
    handler: Callable

class FunctionCaller:
    """函数调用器"""
    def __init__(self, model: MoEModel):
        self.model = model
        self.functions: Dict[str, Function] = {}
        
    def register_function(self, func: Function):
        """注册函数"""
        self.functions[func.name] = func
        
    def get_function_schema(self) -> List[Dict]:
        """获取函数模式"""
        return [
            {
                "name": func.name,
                "description": func.description,
                "parameters": func.parameters,
                "required": func.required
            }
            for func in self.functions.values()
        ]
    
    def parse_function_call(self, text: str) -> Optional[Dict]:
        """解析函数调用"""
        # 匹配函数调用格式
        pattern = r"<function_call>(.*?)</function_call>"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return None
            
        try:
            # 解析JSON
            call_data = json.loads(match.group(1))
            
            # 验证函数名
            if "name" not in call_data:
                return None
                
            func_name = call_data["name"]
            if func_name not in self.functions:
                return None
                
            # 验证参数
            func = self.functions[func_name]
            if "arguments" not in call_data:
                return None
                
            args = call_data["arguments"]
            
            # 检查必需参数
            for param in func.required:
                if param not in args:
                    return None
                    
            return {
                "name": func_name,
                "arguments": args
            }
            
        except json.JSONDecodeError:
            return None
    
    def execute_function(self, call_data: Dict) -> Dict:
        """执行函数调用"""
        func = self.functions[call_data["name"]]
        result = func.handler(**call_data["arguments"])
        
        return {
            "name": call_data["name"],
            "result": result
        }
    
    def generate_with_functions(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """生成带函数调用的回复"""
        # 添加函数描述到提示
        function_schema = self.get_function_schema()
        function_desc = json.dumps(function_schema, ensure_ascii=False)
        
        full_prompt = f"""你是一个AI助手，可以使用以下函数：
{function_desc}

用户: {prompt}
助手: 让我思考一下如何回答这个问题。"""
        
        # 生成回复
        response = self.model.generate(
            full_prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # 解析函数调用
        call_data = self.parse_function_call(response)
        if call_data:
            # 执行函数
            result = self.execute_function(call_data)
            
            # 添加函数结果到回复
            response += f"\n函数调用结果：{json.dumps(result, ensure_ascii=False)}"
            
        return response 