import os
import logging
import torch
from transformers import PreTrainedTokenizerFast
from model.moe_model import MoEModel
from model.config import COT_MODEL_CONFIG, DIRECT_MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 获取项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelInference:
    """模型推理"""
    def __init__(
        self,
        model_type: str = "cot",  # "cot" 或 "direct"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model_type = model_type
        
        # 加载模型和分词器
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        
    def _load_model(self) -> MoEModel:
        """加载模型"""
        if self.model_type == "cot":
            # 思维链模型（MoE + 强化学习，类似DeepSeek-R1）
            model = MoEModel(COT_MODEL_CONFIG).to(self.device)
            model_path = os.path.join(ROOT_DIR, "checkpoints", "best_model.pt")
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            # 直接回答模型（MoE + 传统训练，类似DeepSeek-V3）
            model = MoEModel(DIRECT_MODEL_CONFIG).to(self.device)
            model_path = os.path.join(ROOT_DIR, "checkpoints", "best_model.pt")
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    
    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        """加载分词器"""
        tokenizer_path = os.path.join(ROOT_DIR, "tokenizer", "tokenizer.json")
        return PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="[BOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """生成回答
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: 核采样概率
            num_return_sequences: 返回序列数量
            
        Returns:
            生成的回答
        """
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # 将输入移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return_sequences
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    # 测试用例
    test_cases = [
        "香港的力行幼稚园的学校规模是否存在？",
        "请问香港的力行幼稚园的学校生活是什么？",
        "香港的力行幼稚园的校长是谁？",
        "香港的力行幼稚园的校监是谁？",
        "香港的力行幼稚园的教育宗旨是什么？",
        "香港的力行幼稚园的教学措施或班级结构是什么？",
        "香港的力行幼稚园的入学条件是什么？",
        "香港的力行幼稚园的组织属性英文类别是什么？",
        "香港的力行幼稚园的电子邮件是什么？",
        "香港的力行幼稚园的学校名称是什么？"
    ]
    
    # 测试思维链模型
    logger.info("测试思维链模型（MoE + 强化学习，类似DeepSeek-R1）...")
    cot_inference = ModelInference(model_type="cot")
    for prompt in test_cases:
        response = cot_inference.generate(prompt)
        print(f"\n问题: {prompt}")
        print(f"思维链模型回答: {response}")
    
    # 测试直接回答模型
    logger.info("\n测试直接回答模型（MoE + 传统训练，类似DeepSeek-V3）...")
    direct_inference = ModelInference(model_type="direct")
    for prompt in test_cases:
        response = direct_inference.generate(prompt)
        print(f"\n问题: {prompt}")
        print(f"直接回答模型回答: {response}")

if __name__ == "__main__":
    main() 