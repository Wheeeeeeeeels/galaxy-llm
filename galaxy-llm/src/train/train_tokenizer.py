import os
import json
from collections import Counter
import logging
import jieba

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JiebaTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.idx2token = {v: k for k, v in vocab.items()}
        # 添加自定义词典
        for word in vocab:
            if len(word) > 1 and not word.startswith('['):
                jieba.add_word(word)
    
    def tokenize(self, text):
        # 分词前预处理
        text = text.strip()
        # 使用精确模式分词
        tokens = list(jieba.cut(text, cut_all=False))
        # 过滤空白词
        tokens = [t for t in tokens if len(t.strip()) > 0]
        return tokens
    
    def encode(self, text):
        tokens = ["[CLS]"] + self.tokenize(text) + ["[SEP]"]
        # 将词转换为ID
        ids = []
        for token in tokens:
            # 如果词在词表中，使用对应的ID
            if token in self.vocab:
                ids.append(self.vocab[token])
            # 如果词不在词表中，尝试按字符拆分
            else:
                for char in token:
                    if char in self.vocab:
                        ids.append(self.vocab[char])
                    else:
                        ids.append(self.vocab["[UNK]"])
        return ids
    
    def decode(self, ids):
        tokens = []
        for idx in ids:
            token = self.idx2token.get(idx, "[UNK]")
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                tokens.append(token)
        return "".join(tokens)
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_pretrained(cls, save_directory):
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab)
    
    def __len__(self):
        return len(self.vocab)

def train_tokenizer():
    """训练分词器"""
    # 加载训练数据
    with open("data/学校纯文本数据.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 准备训练文本
    texts = []
    for item in train_data:
        if isinstance(item, dict) and 'text' in item:
            # 按分号分割文本
            sentences = item['text'].split('；')
            for sentence in sentences:
                # 处理问答对
                if '？' in sentence and sentence.count('？') == 1:
                    question, answer = sentence.split('？')
                    if answer.startswith('是：') or answer.startswith('是:'):
                        answer = answer[2:]
                    texts.append(question + '？')
                    if answer:
                        texts.append(answer)
                else:
                    texts.append(sentence)
    
    # 构建词汇表
    token_counter = Counter()
    for text in texts:
        tokens = jieba.cut(text.strip(), cut_all=False)
        token_counter.update(t for t in tokens if len(t.strip()) > 0)
    
    # 创建词汇表
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4
    }
    
    # 添加词语到词汇表（只添加出现次数大于1的词）
    for token, count in token_counter.most_common():
        if count > 1 and token not in vocab and len(token.strip()) > 0:
            vocab[token] = len(vocab)
    
    # 创建分词器
    tokenizer = JiebaTokenizer(vocab)
    
    # 保存分词器
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_pretrained("tokenizer")
    
    # 打印词汇表信息
    print(f"\n词汇表大小: {len(vocab)}")
    print("\n词汇表示例（前20个词）:")
    for token, idx in sorted(vocab.items(), key=lambda x: x[1])[:20]:
        print(f"{token}: {idx}")
    
    # 打印一些训练文本示例
    print("\n训练文本示例:")
    for text in texts[:5]:
        print(f"- {text}")
    
    # 测试分词器
    test_texts = [
        "香港的力行幼稚园的学校生活是什么？",
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
    
    print("\n测试分词结果:\n")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        tokens = tokenizer.tokenize(text)
        
        print(f"原文: {text}")
        print(f"分词: {tokens}")
        print(f"编码: {encoded}")
        print(f"解码: {decoded}")
        print("\n调试信息:")
        print(f"原始文本长度: {len(text)}")
        print(f"分词后长度: {len(tokens)}")
        print(f"编码后长度: {len(encoded)}")
        print(f"词汇表大小: {len(tokenizer)}")
        print("\n" + "="*50 + "\n")

def evaluate_tokenizer(tokenizer_path: str):
    """评估分词器
    
    Args:
        tokenizer_path: 分词器路径
    """
    # 加载分词器
    tokenizer = JiebaTokenizer.from_pretrained(tokenizer_path)
    
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
    
    # 评估结果
    print("\n分词器评估结果:")
    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        print(f"\n原文: {text}")
        print(f"分词: {tokens}")
        print(f"编码: {ids}")
        print(f"解码: {decoded}")

def main():
    # 训练分词器
    train_tokenizer()
    
    # 评估分词器
    evaluate_tokenizer("tokenizer")

if __name__ == "__main__":
    main() 