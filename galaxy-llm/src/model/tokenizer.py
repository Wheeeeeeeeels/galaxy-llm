import torch
import json
import os
from typing import List, Dict, Optional
import re

class ChineseTokenizer:
    """中文分词器"""
    def __init__(
        self,
        vocab_path: str = "data/vocab.json",
        max_length: int = 2048,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        mask_token: str = "[MASK]"
    ):
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        
        # 加载词表
        self.vocab = self._load_vocab(vocab_path)
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        # 特殊token的ID
        self.pad_token_id = self.vocab[pad_token]
        self.unk_token_id = self.vocab[unk_token]
        self.bos_token_id = self.vocab[bos_token]
        self.eos_token_id = self.vocab[eos_token]
        self.mask_token_id = self.vocab[mask_token]
        
    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """加载词表"""
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 创建基础词表
            vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3,
                self.mask_token: 4
            }
            # 保存词表
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)
            return vocab
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 按字符分词
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(char)
            else:
                tokens.append(self.unk_token)
                
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """编码文本"""
        # 分词
        tokens = self._tokenize(text)
        
        # 添加特殊token
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
            
        # 转换为ID
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # 截断
        if truncation and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            
        # 填充
        if padding:
            attention_mask = [1] * len(token_ids)
            if len(token_ids) < self.max_length:
                token_ids = token_ids + [self.pad_token_id] * (self.max_length - len(token_ids))
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))
        else:
            attention_mask = [1] * len(token_ids)
            
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码文本"""
        tokens = [self.id2token.get(token_id, self.unk_token) for token_id in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [
                self.pad_token,
                self.unk_token,
                self.bos_token,
                self.eos_token,
                self.mask_token
            ]]
            
        return "".join(tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """批量编码文本"""
        encoded = [self.encode(text, add_special_tokens, padding, truncation) for text in texts]
        
        return {
            "input_ids": torch.stack([item["input_ids"] for item in encoded]),
            "attention_mask": torch.stack([item["attention_mask"] for item in encoded])
        }
    
    def batch_decode(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """批量解码文本"""
        return [self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_list]
    
    def save_vocab(self, vocab_path: str):
        """保存词表"""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """构建词表"""
        # 统计词频
        word_freq = {}
        for text in texts:
            for char in text:
                word_freq[char] = word_freq.get(char, 0) + 1
                
        # 添加高频词到词表
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                
        # 更新id2token
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        # 保存词表
        self.save_vocab("data/vocab.json") 