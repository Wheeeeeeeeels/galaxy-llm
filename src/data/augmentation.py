import random
import jieba
from typing import List, Dict
import logging

class DataAugmentor:
    def __init__(self):
        # 问题模板
        self.question_templates = {
            "学校名称": [
                "{}的学校名称是什么？",
                "请问{}叫什么名字？",
                "能告诉我{}的名称吗？"
            ],
            "学校英文名称": [
                "{}的英文名称是什么？",
                "请问{}的英文名字是什么？",
                "{}的英文校名是什么？"
            ],
            "学校所在区域或地区": [
                "{}位于哪个区域？",
                "{}在哪个地区？",
                "请问{}所在的区域是哪里？"
            ],
            "中文地址": [
                "{}的具体地址是什么？",
                "{}的中文地址是什么？",
                "请问{}位于哪里？"
            ],
            "组织属性类别": [
                "{}是什么类型的学校？",
                "{}属于什么类别的组织？",
                "请问{}的办学性质是什么？"
            ],
            "教育阶段": [
                "{}属于什么教育阶段？",
                "{}是什么层次的学校？",
                "请问{}提供什么阶段的教育？"
            ],
            "学生性别组成": [
                "{}是男校、女校还是男女校？",
                "{}招收什么性别的学生？",
                "请问{}的学生性别构成是怎样的？"
            ],
            "学校宗教信仰关联": [
                "{}有什么宗教背景吗？",
                "{}是否有宗教信仰？",
                "请问{}与哪个宗教有关联？"
            ],
            "建校年份": [
                "{}是什么时候建校的？",
                "{}创办于哪一年？",
                "请问{}的建校历史有多久了？"
            ]
        }
        
        # 答案模板
        self.answer_templates = {
            "学校名称": [
                "{}的学校名称是：{}",
                "{}叫{}",
                "{}的名字是{}"
            ],
            "学校英文名称": [
                "{}的英文名称是：{}",
                "{}的英文校名是{}",
                "{}英文名字叫{}"
            ],
            "学校所在区域或地区": [
                "{}位于{}",
                "{}在{}",
                "{}所在的区域是{}"
            ],
            "中文地址": [
                "{}的地址是：{}",
                "{}位于{}",
                "{}的具体位置在{}"
            ],
            "组织属性类别": [
                "{}是{}类型的学校",
                "{}属于{}类别",
                "{}的办学性质是{}"
            ],
            "教育阶段": [
                "{}是{}阶段的学校",
                "{}提供{}教育",
                "{}属于{}"
            ],
            "学生性别组成": [
                "{}是{}",
                "{}招收{}学生",
                "{}的学生性别构成是{}"
            ],
            "学校宗教信仰关联": [
                "{}的宗教背景是{}",
                "{}是{}学校",
                "{}与{}有关联"
            ],
            "建校年份": [
                "{}建校于{}",
                "{}创办于{}",
                "{}成立于{}"
            ]
        }
        
    def augment_qa(self, qa_pair: Dict) -> List[Dict]:
        """增强单个问答对"""
        augmented_pairs = []
        
        # 获取原始问答对的字段和内容
        field = list(qa_pair.keys())[0]
        text = qa_pair[field]["text"]
        
        # 解析原始问答
        question, answer = text.split("？", 1)
        school_name = question.split("的")[0]
        answer_content = answer.split("是：")[-1] if "是：" in answer else answer.split("是")[-1]
        
        # 使用不同模板生成新的问答对
        if field in self.question_templates and field in self.answer_templates:
            for q_template in self.question_templates[field]:
                for a_template in self.answer_templates[field]:
                    new_question = q_template.format(school_name)
                    new_answer = a_template.format(school_name, answer_content)
                    
                    augmented_pairs.append({
                        field: {
                            "text": new_question + "？" + new_answer
                        }
                    })
        
        return augmented_pairs
        
    def augment_school_data(self, qa_pairs: List[Dict]) -> List[Dict]:
        """对学校数据进行增强"""
        augmented_data = []
        
        for qa_pair in qa_pairs:
            # 保留原始问答对
            augmented_data.append(qa_pair)
            
            # 生成增强的问答对
            augmented_pairs = self.augment_qa(qa_pair)
            augmented_data.extend(augmented_pairs)
            
        return augmented_data

    def synonym_replacement(self, text: str, replace_ratio: float = 0.1) -> str:
        """同义词替换"""
        words = list(jieba.cut(text))
        n_words = len(words)
        n_replace = max(1, int(n_words * replace_ratio))
        
        for _ in range(n_replace):
            idx = random.randint(0, n_words - 1)
            word = words[idx]
            
            # 只使用预定义的同义词词典
            if word in self.synonym_dict:
                synonyms_list = self.synonym_dict[word]
                if synonyms_list:
                    words[idx] = random.choice(synonyms_list)
                    
        return ''.join(words)
        
    def random_deletion(self, text: str, delete_ratio: float = 0.1) -> str:
        """随机删除"""
        words = list(jieba.cut(text))
        n_words = len(words)
        n_delete = max(1, int(n_words * delete_ratio))
        
        for _ in range(n_delete):
            if len(words) > 1:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
                
        return ''.join(words)
        
    def random_swap(self, text: str, swap_ratio: float = 0.1) -> str:
        """随机交换"""
        words = list(jieba.cut(text))
        n_words = len(words)
        n_swap = max(1, int(n_words * swap_ratio))
        
        for _ in range(n_swap):
            if len(words) > 1:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                
        return ''.join(words)
        
    def augment_text(self, text: str, augment_ratio: float = 0.3) -> List[str]:
        """对文本进行多种增强"""
        augmented_texts = [text]
        
        if random.random() < augment_ratio:
            augmented_texts.append(self.synonym_replacement(text))
        if random.random() < augment_ratio:
            augmented_texts.append(self.random_deletion(text))
        if random.random() < augment_ratio:
            augmented_texts.append(self.random_swap(text))
            
        return augmented_texts
        
    def build_synonym_dict(self, texts: List[str]) -> Dict:
        """构建同义词词典"""
        # 使用预定义的同义词词典
        basic_dict = {
            "学校": ["院校", "校园"],
            "教育": ["培养", "教学"],
            "学生": ["学员", "同学"],
            "老师": ["教师", "导师"],
            "课程": ["科目", "教程"],
            "教学": ["授课", "讲授"],
            "学习": ["学习", "求学"],
            "考试": ["测试", "考核"],
            "成绩": ["分数", "成果"],
            "毕业": ["结业", "完成学业"]
        }
        return basic_dict 