import random
from typing import List, Dict, Any
import jieba
from pathlib import Path
import json
import requests
import time
from online_llm import online_llm_streaming
from tqdm import tqdm
import datetime

class SchoolDataAugmentor:
    def __init__(self, input_file: str, output_dir: str):
        """
        初始化数据增强器
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录路径
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载同义词词典
        self.synonyms_dict = self._load_synonyms()
        
        # 初始化大模型服务
        self.llm = online_llm_streaming(input_query="", api_key="app-Y3l0oHk6c80SqTrO1Ep0UMVk")
        
        # 初始化时间统计
        self.start_time = None
        self.total_items = 0
        self.processed_items = 0
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """
        加载同义词词典
        
        Returns:
            同义词词典
        """
        # 这里可以加载自定义的同义词词典
        # 示例词典
        return {
            '学校': ['学院', '学府', '教育机构'],
            '课程': ['科目', '学科', '教学计划'],
            '教师': ['老师', '教员', '教育工作者'],
            '学生': ['学员', '学子', '受教育者'],
            '设施': ['设备', '硬件', '基础设施'],
            '成绩': ['分数', '表现', '学业水平'],
            '升学': ['升学率', '升学情况', '升学去向'],
            '学费': ['学杂费', '教育费用', '学习费用'],
            '文化': ['氛围', '环境', '校园风气'],
            '位置': ['地点', '地址', '地理位置'],
            '活动': ['项目', '计划', '安排'],
            '条件': ['要求', '标准', '资格'],
            '特色': ['特点', '优势', '亮点'],
            '设施': ['设备', '硬件', '基础设施'],
            '环境': ['氛围', '条件', '状况']
        }
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        同义词替换
        
        Args:
            text: 输入文本
            n: 替换次数
            
        Returns:
            替换后的文本
        """
        words = list(jieba.cut(text))
        n = min(n, len(words))
        
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word in self.synonyms_dict]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.synonyms_dict[random_word]
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
            if num_replaced >= n:
                break
        
        return ''.join(new_words)
    
    def sentence_reordering(self, text: str) -> str:
        """
        句子重组
        
        Args:
            text: 输入文本
            
        Returns:
            重组后的文本
        """
        sentences = text.split('。')
        if len(sentences) <= 1:
            return text
            
        # 保留第一句和最后一句
        first_sentence = sentences[0]
        last_sentence = sentences[-1]
        middle_sentences = sentences[1:-1]
        
        # 随机重组中间句子
        random.shuffle(middle_sentences)
        
        # 重新组合
        return '。'.join([first_sentence] + middle_sentences + [last_sentence]) + '。'
    
    def noise_injection(self, text: str, noise_level: float = 0.1) -> str:
        """
        噪声注入
        
        Args:
            text: 输入文本
            noise_level: 噪声水平
            
        Returns:
            注入噪声后的文本
        """
        words = list(jieba.cut(text))
        n_noise = int(len(words) * noise_level)
        
        # 随机选择位置注入噪声
        noise_positions = random.sample(range(len(words)), n_noise)
        for pos in noise_positions:
            # 只在非关键位置注入噪声
            if not any(keyword in words[pos] for keyword in ['学校', '地址', '电话', '邮箱', '网站']):
                words[pos] = random.choice(['的', '了', '是', '在', '有'])
            
        return ''.join(words)
    
    def text_expansion(self, text: str, context: Dict[str, Any]) -> str:
        """
        文本扩展
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            扩展后的文本
        """
        try:
            # 构建提示
            prompt = f"""请基于以下信息，生成更详细的描述：
            
学校名称：{context.get('school_name', '')}
地址：{context.get('address_zh', '')}
类型：{context.get('organization_type', '')}
教育阶段：{context.get('education_stage', '')}

原始描述：{text}

请生成一个更详细、更生动的描述，保持原有信息的准确性。"""
            
            # 调用大模型服务
            self.llm.inputs = {"input_query": prompt}
            expanded_text = self.llm.run()
            
            return expanded_text
        except Exception as e:
            print(f"文本扩展失败: {e}")
            return text
    
    def entity_replacement(self, text: str, context: Dict[str, Any]) -> str:
        """
        实体替换
        
        Args:
            text: 输入文本
            context: 上下文信息
            
        Returns:
            替换后的文本
        """
        # 实体映射
        entity_map = {
            '学校': context.get('school_name', '学校'),
            '地址': context.get('address_zh', '地址'),
            '电话': context.get('phone', '电话'),
            '邮箱': context.get('email', '邮箱'),
            '网站': context.get('website', '网站'),
            '校长': context.get('principal', '校长'),
            '校监': context.get('supervisor', '校监')
        }
        
        # 替换实体
        for entity, replacement in entity_map.items():
            if entity in text:
                text = text.replace(entity, replacement)
        
        return text
    
    def grammar_transformation(self, text: str) -> str:
        """
        语法变换
        
        Args:
            text: 输入文本
            
        Returns:
            变换后的文本
        """
        # 主动句变被动句
        active_to_passive = {
            '提供': '被提供',
            '开展': '被开展',
            '举办': '被举办',
            '设置': '被设置',
            '安排': '被安排',
            '组织': '被组织',
            '实施': '被实施',
            '开展': '被开展',
            '进行': '被进行',
            '完成': '被完成'
        }
        
        # 肯定句变否定句
        positive_to_negative = {
            '是': '不是',
            '有': '没有',
            '可以': '不可以',
            '能够': '不能够',
            '必须': '不必',
            '需要': '不需要',
            '应该': '不应该',
            '要': '不要',
            '会': '不会',
            '能': '不能'
        }
        
        # 随机选择一种变换
        if random.random() < 0.5:
            # 主动变被动
            for active, passive in active_to_passive.items():
                if active in text:
                    text = text.replace(active, passive)
        else:
            # 肯定变否定
            for positive, negative in positive_to_negative.items():
                if positive in text:
                    text = text.replace(positive, negative)
        
        return text
    
    def augment_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        增强数据
        
        Args:
            data: 原始数据
            
        Returns:
            增强后的数据
        """
        augmented_data = []
        self.total_items = len(data)
        self.processed_items = 0
        
        # 创建进度条
        pbar = tqdm(total=self.total_items, desc="数据增强进度")
        
        for item in data:
            # 原始数据
            augmented_data.append(item)
            
            # 获取需要增强的文本字段
            text_fields = [
                'admission_requirements_cleaned',
                'school_activities_cleaned',
                'cot_reasoning'
            ]
            
            # 需要保留的重要字段
            important_fields = [
                'school_name',
                'school_name_en',
                'district',
                'address_zh',
                'address_en',
                'phone',
                'fax',
                'email',
                'website',
                'organization_type',
                'organization_type_en',
                'education_stage',
                'gender_ratio',
                'religion',
                'founding_year',
                'principal',
                'free_education',
                'supervisor',
                'operation_status',
                'total_teachers',
                'registered_teachers',
                'classroom_count'
            ]
            
            # 对每个文本字段进行增强
            for field in text_fields:
                if field in item['features'] and item['features'][field]:
                    text = item['features'][field]
                    
                    # 同义词替换
                    augmented_text = self.synonym_replacement(text)
                    augmented_item = {
                        'raw_data': item['raw_data'].copy(),
                        'features': item['features'].copy()
                    }
                    # 保留所有重要字段
                    for important_field in important_fields:
                        if important_field in item['features']:
                            augmented_item['features'][important_field] = item['features'][important_field]
                    augmented_item['features'][field] = augmented_text
                    augmented_data.append(augmented_item)
                    
                    # 句子重组
                    augmented_text = self.sentence_reordering(text)
                    augmented_item = {
                        'raw_data': item['raw_data'].copy(),
                        'features': item['features'].copy()
                    }
                    # 保留所有重要字段
                    for important_field in important_fields:
                        if important_field in item['features']:
                            augmented_item['features'][important_field] = item['features'][important_field]
                    augmented_item['features'][field] = augmented_text
                    augmented_data.append(augmented_item)
                    
                    # 噪声注入
                    augmented_text = self.noise_injection(text)
                    augmented_item = {
                        'raw_data': item['raw_data'].copy(),
                        'features': item['features'].copy()
                    }
                    # 保留所有重要字段
                    for important_field in important_fields:
                        if important_field in item['features']:
                            augmented_item['features'][important_field] = item['features'][important_field]
                    augmented_item['features'][field] = augmented_text
                    augmented_data.append(augmented_item)
                    
                    # 文本扩展
                    augmented_text = self.text_expansion(text, item['features'])
                    augmented_item = {
                        'raw_data': item['raw_data'].copy(),
                        'features': item['features'].copy()
                    }
                    # 保留所有重要字段
                    for important_field in important_fields:
                        if important_field in item['features']:
                            augmented_item['features'][important_field] = item['features'][important_field]
                    augmented_item['features'][field] = augmented_text
                    augmented_data.append(augmented_item)
                    
                    # 实体替换
                    augmented_text = self.entity_replacement(text, item['features'])
                    augmented_item = {
                        'raw_data': item['raw_data'].copy(),
                        'features': item['features'].copy()
                    }
                    # 保留所有重要字段
                    for important_field in important_fields:
                        if important_field in item['features']:
                            augmented_item['features'][important_field] = item['features'][important_field]
                    augmented_item['features'][field] = augmented_text
                    augmented_data.append(augmented_item)
                    
                    # 语法变换
                    augmented_text = self.grammar_transformation(text)
                    augmented_item = {
                        'raw_data': item['raw_data'].copy(),
                        'features': item['features'].copy()
                    }
                    # 保留所有重要字段
                    for important_field in important_fields:
                        if important_field in item['features']:
                            augmented_item['features'][important_field] = item['features'][important_field]
                    augmented_item['features'][field] = augmented_text
                    augmented_data.append(augmented_item)
                    
                    # 添加延迟以避免 API 限制
                    time.sleep(1)
            
            # 更新进度
            self.processed_items += 1
            pbar.update(1)
            
            # 计算预计剩余时间
            if self.processed_items > 0:
                elapsed_time = time.time() - self.start_time
                items_per_second = self.processed_items / elapsed_time
                remaining_items = self.total_items - self.processed_items
                estimated_remaining_time = remaining_items / items_per_second
                
                # 更新进度条描述
                pbar.set_description(
                    f"数据增强进度 (预计剩余时间: {datetime.timedelta(seconds=int(estimated_remaining_time))})"
                )
        
        pbar.close()
        return augmented_data
    
    def run(self):
        """
        运行数据增强流程
        """
        print("开始加载数据...")
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"跳过无效的 JSON 行: {e}")
                    continue
        
        print(f"加载了 {len(data)} 条数据")
        print("开始数据增强...")
        
        # 记录开始时间
        self.start_time = time.time()
        
        augmented_data = self.augment_data(data)
        
        # 计算总耗时
        total_time = time.time() - self.start_time
        print(f"增强完成，原始数据 {len(data)} 条，增强后 {len(augmented_data)} 条")
        print(f"总耗时: {datetime.timedelta(seconds=int(total_time))}")
        
        print("保存增强后的数据...")
        output_file = self.output_dir / 'augmented_schools.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in augmented_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("数据保存完成")

if __name__ == '__main__':
    # 设置输入输出路径
    input_file = 'data/processed/processed_schools.json'
    output_dir = 'data/processed'
    
    # 创建增强器并运行
    augmentor = SchoolDataAugmentor(input_file, output_dir)
    augmentor.run() 