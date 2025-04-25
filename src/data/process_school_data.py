import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import re
import os
import random
from tqdm import tqdm
from online_llm import online_llm_streaming

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化在线大模型
llm = online_llm_streaming("", api_key="app-PlB4VJ33aCx2J32ebVAQYV9i")

def clean_text(text: str) -> str:
    """清理文本，去除多余的空格和换行"""
    return re.sub(r'\s+', ' ', text).strip()

def extract_answer(text: str) -> str:
    """从问答文本中提取答案"""
    text = clean_text(text)
    if "是：" in text:
        return text.split("是：")[-1]
    elif "是" in text:
        return text.split("是")[-1]
    return text

def load_school_data(file_path: str) -> List[Dict]:
    """加载学校数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    # 解析每行的JSON数组
                    school_data = json.loads(line)
                    if isinstance(school_data, list):
                        data.extend(school_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON时出错: {str(e)}")
                    continue
        logger.info(f"成功加载 {len(data)} 条数据记录")
        return data
    except Exception as e:
        logger.error(f"加载数据时发生错误: {str(e)}")
        raise

def extract_school_info(qa_pairs: List[Dict]) -> Dict:
    """从问答对中提取学校信息"""
    school_info = {}
    field_mapping = {
        '学校名称': '学校名称',
        '学校类型': '学校类型',
        '地址': '地址',
        '学校规模': '规模',
        '师资力量': '师资',
        '教学设施': '设施',
        '学校特色': '特色',
        '办学理念': '理念',
        '学校性质': '性质',
        '学校级别': '级别',
        '创办年份': '创办年份',
        '所属地区': '地区'
    }
    
    for qa_pair in qa_pairs:
        for field in qa_pair:
            if field in field_mapping:
                text = qa_pair[field]["text"]
                
                # 跳过"是否存在"类型的问答
                if "是否存在" in text:
                    continue
                    
                # 提取答案
                answer = extract_answer(text)
                if answer and answer != "未知" and answer != "不详":
                    school_info[field_mapping[field]] = answer
                    
    return school_info

def process_school_data(data: List[Dict]) -> List[Dict]:
    """处理学校数据，将每个学校的问答对转换为结构化信息"""
    processed_data = []
    current_school = []
    
    for item in data:
        # 检查是否开始新的学校
        if "学校名称" in item and "是否存在" not in item["学校名称"]["text"]:
            if current_school:
                school_info = extract_school_info(current_school)
                if school_info:  # 只添加非空的学校信息
                    processed_data.append(school_info)
            current_school = [item]
        else:
            current_school.append(item)
            
    # 处理最后一个学校
    if current_school:
        school_info = extract_school_info(current_school)
        if school_info:  # 只添加非空的学校信息
            processed_data.append(school_info)
        
    logger.info(f"成功处理 {len(processed_data)} 所学校的信息")
    return processed_data

def save_processed_data(data: List[Dict], output_path: str) -> None:
    """保存处理后的数据"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"数据已保存到 {output_path}")
    except Exception as e:
        logger.error(f"保存数据时发生错误: {str(e)}")
        raise

def generate_enhanced_info(school_info: Dict[str, str]) -> Dict[str, str]:
    """使用在线大模型生成增强的学校信息"""
    try:
        # 构建提示词
        prompt = f"""请根据以下学校信息，生成更详细和专业的描述：
学校名称：{school_info.get('学校名称', '未知')}
学校类型：{school_info.get('学校类型', '未知')}
地址：{school_info.get('地址', '未知')}
规模：{school_info.get('规模', '未知')}
师资：{school_info.get('师资', '未知')}
设施：{school_info.get('设施', '未知')}
特色：{school_info.get('特色', '未知')}
理念：{school_info.get('理念', '未知')}

请生成以下内容：
1. 更详细的学校介绍
2. 教学特色和优势
3. 师资力量的具体描述
4. 校园环境和设施的描述
5. 办学理念的详细阐述"""

        # 生成回答
        llm_instance = online_llm_streaming(prompt)
        response = llm_instance.run()
        
        # 解析回答并更新学校信息
        enhanced_info = school_info.copy()
        enhanced_info['详细描述'] = response
        
        return enhanced_info
    except Exception as e:
        logger.error(f"生成增强信息失败: {str(e)}")
        return school_info

def generate_enhanced_examples(school_info: Dict[str, str], all_schools: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """生成增强的示例"""
    try:
        examples = []
        
        # 1. 基本信息查询
        prompt = f"""请根据以下学校信息，生成一个专业且详细的学校介绍：
{json.dumps(school_info, ensure_ascii=False, indent=2)}

请生成：
1. 一个专业的学校介绍
2. 学校的特色和优势
3. 学校的办学理念
4. 学校的师资和设施情况"""

        llm_instance = online_llm_streaming(prompt)
        response = llm_instance.run()
        examples.append({
            "instruction": "查询学校的基本信息",
            "input": f"请详细介绍{school_info.get('学校名称', '这所学校')}",
            "cot": "1. 首先需要理解用户想要了解学校的基本情况\n2. 然后从数据中提取学校的各项信息\n3. 最后将这些信息组织成专业的介绍",
            "output": response
        })
        
        # 2. 学校比较
        if len(all_schools) > 1:
            other_school = random.choice([s for s in all_schools if s['学校名称'] != school_info['学校名称']])
            prompt = f"""请比较以下两所学校：
学校1: {json.dumps(school_info, ensure_ascii=False, indent=2)}
学校2: {json.dumps(other_school, ensure_ascii=False, indent=2)}

请从以下方面进行比较：
1. 办学特色
2. 师资力量
3. 教学设施
4. 学校规模
5. 办学理念"""

            llm_instance = online_llm_streaming(prompt)
            response = llm_instance.run()
            examples.append({
                "instruction": "比较两所学校的特点",
                "input": f"比较{school_info.get('学校名称', '学校1')}和{other_school.get('学校名称', '学校2')}的特点",
                "cot": "1. 首先需要理解用户想要比较两所学校\n2. 然后分别提取两所学校的特点\n3. 对比两所学校在各个方面的差异\n4. 最后总结比较结果",
                "output": response
            })
        
        # 3. 学校推荐
        student_types = ['理科生', '文科生', '艺术生', '体育生']
        student_type = random.choice(student_types)
        prompt = f"""请根据以下学校信息，为{student_type}推荐合适的学校：
{json.dumps(school_info, ensure_ascii=False, indent=2)}

请考虑：
1. 学校的特色是否适合该类型学生
2. 学校的师资力量是否能够满足该类型学生的需求
3. 学校的设施是否适合该类型学生的发展
4. 学校的办学理念是否与该类型学生的培养目标相符"""

        llm_instance = online_llm_streaming(prompt)
        response = llm_instance.run()
        examples.append({
            "instruction": "根据条件推荐合适的学校",
            "input": f"推荐适合{student_type}的学校",
            "cot": "1. 首先分析学生的特点和需求\n2. 然后评估各个学校的匹配度\n3. 考虑地理位置、专业设置等因素\n4. 最后给出推荐理由",
            "output": response
        })
        
        return examples
    except Exception as e:
        logger.error(f"生成增强示例失败: {str(e)}")
        return []

def process_data():
    """处理数据并生成训练集和评估集"""
    try:
        # 创建输出目录
        os.makedirs("data/training", exist_ok=True)
        
        # 加载原始数据
        logger.info("开始加载原始数据...")
        raw_data = load_school_data("data/学校清洗数据.txt")
        
        # 处理学校数据
        logger.info("开始处理学校数据...")
        processed_data = process_school_data(raw_data)
        
        # 生成带思维链的数据
        logger.info("开始生成带思维链的数据...")
        cot_data = []
        for school_info in tqdm(processed_data, desc="生成带思维链数据"):
            # 使用在线大模型生成增强的示例
            enhanced_examples = generate_enhanced_examples(school_info, processed_data)
            if enhanced_examples:
                cot_data.extend(enhanced_examples)
        
        # 生成不带思维链的数据
        logger.info("开始生成不带思维链的数据...")
        direct_data = []
        for school_info in tqdm(processed_data, desc="生成不带思维链数据"):
            # 使用在线大模型生成增强的示例
            enhanced_examples = generate_enhanced_examples(school_info, processed_data)
            if enhanced_examples:
                for example in enhanced_examples:
                    # 移除思维链
                    direct_example = {
                        "instruction": example["instruction"],
                        "input": example["input"],
                        "output": example["output"]
                    }
                    direct_data.append(direct_example)
        
        # 打乱数据
        logger.info("打乱数据...")
        random.shuffle(cot_data)
        random.shuffle(direct_data)
        
        # 划分训练集和评估集
        cot_train_size = int(len(cot_data) * 0.8)
        direct_train_size = int(len(direct_data) * 0.8)
        
        # 保存带思维链的数据
        logger.info("保存带思维链的数据...")
        save_processed_data(cot_data[:cot_train_size], "data/training/cot_train.json")
        save_processed_data(cot_data[cot_train_size:], "data/training/cot_eval.json")
        
        # 保存不带思维链的数据
        logger.info("保存不带思维链的数据...")
        save_processed_data(direct_data[:direct_train_size], "data/training/direct_train.json")
        save_processed_data(direct_data[direct_train_size:], "data/training/direct_eval.json")
        
        logger.info(f"带思维链数据：训练集 {cot_train_size} 条，评估集 {len(cot_data) - cot_train_size} 条")
        logger.info(f"不带思维链数据：训练集 {direct_train_size} 条，评估集 {len(direct_data) - direct_train_size} 条")
        
    except Exception as e:
        logger.error(f"处理数据时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    process_data() 