import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import re
import os
import random
from tqdm import tqdm
from .online_llm import online_llm_streaming
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('school_data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化在线大模型
llm = online_llm_streaming("", api_key="app-PlB4VJ33aCx2J32ebVAQYV9i")

# 设置online_llm的日志级别
logging.getLogger('online_llm').setLevel(logging.WARNING)

class SchoolDataProcessor:
    def __init__(self):
        # 场景模板
        self.scene_templates = {
            "基础信息查询": [
                "{}的{}是什么？",
                "请问{}的{}是什么？",
                "{}的{}信息是什么？"
            ],
            "学校比较": [
                "{}和{}在{}方面有什么区别？",
                "比较{}和{}的{}",
                "{}和{}的{}对比如何？"
            ],
            "学校推荐": [
                "根据{}，推荐几所{}的学校",
                "请推荐几所{}的{}学校",
                "哪些学校在{}方面比较好？"
            ]
        }
        
        # 思维链模板
        self.chain_of_thought_templates = {
            "基础信息查询": [
                "要查询{}的{}，我需要：\n1. 确认学校名称\n2. 查找相关信息\n3. 整理答案\n\n答案是：{}",
                "查询{}的{}的步骤：\n1. 确定查询目标\n2. 收集相关信息\n3. 验证信息准确性\n\n最终答案是：{}"
            ],
            "学校比较": [
                "比较{}和{}的{}：\n1. 收集两所学校的信息\n2. 对比关键指标\n3. 分析差异\n\n比较结果是：{}",
                "分析{}和{}的{}差异：\n1. 确定比较维度\n2. 收集对比数据\n3. 总结差异\n\n分析结论：{}"
            ],
            "学校推荐": [
                "推荐{}的学校：\n1. 分析需求\n2. 筛选符合条件的学校\n3. 排序推荐\n\n推荐结果：{}",
                "根据{}推荐学校：\n1. 理解推荐标准\n2. 匹配学校特征\n3. 生成推荐列表\n\n推荐如下：{}"
            ]
        }

    def load_raw_data(self, file_path: str, limit: int = None) -> List[Dict]:
        """加载原始数据，可限制数量"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        # 解析JSON数组
                        school_data = json.loads(line)
                        if not isinstance(school_data, list):
                            continue
                            
                        # 提取学校信息
                        school_info = {}
                        for item in school_data:
                            if not isinstance(item, dict):
                                continue
                                
                            for key, value in item.items():
                                if isinstance(value, dict) and "text" in value:
                                    # 提取答案部分
                                    text = value["text"]
                                    if "是：" in text:
                                        answer = text.split("是：")[-1].strip()
                                    elif "是" in text:
                                        answer = text.split("是")[-1].strip()
                                    else:
                                        continue
                                        
                                    # 跳过"是否存在"的问题
                                    if "是否存在" in text:
                                        continue
                                        
                                    # 验证答案的有效性
                                    if answer and answer != "未知" and answer != "不详" and answer != "否":
                                        school_info[key] = answer
                        
                        # 验证学校数据的完整性
                        if school_info and "学校名称" in school_info:
                            # 确保学校名称不包含特殊字符
                            school_name = school_info["学校名称"]
                            if re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9\s]+$', school_name):
                                data.append(school_info)
                                if limit and len(data) >= limit:
                                    break
                                
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"处理学校数据时出错: {str(e)}")
                        continue
            
            logger.info(f"成功加载 {len(data)} 条有效数据")
            return data
            
        except Exception as e:
            logger.error(f"加载数据时发生错误: {str(e)}")
            return []
    
    def _process_school_data(self, school_data: Dict) -> Dict:
        """处理单个学校数据"""
        processed_data = {}
        
        # 定义字段映射
        field_mapping = {
            "学校名称": "name",
            "学校类型": "type",
            "学校地址": "address",
            "学校电话": "phone",
            "学校网站": "website",
            "学校简介": "description",
            "学校特色": "features",
            "学校规模": "scale",
            "学校设施": "facilities",
            "学校师资": "teachers",
            "学校课程": "courses",
            "学校活动": "activities",
            "学校荣誉": "honors",
            "学校历史": "history",
            "学校文化": "culture",
            "学校环境": "environment",
            "学校交通": "transportation",
            "学校周边": "surroundings",
            "学校安全": "safety",
            "学校管理": "management"
        }
        
        # 处理每个字段
        for cn_field, en_field in field_mapping.items():
            if cn_field in school_data:
                value = school_data[cn_field]
                
                # 数据清洗
                if isinstance(value, str):
                    # 移除特殊字符
                    value = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,，。！？!?]', '', value)
                    # 移除多余空格
                    value = re.sub(r'\s+', ' ', value).strip()
                    # 移除重复标点
                    value = re.sub(r'[.,，。！？!?]{2,}', lambda m: m.group()[0], value)
                    
                    # 验证数据有效性
                    if value and value != "未知" and value != "不详" and value != "否":
                        # 根据字段类型进行特定处理
                        if en_field == "name":
                            # 学校名称不能包含特殊字符
                            if re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9\s]+$', value):
                                processed_data[en_field] = value
                        elif en_field == "phone":
                            # 电话号码格式验证
                            if re.match(r'^[\d-]+$', value):
                                processed_data[en_field] = value
                        elif en_field == "website":
                            # 网站格式验证
                            if re.match(r'^https?://[\w.-]+\.\w+', value):
                                processed_data[en_field] = value
                        else:
                            processed_data[en_field] = value
        
        # 验证必要字段
        required_fields = ["name", "type", "address"]
        if all(field in processed_data for field in required_fields):
            return processed_data
        return {}

    def generate_base_qa(self, school_data: Dict) -> List[Dict]:
        """生成基础问答对"""
        qa_pairs = []
        try:
            for field, value in school_data.items():
                if field == "学校名称" or not value:
                    continue
                    
                # 生成简洁的问题
                question = f"{school_data['学校名称']}的{field}是什么？"
                answer = f"{value}"
                
                # 基础问答对
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "type": "基础信息查询"
                })
                
                # 带思维链的问答对
                cot_answer = f"要查询{school_data['学校名称']}的{field}：\n1. 确认学校名称\n2. 查找相关信息\n\n答案是：{value}"
                qa_pairs.append({
                    "question": question,
                    "answer": cot_answer,
                    "type": "基础信息查询",
                    "chain_of_thought": True
                })
            
            return qa_pairs
        except Exception as e:
            logger.error(f"生成基础问答对时出错: {str(e)}")
            return []

    def generate_comparison_qa(self, school1: Dict, school2: Dict) -> List[Dict]:
        """生成学校比较问答对"""
        qa_pairs = []
        try:
            common_fields = set(school1.keys()) & set(school2.keys())
            
            for field in common_fields:
                if field == "学校名称" or not school1[field] or not school2[field]:
                    continue
                    
                # 生成简洁的比较问题
                question = f"{school1['学校名称']}和{school2['学校名称']}的{field}有什么区别？"
                answer = f"{school1['学校名称']}：{school1[field]}\n{school2['学校名称']}：{school2[field]}"
                
                # 基础问答对
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "type": "学校比较"
                })
                
                # 带思维链的问答对
                cot_answer = f"比较{school1['学校名称']}和{school2['学校名称']}的{field}：\n1. 收集两所学校的信息\n2. 对比数据\n\n比较结果：\n{school1['学校名称']}：{school1[field]}\n{school2['学校名称']}：{school2[field]}"
                qa_pairs.append({
                    "question": question,
                    "answer": cot_answer,
                    "type": "学校比较",
                    "chain_of_thought": True
                })
            
            return qa_pairs
        except Exception as e:
            logger.error(f"生成比较问答对时出错: {str(e)}")
            return []

    def generate_recommendation_qa(self, schools: List[Dict]) -> List[Dict]:
        """生成学校推荐问答对"""
        qa_pairs = []
        try:
            if not schools:
                return qa_pairs
            
            # 按不同维度分组学校
            schools_by_region = {}  # 按区域分组
            schools_by_stage = {}   # 按教育阶段分组
            schools_by_type = {}    # 按组织属性分组
            
            for school in schools:
                # 区域分组
                region = school.get("学校所在区域或地区", "其他")
                if region not in schools_by_region:
                    schools_by_region[region] = []
                schools_by_region[region].append(school)
                
                # 教育阶段分组
                stage = school.get("教育阶段", "其他")
                if stage not in schools_by_stage:
                    schools_by_stage[stage] = []
                schools_by_stage[stage].append(school)
                
                # 组织属性分组
                school_type = school.get("组织属性类别", "其他")
                if school_type not in schools_by_type:
                    schools_by_type[school_type] = []
                schools_by_type[school_type].append(school)
            
            # 为每个分组生成推荐
            for group_name, group_schools in [
                ("区域", schools_by_region),
                ("教育阶段", schools_by_stage),
                ("组织属性", schools_by_type)
            ]:
                for key, schools_list in group_schools.items():
                    if len(schools_list) < 2:  # 降低门槛到2所学校
                        continue
                    
                    # 随机选择学校（如果不足6所则全部选择）
                    selected_schools = random.sample(schools_list, min(6, len(schools_list)))
                    
                    # 构建提示词
                    prompt = f"""请根据以下{key}的学校信息，生成一个专业的学校推荐问答对。
要求：
1. 问题要简洁明了，直接询问推荐学校
2. 答案要包含每所学校的关键信息，如学校名称、教育阶段、组织属性等
3. 不要添加任何不存在的信息
4. 保持客观、专业的语气

学校信息：
{json.dumps(selected_schools, ensure_ascii=False, indent=2)}

请生成：
1. 一个推荐问题
2. 一个包含思维链的答案
3. 一个不包含思维链的答案"""

                    # 调用在线大模型生成回答
                    llm_instance = online_llm_streaming(prompt)
                    response = llm_instance.run()
                    
                    try:
                        # 解析回答
                        response_data = json.loads(response)
                        question = response_data.get("question", f"请推荐{key}的学校")
                        answer_with_cot = response_data.get("answer_with_cot", "")
                        answer_without_cot = response_data.get("answer_without_cot", "")
                        
                        # 添加带思维链的问答对
                        qa_pairs.append({
                            "question": question,
                            "answer": answer_with_cot,
                            "type": "学校推荐",
                            "chain_of_thought": True
                        })
                        
                        # 添加不带思维链的问答对
                        qa_pairs.append({
                            "question": question,
                            "answer": answer_without_cot,
                            "type": "学校推荐"
                        })
                        
                    except json.JSONDecodeError:
                        # 如果解析失败，使用默认格式
                        qa_pairs.append({
                            "question": f"请推荐{key}的学校",
                            "answer": response,
                            "type": "学校推荐"
                        })
                    
                    # 生成特定维度的推荐
                    dimensions = [
                        ("教师团队规模", "师资力量"),
                        ("校园情况", "校园环境"),
                        ("教育宗旨", "办学特色"),
                        ("学校规模", "办学规模"),
                        ("学校活动", "特色活动")
                    ]
                    
                    for field, desc in dimensions:
                        valid_schools = [s for s in schools_list if s.get(field)]
                        if len(valid_schools) < 2:  # 同样降低门槛
                            continue
                        
                        selected_schools = random.sample(valid_schools, min(6, len(valid_schools)))
                        
                        # 构建提示词
                        prompt = f"""请根据以下{key}的学校在{desc}方面的信息，生成一个专业的学校推荐问答对。
要求：
1. 问题要简洁明了，直接询问推荐在{desc}方面表现突出的学校
2. 答案要包含每所学校的关键信息，如学校名称、{desc}等
3. 不要添加任何不存在的信息
4. 保持客观、专业的语气

学校信息：
{json.dumps(selected_schools, ensure_ascii=False, indent=2)}

请生成：
1. 一个推荐问题
2. 一个包含思维链的答案
3. 一个不包含思维链的答案"""

                        # 调用在线大模型生成回答
                        llm_instance = online_llm_streaming(prompt)
                        response = llm_instance.run()
                        
                        try:
                            # 解析回答
                            response_data = json.loads(response)
                            question = response_data.get("question", f"请推荐{key}在{desc}方面表现突出的学校")
                            answer_with_cot = response_data.get("answer_with_cot", "")
                            answer_without_cot = response_data.get("answer_without_cot", "")
                            
                            # 添加带思维链的问答对
                            qa_pairs.append({
                                "question": question,
                                "answer": answer_with_cot,
                                "type": "学校推荐",
                                "chain_of_thought": True
                            })
                            
                            # 添加不带思维链的问答对
                            qa_pairs.append({
                                "question": question,
                                "answer": answer_without_cot,
                                "type": "学校推荐"
                            })
                            
                        except json.JSONDecodeError:
                            # 如果解析失败，使用默认格式
                            qa_pairs.append({
                                "question": f"请推荐{key}在{desc}方面表现突出的学校",
                                "answer": response,
                                "type": "学校推荐"
                            })
            
            return qa_pairs
        except Exception as e:
            logger.error(f"生成推荐问答对时出错: {str(e)}")
            return []

    def generate_common_sense_qa(self) -> List[Dict]:
        """生成常识问答对"""
        common_sense_pairs = [
            {
                "question": "香港的教育体系包括哪些阶段？",
                "answer": "香港的教育体系包括幼稚园、小学、中学和大学等阶段。",
                "type": "常识问答",
                "chain_of_thought": False
            },
            {
                "question": "香港的教育体系包括哪些阶段？",
                "answer": "要了解香港的教育体系：\n1. 首先是最基础的幼稚园教育\n2. 然后是6年的小学教育\n3. 接着是6年的中学教育\n4. 最后是高等教育\n\n所以香港的教育体系包括幼稚园、小学、中学和大学等阶段。",
                "type": "常识问答",
                "chain_of_thought": True
            },
            {
                "question": "香港的学校主要分为哪几类？",
                "answer": "香港的学校主要分为官立学校、资助学校、直资学校和私立学校。",
                "type": "常识问答",
                "chain_of_thought": False
            },
            {
                "question": "香港的学校主要分为哪几类？",
                "answer": "分析香港学校分类：\n1. 官立学校由政府直接管理\n2. 资助学校接受政府资助\n3. 直资学校部分接受政府资助\n4. 私立学校完全自筹资金\n\n所以香港的学校主要分为官立学校、资助学校、直资学校和私立学校。",
                "type": "常识问答",
                "chain_of_thought": True
            },
            {
                "question": "什么是DSE考试？",
                "answer": "DSE是香港中学文凭考试，是香港中学生完成中学课程后的公开考试。",
                "type": "常识问答",
                "chain_of_thought": False
            },
            {
                "question": "什么是DSE考试？",
                "answer": "理解DSE考试：\n1. DSE是香港中学文凭考试的简称\n2. 这是香港中学生完成中学课程后的公开考试\n3. 考试成绩用于大学入学申请\n\n所以DSE是香港中学文凭考试，是香港中学生完成中学课程后的公开考试。",
                "type": "常识问答",
                "chain_of_thought": True
            }
        ]
        return common_sense_pairs

    def generate_general_knowledge_qa(self) -> List[Dict]:
        """生成通用知识问答对"""
        general_knowledge_pairs = [
            {
                "question": "1+1等于多少？",
                "answer": "1+1=2",
                "type": "通用知识",
                "chain_of_thought": False
            },
            {
                "question": "1+1等于多少？",
                "answer": "计算1+1：\n1. 第一个数字是1\n2. 第二个数字是1\n3. 将两个数字相加\n\n所以1+1=2",
                "type": "通用知识",
                "chain_of_thought": True
            },
            {
                "question": "一年有多少个月？",
                "answer": "一年有12个月。",
                "type": "通用知识",
                "chain_of_thought": False
            },
            {
                "question": "一年有多少个月？",
                "answer": "分析一年的月份：\n1. 一年分为四个季度\n2. 每个季度有3个月\n3. 4个季度共12个月\n\n所以一年有12个月。",
                "type": "通用知识",
                "chain_of_thought": True
            },
            {
                "question": "一周有多少天？",
                "answer": "一周有7天。",
                "type": "通用知识",
                "chain_of_thought": False
            },
            {
                "question": "一周有多少天？",
                "answer": "分析一周的天数：\n1. 一周从星期一开始\n2. 到星期日结束\n3. 共7天\n\n所以一周有7天。",
                "type": "通用知识",
                "chain_of_thought": True
            }
        ]
        return general_knowledge_pairs

    def process_data(self, input_file: str, output_dir: str, max_workers: int = 8, test_mode: bool = False, test_size: int = 100):
        """处理数据并生成训练集，使用多线程加速
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            max_workers: 最大线程数
            test_mode: 是否启用测试模式
            test_size: 测试模式下的数据量
        """
        try:
            # 加载原始数据
            if test_mode:
                logging.info(f"测试模式：正在加载{test_size}条数据...")
                raw_data = self.load_raw_data(input_file, limit=test_size)
            else:
                logging.info("正在加载原始数据...")
                raw_data = self.load_raw_data(input_file)
            
            if not raw_data:
                logging.error("未能加载任何数据，请检查输入文件")
                return
            
            logging.info(f"加载完成，共 {len(raw_data)} 条数据")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取当前日期
            current_date = datetime.now().strftime("%Y%m%d")
            
            # 使用线程池处理数据
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 生成基础问答对
                logging.info("正在生成基础问答对...")
                base_qa_futures = [
                    executor.submit(self.generate_base_qa, school)
                    for school in raw_data
                ]
                base_qa_pairs = []
                for future in tqdm(base_qa_futures, desc="处理基础问答"):
                    try:
                        result = future.result()
                        if result:
                            base_qa_pairs.extend(result)
                    except Exception as e:
                        logging.error(f"生成基础问答对时出错: {str(e)}")
                
                logging.info(f"基础问答对生成完成，共 {len(base_qa_pairs)} 条")
                
                # 生成比较问答对
                logging.info("正在生成比较问答对...")
                comparison_qa_futures = []
                for i in range(len(raw_data)):
                    for j in range(i + 1, min(i + 10, len(raw_data))):
                        comparison_qa_futures.append(
                            executor.submit(
                                self.generate_comparison_qa,
                                raw_data[i],
                                raw_data[j]
                            )
                        )
                
                comparison_qa_pairs = []
                for future in tqdm(comparison_qa_futures, desc="处理比较问答"):
                    try:
                        result = future.result()
                        if result:
                            comparison_qa_pairs.extend(result)
                    except Exception as e:
                        logging.error(f"生成比较问答对时出错: {str(e)}")
                
                logging.info(f"比较问答对生成完成，共 {len(comparison_qa_pairs)} 条")
                
                # 生成推荐问答对
                logging.info("正在生成推荐问答对...")
                recommendation_qa_futures = [
                    executor.submit(self.generate_recommendation_qa, raw_data)
                ]
                recommendation_qa_pairs = []
                for future in tqdm(recommendation_qa_futures, desc="处理推荐问答"):
                    try:
                        result = future.result()
                        if result:
                            recommendation_qa_pairs.extend(result)
                    except Exception as e:
                        logging.error(f"生成推荐问答对时出错: {str(e)}")
                
                logging.info(f"推荐问答对生成完成，共 {len(recommendation_qa_pairs)} 条")
                
                # 生成常识问答对
                logging.info("正在生成常识问答对...")
                try:
                    common_sense_qa_pairs = self.generate_common_sense_qa()
                    logging.info(f"常识问答对生成完成，共 {len(common_sense_qa_pairs)} 条")
                except Exception as e:
                    logging.error(f"生成常识问答对时出错: {str(e)}")
                    common_sense_qa_pairs = []
                
                # 生成通用知识问答对
                logging.info("正在生成通用知识问答对...")
                try:
                    general_knowledge_qa_pairs = self.generate_general_knowledge_qa()
                    logging.info(f"通用知识问答对生成完成，共 {len(general_knowledge_qa_pairs)} 条")
                except Exception as e:
                    logging.error(f"生成通用知识问答对时出错: {str(e)}")
                    general_knowledge_qa_pairs = []
            
            # 合并所有问答对
            all_qa_pairs = (
                base_qa_pairs + 
                comparison_qa_pairs + 
                recommendation_qa_pairs +
                common_sense_qa_pairs +
                general_knowledge_qa_pairs
            )
            
            if not all_qa_pairs:
                logging.error("未能生成任何问答对")
                return
            
            # 分离带思维链和不带思维链的数据
            with_chain = [qa for qa in all_qa_pairs if qa.get("chain_of_thought", False)]
            without_chain = [qa for qa in all_qa_pairs if not qa.get("chain_of_thought", False)]
            
            # 保存数据
            logging.info("正在保存数据...")
            try:
                # 根据模式选择输出文件名
                mode_suffix = "_test" if test_mode else ""
                
                with open(os.path.join(output_dir, f"qa_pairs{mode_suffix}_{current_date}.json"), "w", encoding="utf-8") as f:
                    json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
                logging.info(f"成功保存所有问答对到 qa_pairs{mode_suffix}_{current_date}.json")
                
                with open(os.path.join(output_dir, f"with_chain_of_thought{mode_suffix}_{current_date}.json"), "w", encoding="utf-8") as f:
                    json.dump(with_chain, f, ensure_ascii=False, indent=2)
                logging.info(f"成功保存带思维链的问答对到 with_chain_of_thought{mode_suffix}_{current_date}.json")
                
                with open(os.path.join(output_dir, f"without_chain_of_thought{mode_suffix}_{current_date}.json"), "w", encoding="utf-8") as f:
                    json.dump(without_chain, f, ensure_ascii=False, indent=2)
                logging.info(f"成功保存不带思维链的问答对到 without_chain_of_thought{mode_suffix}_{current_date}.json")
            except Exception as e:
                logging.error(f"保存数据时出错: {str(e)}")
                return
            
            # 打印统计信息
            logging.info("\n数据生成完成!")
            logging.info(f"总问答对数量: {len(all_qa_pairs)}")
            logging.info(f"带思维链的问答对: {len(with_chain)}")
            logging.info(f"不带思维链的问答对: {len(without_chain)}")
            logging.info(f"基础信息查询: {len(base_qa_pairs)}")
            logging.info(f"学校比较: {len(comparison_qa_pairs)}")
            logging.info(f"学校推荐: {len(recommendation_qa_pairs)}")
            logging.info(f"常识问答: {len(common_sense_qa_pairs)}")
            logging.info(f"通用知识问答: {len(general_knowledge_qa_pairs)}")
            
        except Exception as e:
            logging.error(f"处理数据时发生错误: {str(e)}")
            raise

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
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"数据已保存到 {output_path}")
        
        # 打印一些示例数据
        if data:
            logger.info("\n示例数据：")
            for example in data[:2]:
                logger.info(json.dumps(example, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"保存数据时发生错误: {str(e)}")
        raise

def generate_enhanced_info(school_info: Dict[str, str]) -> Dict[str, str]:
    """使用在线大模型生成增强的学校信息，但只使用原始数据中的信息"""
    try:
        # 构建提示词，明确要求只使用提供的信息
        prompt = f"""请根据以下学校信息，生成更详细和专业的描述。注意：只能使用提供的信息，不要添加任何不存在的信息。

学校名称：{school_info.get('学校名称', '未知')}
学校类型：{school_info.get('学校类型', '未知')}
地址：{school_info.get('地址', '未知')}
规模：{school_info.get('规模', '未知')}
师资：{school_info.get('师资', '未知')}
设施：{school_info.get('设施', '未知')}
特色：{school_info.get('特色', '未知')}
理念：{school_info.get('理念', '未知')}

请生成以下内容，但只能使用上述信息：
1. 学校介绍（使用已有信息重新组织）
2. 教学特色（使用已有信息）
3. 师资力量（使用已有信息）
4. 校园环境（使用已有信息）
5. 办学理念（使用已有信息）

注意：
1. 只能使用提供的信息，不要添加任何不存在的信息
2. 如果某项信息未知，请直接说明"暂无相关信息"
3. 不要生成任何虚构的细节
4. 不要添加任何没有在原始数据中的描述"""

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
    """以香港择校专家的身份，基于已有信息进行专业分析和推荐"""
    try:
        examples = []
        
        # 1. 基本信息查询
        prompt = f"""作为香港资深择校专家，请根据以下学校信息，结合香港教育体系特点，生成一个专业且详细的学校介绍。注意：只能使用提供的信息，不要添加任何不存在的信息。

{json.dumps(school_info, ensure_ascii=False, indent=2)}

请从以下方面进行分析：
1. 学校基本信息
   - 学校名称和类型
   - 地址和位置
   - 办学性质

2. 办学特色
   - 办学理念
   - 教学特色
   - 课程设置

3. 师资设施
   - 师资力量
   - 教学设施
   - 校园环境

4. 发展情况
   - 学校规模
   - 办学历史
   - 发展现状

请以专业、客观的角度进行分析，但只能使用提供的信息。如果某项信息未知，请直接说明"暂无相关信息"。"""

        llm_instance = online_llm_streaming(prompt)
        response = llm_instance.run()
        examples.append({
            "instruction": "查询学校的基本信息",
            "input": f"请详细介绍{school_info.get('学校名称', '这所学校')}",
            "cot": "1. 提取学校的基本信息\n2. 分析办学特色和理念\n3. 评估师资和设施情况\n4. 总结学校发展现状",
            "output": response
        })
        
        # 2. 学校比较 - 增加更多比较场景
        if len(all_schools) > 1:
            other_school = random.choice([s for s in all_schools if s['学校名称'] != school_info['学校名称']])
            
            # 场景1：同类型学校比较
            prompt = f"""作为香港资深择校专家，请对以下两所同类型学校进行专业比较分析。注意：只能使用提供的信息，不要添加任何不存在的信息。

学校1: {json.dumps(school_info, ensure_ascii=False, indent=2)}
学校2: {json.dumps(other_school, ensure_ascii=False, indent=2)}

请从以下方面进行比较：
1. 基本信息对比
   - 学校类型
   - 办学性质
   - 地理位置

2. 办学特色对比
   - 办学理念
   - 教学特色
   - 课程设置

3. 师资设施对比
   - 师资力量
   - 教学设施
   - 校园环境

4. 发展情况对比
   - 学校规模
   - 办学历史
   - 发展现状

请以专业、客观的角度进行分析，但只能使用提供的信息。如果某项信息未知，请直接说明"暂无相关信息"。"""

            llm_instance = online_llm_streaming(prompt)
            response = llm_instance.run()
            examples.append({
                "instruction": "比较两所学校的特点",
                "input": f"比较{school_info.get('学校名称', '学校1')}和{other_school.get('学校名称', '学校2')}的特点",
                "cot": "1. 对比两校的基本信息\n2. 比较办学特色和理念\n3. 评估师资和设施差异\n4. 总结发展情况对比",
                "output": response
            })
            
            # 场景2：不同类型学校比较
            if school_info.get('学校类型') != other_school.get('学校类型'):
                prompt = f"""作为香港资深择校专家，请对以下两所不同类型学校进行专业比较分析。注意：只能使用提供的信息，不要添加任何不存在的信息。

学校1: {json.dumps(school_info, ensure_ascii=False, indent=2)}
学校2: {json.dumps(other_school, ensure_ascii=False, indent=2)}

请从以下方面进行比较：
1. 学校类型差异
   - 办学层次
   - 教育阶段
   - 培养目标

2. 办学特色差异
   - 办学理念
   - 教学特色
   - 课程设置

3. 师资设施差异
   - 师资力量
   - 教学设施
   - 校园环境

4. 发展情况差异
   - 学校规模
   - 办学历史
   - 发展现状

请以专业、客观的角度进行分析，但只能使用提供的信息。如果某项信息未知，请直接说明"暂无相关信息"。"""

                llm_instance = online_llm_streaming(prompt)
                response = llm_instance.run()
                examples.append({
                    "instruction": "比较两所学校的特点",
                    "input": f"比较{school_info.get('学校名称', '学校1')}和{other_school.get('学校名称', '学校2')}的特点",
                    "cot": "1. 分析学校类型差异\n2. 比较办学特色差异\n3. 评估师资设施差异\n4. 总结发展情况差异",
                "output": response
            })
        
        # 3. 学校推荐 - 增加更多推荐场景
        # 场景1：按学生类型推荐
        student_types = ['理科生', '文科生', '艺术生', '体育生', '国际生', '本地生']
        for student_type in student_types:
            prompt = f"""作为香港资深择校专家，请针对{student_type}的特点，分析该学校的适合度。注意：只能使用提供的信息，不要添加任何不存在的信息。

{json.dumps(school_info, ensure_ascii=False, indent=2)}

请从以下方面进行分析：
1. 基本信息匹配度
   - 学校类型
   - 办学性质
   - 地理位置

2. 课程设置匹配度
   - 课程特色
   - 教学方式
   - 专业方向

3. 师资设施匹配度
   - 师资力量
   - 教学设施
   - 校园环境

4. 发展前景匹配度
   - 升学方向
   - 发展机会
   - 就业前景

请以专业、客观的角度进行分析，但只能使用提供的信息。如果某项信息未知，请直接说明"暂无相关信息"。"""

        llm_instance = online_llm_streaming(prompt)
        response = llm_instance.run()
        examples.append({
            "instruction": "根据条件推荐合适的学校",
            "input": f"推荐适合{student_type}的学校",
                "cot": "1. 分析学生类型特点\n2. 评估学校匹配度\n3. 考虑发展前景\n4. 给出择校建议",
                "output": response
            })
        
        # 场景2：按学习阶段推荐
        study_stages = ['幼儿园', '小学', '初中', '高中']
        for stage in study_stages:
            prompt = f"""作为香港资深择校专家，请针对{stage}阶段的学生，分析该学校的适合度。注意：只能使用提供的信息，不要添加任何不存在的信息。

{json.dumps(school_info, ensure_ascii=False, indent=2)}

请从以下方面进行分析：
1. 教育阶段匹配度
   - 学校类型
   - 办学层次
   - 教育阶段

2. 课程设置匹配度
   - 课程特色
   - 教学方式
   - 学习目标

3. 师资设施匹配度
   - 师资力量
   - 教学设施
   - 校园环境

4. 发展前景匹配度
   - 升学方向
   - 发展机会
   - 未来规划

请以专业、客观的角度进行分析，但只能使用提供的信息。如果某项信息未知，请直接说明"暂无相关信息"。"""

            llm_instance = online_llm_streaming(prompt)
            response = llm_instance.run()
            examples.append({
                "instruction": "根据条件推荐合适的学校",
                "input": f"推荐适合{stage}阶段的学校",
                "cot": "1. 分析教育阶段特点\n2. 评估学校匹配度\n3. 考虑发展前景\n4. 给出择校建议",
            "output": response
        })
        
        return examples
    except Exception as e:
        logger.error(f"生成增强示例失败: {str(e)}")
        return []

def process_data(max_workers: int = 4, limit: int = 20):
    """处理学校数据的主函数
    
    Args:
        max_workers: 最大线程数
        limit: 处理的数据条数限制
    """
    try:
        logger.info("正在加载原始数据(限制%d条)...", limit)
        processor = SchoolDataProcessor()
        
        # 加载原始数据
        raw_data = processor.load_raw_data("data/学校清洗数据.txt", limit=limit)
        logger.info("加载完成，共 %d 条数据", len(raw_data))
        
        # 生成基础问答对
        logger.info("正在生成基础问答对...")
        base_qa_pairs = []
        with tqdm(total=len(raw_data), desc="处理基础问答") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(processor.generate_base_qa, school) for school in raw_data]
                for future in futures:
                    qa_pairs = future.result()
                    base_qa_pairs.extend(qa_pairs)
                    pbar.update(1)
        
        # 生成比较问答对
        logger.info("正在生成比较问答对...")
        comparison_qa_pairs = []
        school_pairs = [(s1, s2) for i, s1 in enumerate(raw_data) for s2 in raw_data[i+1:]]
        with tqdm(total=len(school_pairs), desc="处理比较问答") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(processor.generate_comparison_qa, s1, s2) 
                          for s1, s2 in school_pairs]
                for future in futures:
                    qa_pairs = future.result()
                    comparison_qa_pairs.extend(qa_pairs)
                    pbar.update(1)
        
        # 生成推荐问答对
        logger.info("正在生成推荐问答对...")
        recommendation_qa_pairs = processor.generate_recommendation_qa(raw_data)
        
        # 合并所有问答对
        all_qa_pairs = base_qa_pairs + comparison_qa_pairs + recommendation_qa_pairs
        
        # 分离带思维链和不带思维链的数据
        with_chain = [qa for qa in all_qa_pairs if qa.get("chain_of_thought", False)]
        without_chain = [qa for qa in all_qa_pairs if not qa.get("chain_of_thought", False)]
        
        # 保存处理后的数据
        logger.info("正在保存数据...")
        os.makedirs("data/processed", exist_ok=True)
        
        # 保存所有问答对
        save_processed_data(all_qa_pairs, "data/processed/qa_pairs.json")
        
        # 分别保存带思维链和不带思维链的数据
        save_processed_data(with_chain, "data/processed/with_chain_of_thought.json")
        save_processed_data(without_chain, "data/processed/without_chain_of_thought.json")
        
        # 输出统计信息
        logger.info("\n数据生成完成!")
        logger.info("总问答对数量: %d", len(all_qa_pairs))
        logger.info("带思维链的问答对: %d", len(with_chain))
        logger.info("不带思维链的问答对: %d", len(without_chain))
        logger.info("基础信息查询: %d", len(base_qa_pairs))
        logger.info("学校比较: %d", len(comparison_qa_pairs))
        logger.info("学校推荐: %d", len(recommendation_qa_pairs))
        
    except Exception as e:
        logger.error("处理数据时发生错误: %s", str(e))
        raise

if __name__ == "__main__":
    process_data() 