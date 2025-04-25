import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any
from augmentation import DataAugmentor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self, schools_data: List[Dict]):
        self.schools_data = schools_data
        self.augmentor = DataAugmentor()
        
    def generate_info_query_data(self, with_cot: bool = False) -> List[Dict]:
        """生成学校信息咨询的训练数据"""
        query_data = []
        for school in self.schools_data:
            school_name = school["学校名称"]
            
            # 为每个字段生成问答对
            for field, value in school.items():
                if field == "学校名称":
                    continue
                    
                # 使用模板生成多个问题变体
                if field in self.augmentor.question_templates:
                    for q_template in self.augmentor.question_templates[field]:
                        for a_template in self.augmentor.answer_templates[field]:
                            if with_cot:
                                # 带思维链的回答
                                reasoning = f"要回答这个问题，我需要：\n1. 理解问题在询问{school_name}的{field}\n2. 查找{school_name}的相关信息\n3. 提取{field}的具体内容"
                                output = f"{reasoning}\n\n最终答案是：{a_template.format(school_name, value)}"
                            else:
                                # 直接回答
                                output = a_template.format(school_name, value)
                                
                            query = {
                                "instruction": "请回答关于学校信息的问题。" + ("请详细说明你的思考过程。" if with_cot else ""),
                                "input": q_template.format(school_name),
                                "output": output
                            }
                            query_data.append(query)
                            
        return query_data
        
    def generate_school_comparison_data(self, with_cot: bool = False) -> List[Dict]:
        """生成学校比较的训练数据"""
        comparison_data = []
        comparison_fields = ["教育阶段", "组织属性类别", "学生性别组成", "学校宗教信仰关联"]
        
        for i in range(len(self.schools_data)):
            school1 = self.schools_data[i]
            # 随机选择另一所学校进行比较
            school2 = random.choice(self.schools_data[:i] + self.schools_data[i+1:])
            
            # 生成比较问题
            for field in comparison_fields:
                if field in school1 and field in school2:
                    question = f"请比较{school1['学校名称']}和{school2['学校名称']}在{field}方面的异同。"
                    
                    if with_cot:
                        # 带思维链的回答
                        reasoning = f"""要比较这两所学校，我需要：
1. 分别查看{school1['学校名称']}和{school2['学校名称']}的{field}
2. 分析两所学校在{field}方面的相同点和不同点
3. 总结比较结果"""
                        answer = f"{reasoning}\n\n比较结果：\n{school1['学校名称']}的{field}是{school1[field]}，而{school2['学校名称']}的{field}是{school2[field]}。"
                    else:
                        # 直接回答
                        answer = f"{school1['学校名称']}的{field}是{school1[field]}，而{school2['学校名称']}的{field}是{school2[field]}。"
                    
                    comparison = {
                        "instruction": "请比较两所学校的特点。" + ("请详细说明你的思考过程。" if with_cot else ""),
                        "input": question,
                        "output": answer
                    }
                    comparison_data.append(comparison)
                    
        return comparison_data
        
    def generate_school_recommendation_data(self, with_cot: bool = False) -> List[Dict]:
        """生成学校推荐的训练数据"""
        recommendation_data = []
        
        # 按不同维度对学校进行分组
        schools_by_district = {}
        schools_by_type = {}
        
        for school in self.schools_data:
            district = school.get("学校所在区域或地区", "未知")
            school_type = school.get("教育阶段", "未知")
            
            schools_by_district.setdefault(district, []).append(school)
            schools_by_type.setdefault(school_type, []).append(school)
            
        # 生成地区相关的推荐
        for district, schools in schools_by_district.items():
            if len(schools) >= 2:
                question = f"请推荐{district}的学校。"
                recommended_schools = random.sample(schools, min(3, len(schools)))
                
                if with_cot:
                    # 带思维链的回答
                    reasoning = f"""要推荐{district}的学校，我需要：
1. 了解{district}有哪些学校
2. 分析这些学校的特点和优势
3. 根据学校的教育阶段、组织属性等特征进行筛选
4. 选择最具代表性的几所学校进行推荐"""
                    answer = f"{reasoning}\n\n推荐结果：\n在{district}，我推荐以下学校：\n" + "\n".join(
                        f"- {school['学校名称']}：{school.get('教育阶段', '未知')}，{school.get('组织属性类别', '未知')}"
                        for school in recommended_schools
                    )
                else:
                    # 直接回答
                    answer = f"在{district}，我推荐以下学校：\n" + "\n".join(
                        f"- {school['学校名称']}：{school.get('教育阶段', '未知')}，{school.get('组织属性类别', '未知')}"
                        for school in recommended_schools
                    )
                
                recommendation = {
                    "instruction": "请根据地区推荐合适的学校。" + ("请详细说明你的思考过程。" if with_cot else ""),
                    "input": question,
                    "output": answer
                }
                recommendation_data.append(recommendation)
                
        # 生成教育阶段相关的推荐
        for school_type, schools in schools_by_type.items():
            if len(schools) >= 2:
                question = f"请推荐一些{school_type}。"
                recommended_schools = random.sample(schools, min(3, len(schools)))
                
                if with_cot:
                    # 带思维链的回答
                    reasoning = f"""要推荐{school_type}，我需要：
1. 了解所有{school_type}的信息
2. 分析这些学校的特点和优势
3. 根据学校的地理位置、组织属性等特征进行筛选
4. 选择最具代表性的几所学校进行推荐"""
                    answer = f"{reasoning}\n\n推荐结果：\n以下是推荐的{school_type}：\n" + "\n".join(
                        f"- {school['学校名称']}：位于{school.get('学校所在区域或地区', '未知')}，{school.get('组织属性类别', '未知')}"
                        for school in recommended_schools
                    )
                else:
                    # 直接回答
                    answer = f"以下是推荐的{school_type}：\n" + "\n".join(
                        f"- {school['学校名称']}：位于{school.get('学校所在区域或地区', '未知')}，{school.get('组织属性类别', '未知')}"
                        for school in recommended_schools
                    )
                
                recommendation = {
                    "instruction": "请根据教育阶段推荐合适的学校。" + ("请详细说明你的思考过程。" if with_cot else ""),
                    "input": question,
                    "output": answer
                }
                recommendation_data.append(recommendation)
                
        return recommendation_data
        
    def generate_all_training_data(self, with_cot: bool = False) -> Dict[str, List[Dict]]:
        """生成所有类型的训练数据"""
        info_query_data = self.generate_info_query_data(with_cot)
        comparison_data = self.generate_school_comparison_data(with_cot)
        recommendation_data = self.generate_school_recommendation_data(with_cot)
        
        logger.info(f"生成了 {len(info_query_data)} 条信息咨询数据")
        logger.info(f"生成了 {len(comparison_data)} 条学校比较数据")
        logger.info(f"生成了 {len(recommendation_data)} 条学校推荐数据")
        
        return {
            "info_query": info_query_data,
            "comparison": comparison_data,
            "recommendation": recommendation_data
        }

def save_training_data(data: Dict[str, List[Dict]], output_dir: Path, with_cot: bool = False) -> None:
    """保存训练数据"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存各类型数据
    for data_type, items in data.items():
        output_file = output_dir / f"{data_type}_{'cot' if with_cot else 'direct'}.json"
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"保存了 {len(items)} 条{data_type}数据到 {output_file}")
        
    # 保存合并后的数据
    all_data = []
    for items in data.values():
        all_data.extend(items)
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 划分训练集和验证集
    train_size = int(len(all_data) * 0.9)
    train_data = all_data[:train_size]
    eval_data = all_data[train_size:]
    
    # 保存训练集和验证集
    with (output_dir / f"train_{'cot' if with_cot else 'direct'}.json").open('w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with (output_dir / f"eval_{'cot' if with_cot else 'direct'}.json").open('w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"保存了 {len(train_data)} 条训练数据和 {len(eval_data)} 条验证数据")

def main():
    # 加载处理后的学校数据
    schools_file = Path('data/processed/schools.json')
    with schools_file.open('r', encoding='utf-8') as f:
        schools_data = json.load(f)
        
    # 创建训练数据生成器
    generator = TrainingDataGenerator(schools_data)
    
    # 生成带思维链的训练数据
    logger.info("生成带思维链的训练数据...")
    cot_training_data = generator.generate_all_training_data(with_cot=True)
    save_training_data(cot_training_data, Path('data/training/cot'), with_cot=True)
    
    # 生成不带思维链的训练数据
    logger.info("生成不带思维链的训练数据...")
    direct_training_data = generator.generate_all_training_data(with_cot=False)
    save_training_data(direct_training_data, Path('data/training/direct'), with_cot=False)

if __name__ == '__main__':
    main() 