import json
import re
import jieba
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

class SchoolDataPreprocessor:
    def __init__(self, input_file: str, output_dir: str):
        """
        初始化数据预处理器
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录路径
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化jieba分词
        jieba.initialize()
        
    def clean_text(self, text: str) -> str:
        """
        清理文本数据
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def generate_cot_reasoning(self, features: Dict[str, Any]) -> str:
        """
        生成 Chain of Thought 推理过程
        
        Args:
            features: 学校特征
            
        Returns:
            CoT 推理文本
        """
        reasoning_steps = []
        
        # 1. 学校基本信息分析
        if 'school_name_cleaned' in features:
            reasoning_steps.append(f"这是一所名为{features['school_name_cleaned']}的学校")
            
        if 'education_stage' in features:
            reasoning_steps.append(f"该学校属于{features['education_stage']}教育阶段")
            
        if 'organization_type' in features:
            reasoning_steps.append(f"学校类型为{features['organization_type']}")
            
        # 2. 地理位置分析
        if 'district' in features:
            reasoning_steps.append(f"学校位于{features['district']}")
            
        if 'address_zh_cleaned' in features:
            reasoning_steps.append(f"具体地址是{features['address_zh_cleaned']}")
            
        # 3. 师资力量分析
        if 'total_teachers_num' in features:
            reasoning_steps.append(f"学校共有{features['total_teachers_num']}名教师")
            
        if 'registered_teachers' in features:
            reasoning_steps.append(f"其中在编教师{features['registered_teachers']}名")
            
        # 4. 办学条件分析
        if 'classroom_count_num' in features:
            reasoning_steps.append(f"学校有{features['classroom_count_num']}间教室")
            
        if 'founding_year_num' in features:
            years = 2024 - features['founding_year_num']
            reasoning_steps.append(f"学校创办于{features['founding_year_num']}年，已有{years}年办学历史")
            
        # 5. 特色分析
        if 'admission_requirements_cleaned' in features:
            reasoning_steps.append(f"入学要求：{features['admission_requirements_cleaned']}")
            
        if 'school_activities_cleaned' in features:
            reasoning_steps.append(f"学校特色活动：{features['school_activities_cleaned']}")
            
        # 6. 综合评估
        reasoning_steps.append("基于以上信息，可以得出以下结论：")
        
        # 根据教师数量评估规模
        if 'total_teachers_num' in features:
            if features['total_teachers_num'] > 50:
                reasoning_steps.append("这是一所规模较大的学校")
            elif features['total_teachers_num'] > 20:
                reasoning_steps.append("这是一所中等规模的学校")
            else:
                reasoning_steps.append("这是一所规模较小的学校")
                
        # 根据建校时间评估历史
        if 'founding_year_num' in features:
            if 2024 - features['founding_year_num'] > 50:
                reasoning_steps.append("这是一所历史悠久的学校")
            elif 2024 - features['founding_year_num'] > 20:
                reasoning_steps.append("这是一所办学历史较长的学校")
            else:
                reasoning_steps.append("这是一所较新的学校")
                
        # 根据活动丰富度评估特色
        if 'school_activities_cleaned' in features:
            activities = features['school_activities_cleaned'].split('、')
            if len(activities) > 5:
                reasoning_steps.append("学校活动丰富，注重学生全面发展")
            elif len(activities) > 2:
                reasoning_steps.append("学校有一定特色活动")
            else:
                reasoning_steps.append("学校活动相对简单")
                
        return '。'.join(reasoning_steps)
    
    def extract_features(self, school_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从学校数据中提取特征
        
        Args:
            school_data: 学校数据
            
        Returns:
            提取的特征字典
        """
        features = {}
        
        # 提取基本信息
        if '学校名称' in school_data:
            features['school_name'] = school_data['学校名称']['text']
            features['school_name_cleaned'] = self.clean_text(features['school_name'])
            
        if '学校英文名称' in school_data:
            features['school_name_en'] = school_data['学校英文名称']['text']
            
        if '学校所在区域或地区' in school_data:
            features['district'] = school_data['学校所在区域或地区']['text']
            
        if '中文地址' in school_data:
            features['address_zh'] = school_data['中文地址']['text']
            features['address_zh_cleaned'] = self.clean_text(features['address_zh'])
            
        if '英文地址' in school_data:
            features['address_en'] = school_data['英文地址']['text']
            
        if '联系电话' in school_data:
            features['phone'] = school_data['联系电话']['text']
            
        if '传真号码' in school_data:
            features['fax'] = school_data['传真号码']['text']
            
        if '电子邮箱' in school_data:
            features['email'] = school_data['电子邮箱']['text']
            
        if '官方网站链接' in school_data:
            features['website'] = school_data['官方网站链接']['text']
            
        if '组织属性类别' in school_data:
            features['organization_type'] = school_data['组织属性类别']['text']
            
        if '组织属性英文类别' in school_data:
            features['organization_type_en'] = school_data['组织属性英文类别']['text']
            
        if '教育阶段' in school_data:
            features['education_stage'] = school_data['教育阶段']['text']
            
        if '学生性别组成' in school_data:
            features['gender_ratio'] = school_data['学生性别组成']['text']
            
        if '学校宗教信谊关联' in school_data:
            features['religion'] = school_data['学校宗教信谊关联']['text']
            
        if '建校年份' in school_data:
            features['founding_year'] = school_data['建校年份']['text']
            
        if '校长' in school_data:
            features['principal'] = school_data['校长']['text']
            
        if '儿童免费计划' in school_data:
            features['free_education'] = school_data['儿童免费计划']['text']
            
        if '校监' in school_data:
            features['supervisor'] = school_data['校监']['text']
            
        if '学校运营状态' in school_data:
            features['operation_status'] = school_data['学校运营状态']['text']
            
        if '总教师数' in school_data:
            features['total_teachers'] = school_data['总教师数']['text']
            
        if '教师在编总数' in school_data:
            features['registered_teachers'] = school_data['教师在编总数']['text']
            
        if '入学条件' in school_data:
            features['admission_requirements'] = school_data['入学条件']['text']
            features['admission_requirements_cleaned'] = self.clean_text(features['admission_requirements'])
            
        if '学校活动' in school_data:
            features['school_activities'] = school_data['学校活动']['text']
            features['school_activities_cleaned'] = self.clean_text(features['school_activities'])
            
        if '教室数量' in school_data:
            features['classroom_count'] = school_data['教室数量']['text']
            
        # 提取数值特征
        if '总教师数' in features:
            numbers = re.findall(r'\d+', features['总教师数'])
            if numbers:
                features['total_teachers_num'] = int(numbers[0])
                
        if '教室数量' in features:
            numbers = re.findall(r'\d+', features['教室数量'])
            if numbers:
                features['classroom_count_num'] = int(numbers[0])
                
        if '建校年份' in features:
            numbers = re.findall(r'\d{4}', features['建校年份'])
            if numbers:
                features['founding_year_num'] = int(numbers[0])
                
        # 生成 CoT 推理
        features['cot_reasoning'] = self.generate_cot_reasoning(features)
        
        return features
    
    def process_school_data(self) -> List[Dict[str, Any]]:
        """
        处理学校数据
        
        Returns:
            处理后的学校数据列表
        """
        processed_data = []
        
        # 读取原始数据
        with open(self.input_file, 'r', encoding='utf-8') as f:
            # 读取所有行
            lines = f.readlines()
            
            # 处理每一行
            for line in lines:
                try:
                    # 解析 JSON
                    school_data = json.loads(line.strip())
                    
                    # 提取特征
                    features = self.extract_features(school_data)
                    
                    # 构建学校数据
                    school_data = {
                        'raw_data': school_data,
                        'features': features
                    }
                    
                    processed_data.append(school_data)
                except json.JSONDecodeError as e:
                    print(f"跳过无效的 JSON 行: {e}")
                    continue
        
        return processed_data
    
    def save_processed_data(self, data: List[Dict[str, Any]]):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据
        """
        # 保存为JSON格式
        output_file = self.output_dir / 'processed_schools.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 保存为CSV格式
        output_file = self.output_dir / 'processed_schools.csv'
        df = pd.DataFrame([item['features'] for item in data])
        df.to_csv(output_file, index=False, encoding='utf-8')
    
    def run(self):
        """
        运行数据预处理流程
        """
        print("开始处理数据...")
        processed_data = self.process_school_data()
        print(f"处理完成，共处理 {len(processed_data)} 条数据")
        
        print("保存处理后的数据...")
        self.save_processed_data(processed_data)
        print("数据保存完成")

if __name__ == '__main__':
    # 设置输入输出路径
    input_file = 'data/学校清洗数据.txt'
    output_dir = 'data/processed'
    
    # 创建预处理器并运行
    preprocessor = SchoolDataPreprocessor(input_file, output_dir)
    preprocessor.run() 