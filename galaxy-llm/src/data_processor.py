import json
import pandas as pd
from typing import List, Dict, Any

class SchoolDataProcessor:
    def __init__(self, data_file: str):
        """初始化数据处理器
        
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.schools_data = []
        
    def load_data(self) -> None:
        """加载并解析数据文件"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 将文本按学校记录分割
                school_records = content.strip().split('[{')
                for record in school_records:
                    if not record.strip():
                        continue
                    try:
                        # 修复JSON格式
                        record = '[{' + record if not record.startswith('[{') else record
                        record = record.strip()
                        if record.endswith(','):
                            record = record[:-1]
                        # 解析JSON
                        school_data = json.loads(record)
                        self.schools_data.extend(school_data)
                    except json.JSONDecodeError as e:
                        print(f"解析记录时出错: {e}")
                        continue
                        
        except Exception as e:
            print(f"加载数据文件时出错: {e}")
            raise
            
    def extract_school_info(self) -> List[Dict[str, Any]]:
        """提取学校基本信息
        
        Returns:
            包含学校基本信息的字典列表
        """
        schools = []
        for item in self.schools_data:
            school = {}
            for key, value in item.items():
                if isinstance(value, dict) and 'text' in value:
                    # 提取问答中的实际值
                    text = value['text']
                    if '是：' in text:
                        actual_value = text.split('是：')[-1]
                        school[key] = actual_value
                    elif '是否存在？' in text:
                        exists = text.split('是否存在？')[-1]
                        school[f"{key}_exists"] = exists == 'True'
            schools.append(school)
        return schools
        
    def create_dataframe(self) -> pd.DataFrame:
        """创建包含学校信息的DataFrame
        
        Returns:
            pandas DataFrame对象
        """
        schools = self.extract_school_info()
        df = pd.DataFrame(schools)
        return df
        
    def analyze_data(self) -> Dict[str, Any]:
        """分析学校数据
        
        Returns:
            包含分析结果的字典
        """
        df = self.create_dataframe()
        analysis = {
            'total_schools': len(df),
            'districts': df['学校所在区域或地区'].value_counts().to_dict() if '学校所在区域或地区' in df.columns else {},
            'school_types': df['教育阶段'].value_counts().to_dict() if '教育阶段' in df.columns else {},
            'religious_schools': df['学校宗教信仰关联'].value_counts().to_dict() if '学校宗教信仰关联' in df.columns else {},
        }
        return analysis

def main():
    # 使用示例
    processor = SchoolDataProcessor('data/学校清洗数据.txt')
    processor.load_data()
    
    # 创建DataFrame
    df = processor.create_dataframe()
    print("\n数据框架预览:")
    print(df.head())
    
    # 分析数据
    analysis = processor.analyze_data()
    print("\n数据分析结果:")
    for key, value in analysis.items():
        print(f"\n{key}:")
        print(value)

if __name__ == "__main__":
    main() 