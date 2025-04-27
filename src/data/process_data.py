import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from .data_processor import EducationDataProcessor
from .utils import DataUtils
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def validate_data(data: Dict) -> bool:
    """验证处理后的数据"""
    required_fields = {
        'school_info': ['name', 'type', 'location'],
        'curriculum': ['subject', 'level', 'description'],
        'admission': ['year', 'requirements', 'deadline', 'process']
    }
    
    if not data:
        logging.error("数据为空，无法进行验证")
        return False
        
    for school_id, school_data in data.items():
        # 验证学校信息
        for field in required_fields['school_info']:
            if field not in school_data['school_info']:
                logging.error(f"学校 {school_id} 缺少必要字段: {field}")
                return False
                
        # 验证课程信息
        for course in school_data['curriculum']:
            for field in required_fields['curriculum']:
                if field not in course:
                    logging.error(f"学校 {school_id} 的课程缺少必要字段: {field}")
                    return False
                    
        # 验证招生信息
        for admission in school_data['admission']:
            for field in required_fields['admission']:
                if field not in admission:
                    logging.error(f"学校 {school_id} 的招生信息缺少必要字段: {field}")
                    return False
                    
    return True

def process_education_data():
    """处理教育数据的主函数"""
    try:
        # 初始化处理器
        data_dir = Path("data")
        if not data_dir.exists():
            logging.error(f"数据目录不存在: {data_dir.absolute()}")
            return None
            
        processor = EducationDataProcessor(data_dir=str(data_dir))
        
        # 加载数据
        logging.info("正在加载数据...")
        raw_data_path = data_dir / "学校清洗数据.txt"
        if not raw_data_path.exists():
            logging.error(f"原始数据文件不存在: {raw_data_path}")
            return None
            
        processor.load_data()
        logging.info(f"成功加载数据文件: {raw_data_path}")
        
        if not processor.raw_data:
            logging.error("加载的数据为空")
            return None
            
        # 处理数据
        logging.info("正在处理数据...")
        processed_data = processor.process_data()
        
        if not processed_data:
            logging.error("数据处理后结果为空")
            return None
            
        # 验证数据
        logging.info("正在验证数据...")
        if not validate_data(processed_data):
            logging.error("数据验证失败")
            return None
            
        # 保存处理后的数据
        logging.info("正在保存处理后的数据...")
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        processor.save_processed_data(str(output_dir))
        logging.info(f"数据已保存到: {output_dir}")
        
        # 数据统计
        logging.info("\n数据统计：")
        logging.info(f"学校数量：{len(processed_data)}")
        
        # 学校类型分布
        type_counts = {}
        for school_id, data in processed_data.items():
            school_type = data['school_info']['type']
            type_counts[school_type] = type_counts.get(school_type, 0) + 1
        
        logging.info("\n学校类型分布：")
        for school_type, count in type_counts.items():
            logging.info(f"{school_type}: {count}所")
        
        # 课程统计
        total_courses = sum(len(data['curriculum']) for data in processed_data.values())
        logging.info(f"\n总课程数：{total_courses}")
        
        # 招生信息统计
        total_admissions = sum(len(data['admission']) for data in processed_data.values())
        logging.info(f"总招生信息数：{total_admissions}")
        
        return processed_data
        
    except Exception as e:
        logging.error(f"数据处理过程中发生错误: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    process_education_data() 