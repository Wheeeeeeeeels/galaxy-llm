import json
import pandas as pd
from typing import Dict, List, Union, Tuple
from pathlib import Path
import re
import logging
import chardet
from .augmentation import DataAugmentor
from .validator import DataValidator

class EducationDataProcessor:
    def __init__(self, data_dir: str = "data", use_cache: bool = True):
        self.data_dir = Path(data_dir)
        self.raw_data = None
        self.schools_data = None
        self.curriculum_data = None
        self.admission_data = None
        self.use_cache = use_cache
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据增强器和验证器
        self.augmentor = DataAugmentor()
        self.validator = DataValidator()
        
    def detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            logging.info(f"检测到文件编码: {encoding}, 置信度: {result['confidence']}")
            return encoding
            
    def load_data(self) -> None:
        """加载所有教育数据"""
        # 检查缓存
        if self.use_cache:
            cache_file = self.cache_dir / "raw_data.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        self.raw_data = json.load(f)
                    logging.info("从缓存加载数据成功")
                    return
                except Exception as e:
                    logging.warning(f"缓存加载失败: {e}")
        
        # 加载原始数据
        raw_data_path = self.data_dir / "学校清洗数据.txt"
        if raw_data_path.exists():
            # 检测文件编码
            encoding = self.detect_encoding(raw_data_path)
            
            # 尝试不同的编码方式读取
            encodings_to_try = [encoding, 'utf-8', 'gbk', 'gb2312', 'big5']
            for enc in encodings_to_try:
                try:
                    with open(raw_data_path, 'r', encoding=enc) as f:
                        self.raw_data = f.readlines()
                        # 检查第一行是否包含中文字符
                        if any('\u4e00' <= char <= '\u9fff' for char in self.raw_data[0]):
                            logging.info(f"成功使用 {enc} 编码读取数据")
                            break
                except UnicodeDecodeError:
                    continue
                    
            if self.raw_data:
                logging.info(f"成功加载 {len(self.raw_data)} 行数据")
                # 打印第一行数据用于检查
                logging.info(f"第一行数据: {self.raw_data[0][:200]}")
                
                # 保存到缓存
                if self.use_cache:
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(self.raw_data, f, ensure_ascii=False)
                        logging.info("数据已缓存")
                    except Exception as e:
                        logging.warning(f"缓存保存失败: {e}")
            else:
                logging.error("无法使用任何编码方式读取数据")
                
    def parse_school_info(self, text: str) -> Dict:
        """解析学校信息"""
        info = {}
        
        # 提取学校名称
        name_match = re.search(r'学校名称[:：]\s*(.*?)(?:\n|$)', text)
        if name_match:
            info['name'] = name_match.group(1).strip()
        else:
            logging.warning("未找到学校名称")
            
        # 提取学校类型
        type_match = re.search(r'学校类型[:：]\s*(.*?)(?:\n|$)', text)
        if type_match:
            info['type'] = type_match.group(1).strip()
        else:
            logging.warning("未找到学校类型")
            
        # 提取地理位置
        location_match = re.search(r'地理位置[:：]\s*(.*?)(?:\n|$)', text)
        if location_match:
            info['location'] = location_match.group(1).strip()
        else:
            logging.warning("未找到地理位置")
            
        # 提取特色
        features_match = re.search(r'特色[:：]\s*(.*?)(?:\n|$)', text)
        if features_match:
            features = features_match.group(1).strip()
            info['features'] = [f.strip() for f in features.split('、') if f.strip()]
            
        # 提取设施
        facilities_match = re.search(r'设施[:：]\s*(.*?)(?:\n|$)', text)
        if facilities_match:
            facilities = facilities_match.group(1).strip()
            info['facilities'] = [f.strip() for f in facilities.split('、') if f.strip()]
            
        return info
        
    def parse_curriculum_info(self, text: str) -> List[Dict]:
        """解析课程信息"""
        curriculum = []
        # 查找课程信息部分
        curriculum_section = re.search(r'课程信息[:：]\s*(.*?)(?=招生信息|$)', text, re.DOTALL)
        if curriculum_section:
            courses = curriculum_section.group(1).strip().split('\n')
            for course in courses:
                if not course.strip():
                    continue
                # 解析课程信息
                course_match = re.search(r'-\s*(.*?)\s*\((.*?)\):\s*(.*)', course)
                if course_match:
                    curriculum.append({
                        'subject': course_match.group(1).strip(),
                        'level': course_match.group(2).strip(),
                        'description': course_match.group(3).strip()
                    })
                else:
                    logging.warning(f"无法解析课程信息: {course}")
        return curriculum
        
    def parse_admission_info(self, text: str) -> List[Dict]:
        """解析招生信息"""
        admission = []
        # 查找招生信息部分
        admission_section = re.search(r'招生信息[:：]\s*(.*?)$', text, re.DOTALL)
        if admission_section:
            admissions = admission_section.group(1).strip().split('\n')
            current_year = None
            current_info = {}
            
            for line in admissions:
                if not line.strip():
                    continue
                # 解析年份
                year_match = re.search(r'- (\d{4})年[:：]', line)
                if year_match:
                    if current_year and current_info:
                        admission.append(current_info)
                    current_year = year_match.group(1)
                    current_info = {'year': current_year}
                # 解析要求
                elif '要求：' in line:
                    current_info['requirements'] = line.split('要求：')[1].strip()
                # 解析截止日期
                elif '截止日期：' in line:
                    current_info['deadline'] = line.split('截止日期：')[1].strip()
                # 解析流程
                elif '流程：' in line:
                    current_info['process'] = line.split('流程：')[1].strip()
                    
            if current_year and current_info:
                admission.append(current_info)
                
        return admission
        
    def process_data(self) -> Dict:
        """处理原始数据"""
        if not self.raw_data:
            logging.error("原始数据为空")
            return {}
            
        processed_data = {}
        current_text = ""
        school_count = 0
        
        for line in self.raw_data:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是新的学校信息开始
            if line.startswith('学校名称：'):
                if current_text:
                    # 处理上一个学校的数据
                    school_count += 1
                    school_info = self.parse_school_info(current_text)
                    if school_info.get('name'):  # 只处理有效的学校信息
                        school_data = {
                            'school_info': school_info,
                            'curriculum': self.parse_curriculum_info(current_text),
                            'admission': self.parse_admission_info(current_text)
                        }
                        
                        # 验证数据
                        valid, errors = self.validator.validate_school_data(school_data)
                        if valid:
                            processed_data[school_count] = school_data
                            logging.info(f"成功处理学校: {school_info.get('name')}")
                        else:
                            logging.warning(f"学校数据验证失败: {school_info.get('name')}")
                            logging.warning(f"错误信息: {errors}")
                current_text = line + '\n'
            else:
                current_text += line + '\n'
                
        # 处理最后一个学校
        if current_text:
            school_count += 1
            school_info = self.parse_school_info(current_text)
            if school_info.get('name'):  # 只处理有效的学校信息
                school_data = {
                    'school_info': school_info,
                    'curriculum': self.parse_curriculum_info(current_text),
                    'admission': self.parse_admission_info(current_text)
                }
                
                # 验证数据
                valid, errors = self.validator.validate_school_data(school_data)
                if valid:
                    processed_data[school_count] = school_data
                    logging.info(f"成功处理最后一个学校: {school_info.get('name')}")
                else:
                    logging.warning(f"学校数据验证失败: {school_info.get('name')}")
                    logging.warning(f"错误信息: {errors}")
            
        logging.info(f"总共处理了 {len(processed_data)} 所学校的数据")
        return processed_data
        
    def save_processed_data(self, output_path: str = "data/processed") -> None:
        """保存处理后的数据"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data = self.process_data()
        if not processed_data:
            logging.error("没有数据可保存")
            return
            
        # 保存原始处理后的数据
        output_file = output_dir / "processed_schools.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        logging.info(f"原始数据已保存到: {output_file}")
        
        # 生成增强数据
        augmented_data = {}
        for school_id, school_data in processed_data.items():
            augmented_schools = self.augmentor.augment_school_data(school_data)
            for i, aug_school in enumerate(augmented_schools):
                augmented_data[f"{school_id}_{i}"] = aug_school
                
        # 保存增强后的数据
        aug_output_file = output_dir / "augmented_schools.json"
        with open(aug_output_file, 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        logging.info(f"增强数据已保存到: {aug_output_file}")
            
    def get_school_recommendations(self, criteria: Dict) -> List[Dict]:
        """根据条件推荐学校"""
        processed_data = self.process_data()
        if not processed_data:
            logging.warning("没有可用的学校数据")
            return []
            
        recommendations = []
        for school_id, data in processed_data.items():
            score = 0
            school_info = data['school_info']
            
            # 根据条件计算匹配分数
            if criteria.get('type') == school_info.get('type'):
                score += 2
            if criteria.get('location') in school_info.get('location'):
                score += 1
            if any(feature in school_info.get('features', []) for feature in criteria.get('features', [])):
                score += 1
                
            if score > 0:
                recommendations.append({
                    'school': data,
                    'score': score
                })
                
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        logging.info(f"找到 {len(recommendations)} 个匹配的学校")
        return recommendations 