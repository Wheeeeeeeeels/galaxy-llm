from typing import Dict, List, Tuple
import re
import logging
from datetime import datetime

class DataValidator:
    def __init__(self):
        self.required_fields = {
            'school_info': ['name', 'type', 'location'],
            'curriculum': ['subject', 'level', 'description'],
            'admission': ['year', 'requirements', 'deadline', 'process']
        }
        
    def validate_school_info(self, school_info: Dict) -> Tuple[bool, List[str]]:
        """验证学校信息"""
        errors = []
        
        # 检查必填字段
        for field in self.required_fields['school_info']:
            if field not in school_info or not school_info[field]:
                errors.append(f"缺少必填字段: {field}")
                
        # 验证学校名称
        if 'name' in school_info:
            if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9\s]+$', school_info['name']):
                errors.append("学校名称包含非法字符")
                
        # 验证学校类型
        valid_types = ['小学', '中学', '高中', '大学', '职业学校', '特殊教育学校']
        if 'type' in school_info and school_info['type'] not in valid_types:
            errors.append(f"无效的学校类型: {school_info['type']}")
            
        # 验证地理位置
        if 'location' in school_info:
            if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9\s]+$', school_info['location']):
                errors.append("地理位置包含非法字符")
                
        return len(errors) == 0, errors
        
    def validate_curriculum(self, curriculum: List[Dict]) -> Tuple[bool, List[str]]:
        """验证课程信息"""
        errors = []
        
        if not curriculum:
            errors.append("课程信息为空")
            return False, errors
            
        for i, course in enumerate(curriculum):
            # 检查必填字段
            for field in self.required_fields['curriculum']:
                if field not in course or not course[field]:
                    errors.append(f"课程 {i+1} 缺少必填字段: {field}")
                    
            # 验证课程级别
            valid_levels = ['初级', '中级', '高级', '基础', '进阶']
            if 'level' in course and course['level'] not in valid_levels:
                errors.append(f"课程 {i+1} 无效的级别: {course['level']}")
                
        return len(errors) == 0, errors
        
    def validate_admission(self, admission: List[Dict]) -> Tuple[bool, List[str]]:
        """验证招生信息"""
        errors = []
        
        if not admission:
            errors.append("招生信息为空")
            return False, errors
            
        for i, info in enumerate(admission):
            # 检查必填字段
            for field in self.required_fields['admission']:
                if field not in info or not info[field]:
                    errors.append(f"招生信息 {i+1} 缺少必填字段: {field}")
                    
            # 验证年份
            if 'year' in info:
                try:
                    year = int(info['year'])
                    current_year = datetime.now().year
                    if not (2000 <= year <= current_year + 1):
                        errors.append(f"招生信息 {i+1} 无效的年份: {year}")
                except ValueError:
                    errors.append(f"招生信息 {i+1} 年份格式错误: {info['year']}")
                    
            # 验证截止日期
            if 'deadline' in info:
                try:
                    datetime.strptime(info['deadline'], '%Y-%m-%d')
                except ValueError:
                    errors.append(f"招生信息 {i+1} 截止日期格式错误: {info['deadline']}")
                    
        return len(errors) == 0, errors
        
    def validate_school_data(self, school_data: Dict) -> Tuple[bool, List[str]]:
        """验证完整的学校数据"""
        errors = []
        
        # 验证学校信息
        school_info_valid, school_info_errors = self.validate_school_info(school_data.get('school_info', {}))
        errors.extend(school_info_errors)
        
        # 验证课程信息
        curriculum_valid, curriculum_errors = self.validate_curriculum(school_data.get('curriculum', []))
        errors.extend(curriculum_errors)
        
        # 验证招生信息
        admission_valid, admission_errors = self.validate_admission(school_data.get('admission', []))
        errors.extend(admission_errors)
        
        return len(errors) == 0, errors
        
    def validate_dataset(self, dataset: List[Dict]) -> Tuple[bool, Dict[int, List[str]]]:
        """验证整个数据集"""
        errors = {}
        all_valid = True
        
        for i, school_data in enumerate(dataset):
            valid, school_errors = self.validate_school_data(school_data)
            if not valid:
                all_valid = False
                errors[i] = school_errors
                
        return all_valid, errors 