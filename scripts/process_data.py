import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.process_data import process_education_data

if __name__ == "__main__":
    print("开始处理教育数据...")
    processed_data = process_education_data()
    print("数据处理完成！") 