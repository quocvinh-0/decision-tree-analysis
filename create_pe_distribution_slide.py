"""
Script để tạo biểu đồ phân phối PE cho slide
Chạy script này để tạo biểu đồ phân phối biến mục tiêu PE
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend non-interactive để tránh lỗi tkinter
import matplotlib.pyplot as plt
import os
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from improved.data_loader_improved import load_and_prepare_data
from visualization import create_pe_distribution_slide

def main():
    """Hàm chính để tạo biểu đồ phân phối PE"""
    
    print("="*70)
    print("TẠO BIỂU ĐỒ PHÂN PHỐI PE CHO SLIDE")
    print("="*70)
    
    # Đọc dữ liệu
    dataset_path = 'Folds5x2_pp.xlsx'
    print(f"\n📂 Đang đọc dữ liệu từ: {dataset_path}")
    X, y = load_and_prepare_data(dataset_path, use_enhanced_features=False)
    
    # Tạo biểu đồ
    print("\n📊 Đang tạo biểu đồ phân phối PE...")
    create_pe_distribution_slide(y)
    
    print("\n✅ Hoàn thành! Biểu đồ đã được lưu tại: img/pe_distribution_slide.png")
    print("="*70)

if __name__ == "__main__":
    main()

