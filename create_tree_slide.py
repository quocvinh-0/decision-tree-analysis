"""Script để tạo cây quyết định từ 10 mẫu đầu (giống slide)"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend non-interactive để tránh lỗi tkinter
import matplotlib.pyplot as plt
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from improved.data_loader_improved import load_and_prepare_data
from visualization import plot_decision_tree_slide

def main():
    print("="*70)
    print("TẠO CÂY QUYẾT ĐỊNH TỪ 10 MẪU ĐẦU (CHO SLIDE)")
    print("="*70)
    
    # Đọc dữ liệu
    dataset_path = 'Folds5x2_pp.xlsx'
    X, y = load_and_prepare_data(dataset_path, use_enhanced_features=False)
    
    # Tạo cây quyết định từ 10 mẫu đầu
    plot_decision_tree_slide(X, y)
    
    print("\n✅ Hoàn thành!")
    print("="*70)
    print("LƯU Ý:")
    print("- img/decision_tree.png: Cây từ mô hình tốt nhất (toàn bộ dataset)")
    print("- img/decision_tree_slide.png: Cây từ 10 mẫu đầu (giống slide)")
    print("="*70)

if __name__ == "__main__":
    main()

