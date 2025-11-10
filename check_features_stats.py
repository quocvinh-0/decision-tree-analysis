"""Script để kiểm tra thống kê các thuộc tính"""
import pandas as pd
import numpy as np
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Load dữ liệu
dataset_path = 'Folds5x2_pp.xlsx'
df = pd.read_excel(dataset_path, sheet_name=None)
all_data = pd.concat(df.values(), ignore_index=True)

# Định nghĩa các thuộc tính và đơn vị
features = {
    'AT': ('°C', 'Nhiệt độ môi trường'),
    'V': ('cmHg', 'Áp suất hơi'),
    'AP': ('mbar', 'Áp suất khí quyển'),
    'RH': ('%', 'Độ ẩm tương đối')
}

print("Thống kê dữ liệu thực tế:")
print("="*60)
print(f"{'STT':<5} {'Thuộc tính':<30} {'Min':<15} {'Max':<15} {'Mean':<15}")
print("-"*60)

for idx, (feat, (unit, desc)) in enumerate(features.items(), 1):
    min_val = all_data[feat].min()
    max_val = all_data[feat].max()
    mean_val = all_data[feat].mean()
    
    print(f"{idx:<5} {feat} - {desc:<20} {min_val:>10.2f} {unit:<4} {max_val:>10.2f} {unit:<4} {mean_val:>10.2f} {unit:<4}")

print("="*60)

# So sánh với slide
print("\nSo sánh với slide:")
print("="*60)
slide_data = {
    'AT': {'min': 1.81, 'max': 37.11, 'mean': 19.65},
    'V': {'min': 25.36, 'max': 81.56, 'mean': 54.31},
    'AP': {'min': 992.89, 'max': 1033.30, 'mean': 1013.26},
    'RH': {'min': 25.56, 'max': 100.16, 'mean': 73.31}
}

for feat in features.keys():
    actual_min = all_data[feat].min()
    actual_max = all_data[feat].max()
    actual_mean = all_data[feat].mean()
    
    slide_min = slide_data[feat]['min']
    slide_max = slide_data[feat]['max']
    slide_mean = slide_data[feat]['mean']
    
    min_match = abs(actual_min - slide_min) < 0.01
    max_match = abs(actual_max - slide_max) < 0.01
    mean_match = abs(actual_mean - slide_mean) < 0.01
    
    print(f"\n{feat}:")
    print(f"  Min:  Thực tế={actual_min:.2f}, Slide={slide_min:.2f}, {'✅ Đúng' if min_match else '❌ Sai'}")
    print(f"  Max:  Thực tế={actual_max:.2f}, Slide={slide_max:.2f}, {'✅ Đúng' if max_match else '❌ Sai'}")
    print(f"  Mean: Thực tế={actual_mean:.2f}, Slide={slide_mean:.2f}, {'✅ Đúng' if mean_match else '❌ Sai'}")

