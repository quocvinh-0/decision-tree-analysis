"""Script để kiểm tra phân phối có điều kiện của PE theo các thuộc tính"""
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

# Phân loại PE thành 3 lớp: Thấp, Trung bình, Cao
pe = all_data['PE'].values
pe_q1 = np.percentile(pe, 33.33)
pe_q2 = np.percentile(pe, 66.67)

def classify_pe(value):
    if value < pe_q1:
        return 'Thấp'
    elif value < pe_q2:
        return 'Trung bình'
    else:
        return 'Cao'

all_data['PE_class'] = all_data['PE'].apply(classify_pe)

print("Phân loại PE:")
print(f"  Thấp: < {pe_q1:.2f} MW")
print(f"  Trung bình: {pe_q1:.2f} - {pe_q2:.2f} MW")
print(f"  Cao: >= {pe_q2:.2f} MW")
print()

# Các ngưỡng từ slide
thresholds = {
    'AT': 14.80,
    'V': 44.34,
    'AP': 1014.58,
    'RH': 71.94
}

print("="*70)
print("KIỂM TRA PHÂN PHỐI CÓ ĐIỀU KIỆN")
print("="*70)

for feature, threshold in thresholds.items():
    print(f"\n{feature} (Ngưỡng: {threshold}):")
    print("-"*70)
    
    # Lọc dữ liệu theo khoảng
    below = all_data[all_data[feature] < threshold]
    above = all_data[all_data[feature] >= threshold]
    
    print(f"  Khoảng < {threshold}:")
    if len(below) > 0:
        below_counts = below['PE_class'].value_counts()
        total_below = len(below)
        for cls in ['Thấp', 'Trung bình', 'Cao']:
            count = below_counts.get(cls, 0)
            print(f"    {cls}: {count}/{total_below} ({count/total_below*100:.1f}%)")
    else:
        print("    Không có dữ liệu")
    
    print(f"  Khoảng >= {threshold}:")
    if len(above) > 0:
        above_counts = above['PE_class'].value_counts()
        total_above = len(above)
        for cls in ['Thấp', 'Trung bình', 'Cao']:
            count = above_counts.get(cls, 0)
            print(f"    {cls}: {count}/{total_above} ({count/total_above*100:.1f}%)")
    else:
        print("    Không có dữ liệu")
    
    print(f"  Tổng số mẫu: {len(below)} + {len(above)} = {len(below) + len(above)}")

print("\n" + "="*70)
print("LƯU Ý: Slide hiển thị X/5, có thể là dữ liệu mẫu hoặc từ một nút cụ thể")
print("="*70)

# Kiểm tra xem có thể tìm được 5 mẫu nào khớp với slide không
print("\nTìm các mẫu có thể khớp với slide (lấy 5 mẫu đầu tiên trong mỗi khoảng):")
print("="*70)

for feature, threshold in thresholds.items():
    print(f"\n{feature}:")
    below_sample = all_data[all_data[feature] < threshold].head(5)
    above_sample = all_data[all_data[feature] >= threshold].head(5)
    
    print(f"  Khoảng < {threshold} (5 mẫu đầu):")
    if len(below_sample) > 0:
        below_counts = below_sample['PE_class'].value_counts()
        for cls in ['Thấp', 'Trung bình', 'Cao']:
            count = below_counts.get(cls, 0)
            print(f"    {cls}: {count}/5")
    
    print(f"  Khoảng >= {threshold} (5 mẫu đầu):")
    if len(above_sample) > 0:
        above_counts = above_sample['PE_class'].value_counts()
        for cls in ['Thấp', 'Trung bình', 'Cao']:
            count = above_counts.get(cls, 0)
            print(f"    {cls}: {count}/5")

