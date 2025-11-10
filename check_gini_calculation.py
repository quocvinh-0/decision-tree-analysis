"""Script để kiểm tra tính toán Gini impurity cho thuộc tính AT"""
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

print("="*70)
print("KIỂM TRA TÍNH TOÁN GINI IMPURITY CHO THUỘC TÍNH AT")
print("="*70)

# Lấy 5 mẫu đầu tiên trong mỗi khoảng (như trong slide)
threshold = 14.80
below = all_data[all_data['AT'] < threshold].head(5)
above = all_data[all_data['AT'] >= threshold].head(5)

print(f"\nNgưỡng AT: {threshold}")
print(f"Tổng số mẫu: {len(below)} + {len(above)} = {len(below) + len(above)}")
print()

# Phân phối cho AT <= 14.80
print("Khoảng AT <= 14.80 (5 mẫu đầu):")
below_counts = below['PE_class'].value_counts()
below_total = len(below)
below_probs = {
    'Thấp': below_counts.get('Thấp', 0) / below_total,
    'Trung bình': below_counts.get('Trung bình', 0) / below_total,
    'Cao': below_counts.get('Cao', 0) / below_total
}

print(f"  Thấp: {below_counts.get('Thấp', 0)}/{below_total} = {below_probs['Thấp']:.4f}")
print(f"  Trung bình: {below_counts.get('Trung bình', 0)}/{below_total} = {below_probs['Trung bình']:.4f}")
print(f"  Cao: {below_counts.get('Cao', 0)}/{below_total} = {below_probs['Cao']:.4f}")

# Tính Gini cho AT <= 14.80
gini_below = 1 - (below_probs['Thấp']**2 + below_probs['Trung bình']**2 + below_probs['Cao']**2)
print(f"\n  Gini(AT <= 14.80) = 1 - [({below_counts.get('Thấp', 0)}/{below_total})² + ({below_counts.get('Trung bình', 0)}/{below_total})² + ({below_counts.get('Cao', 0)}/{below_total})²]")
print(f"                    = 1 - [{below_probs['Thấp']:.4f}² + {below_probs['Trung bình']:.4f}² + {below_probs['Cao']:.4f}²]")
print(f"                    = 1 - [{below_probs['Thấp']**2:.4f} + {below_probs['Trung bình']**2:.4f} + {below_probs['Cao']**2:.4f}]")
print(f"                    = {gini_below:.4f}")

# Phân phối cho AT > 14.80
print("\nKhoảng AT > 14.80 (5 mẫu đầu):")
above_counts = above['PE_class'].value_counts()
above_total = len(above)
above_probs = {
    'Thấp': above_counts.get('Thấp', 0) / above_total,
    'Trung bình': above_counts.get('Trung bình', 0) / above_total,
    'Cao': above_counts.get('Cao', 0) / above_total
}

print(f"  Thấp: {above_counts.get('Thấp', 0)}/{above_total} = {above_probs['Thấp']:.4f}")
print(f"  Trung bình: {above_counts.get('Trung bình', 0)}/{above_total} = {above_probs['Trung bình']:.4f}")
print(f"  Cao: {above_counts.get('Cao', 0)}/{above_total} = {above_probs['Cao']:.4f}")

# Tính Gini cho AT > 14.80
gini_above = 1 - (above_probs['Thấp']**2 + above_probs['Trung bình']**2 + above_probs['Cao']**2)
print(f"\n  Gini(AT > 14.80) = 1 - [({above_counts.get('Thấp', 0)}/{above_total})² + ({above_counts.get('Trung bình', 0)}/{above_total})² + ({above_counts.get('Cao', 0)}/{above_total})²]")
print(f"                   = 1 - [{above_probs['Thấp']:.4f}² + {above_probs['Trung bình']:.4f}² + {above_probs['Cao']:.4f}²]")
print(f"                   = 1 - [{above_probs['Thấp']**2:.4f} + {above_probs['Trung bình']**2:.4f} + {above_probs['Cao']**2:.4f}]")
print(f"                   = {gini_above:.4f}")

# Tính Gini split tổng thể
total_samples = below_total + above_total
weight_below = below_total / total_samples
weight_above = above_total / total_samples
gini_split = weight_below * gini_below + weight_above * gini_above

print(f"\nGini Split cho AT:")
print(f"  Gini(AT) = ({below_total}/{total_samples}) × {gini_below:.4f} + ({above_total}/{total_samples}) × {gini_above:.4f}")
print(f"           = {weight_below:.2f} × {gini_below:.4f} + {weight_above:.2f} × {gini_above:.4f}")
print(f"           = {weight_below * gini_below:.4f} + {weight_above * gini_above:.4f}")
print(f"           = {gini_split:.4f}")

print("\n" + "="*70)
print("SO SÁNH VỚI SLIDE:")
print("="*70)

slide_gini_below = 0.0000
slide_gini_above = 0.3200
slide_gini_split = 0.1600

print(f"\nGini(AT <= 14.80):")
print(f"  Slide: {slide_gini_below:.4f}")
print(f"  Tính toán: {gini_below:.4f}")
print(f"  {'✅ Đúng' if abs(gini_below - slide_gini_below) < 0.0001 else '❌ Sai'}")

print(f"\nGini(AT > 14.80):")
print(f"  Slide: {slide_gini_above:.4f}")
print(f"  Tính toán: {gini_above:.4f}")
print(f"  {'✅ Đúng' if abs(gini_above - slide_gini_above) < 0.0001 else '❌ Sai'}")

print(f"\nGini Split(AT):")
print(f"  Slide: {slide_gini_split:.4f}")
print(f"  Tính toán: {gini_split:.4f}")
print(f"  {'✅ Đúng' if abs(gini_split - slide_gini_split) < 0.0001 else '❌ Sai'}")

