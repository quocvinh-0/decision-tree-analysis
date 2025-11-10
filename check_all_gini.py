"""Script để kiểm tra tính toán Gini impurity cho tất cả các thuộc tính"""
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

def calculate_gini(counts_dict, total):
    """Tính Gini impurity từ dictionary counts"""
    probs = {
        'Thấp': counts_dict.get('Thấp', 0) / total,
        'Trung bình': counts_dict.get('Trung bình', 0) / total,
        'Cao': counts_dict.get('Cao', 0) / total
    }
    gini = 1 - (probs['Thấp']**2 + probs['Trung bình']**2 + probs['Cao']**2)
    return gini, probs

def calculate_gini_split(feature, threshold):
    """Tính Gini split cho một thuộc tính"""
    # Lấy 5 mẫu đầu tiên trong mỗi khoảng (như trong slide)
    below = all_data[all_data[feature] < threshold].head(5)
    above = all_data[all_data[feature] >= threshold].head(5)
    
    below_counts = below['PE_class'].value_counts().to_dict()
    above_counts = above['PE_class'].value_counts().to_dict()
    
    below_total = len(below)
    above_total = len(above)
    total = below_total + above_total
    
    gini_below, probs_below = calculate_gini(below_counts, below_total)
    gini_above, probs_above = calculate_gini(above_counts, above_total)
    
    weight_below = below_total / total
    weight_above = above_total / total
    gini_split = weight_below * gini_below + weight_above * gini_above
    
    return {
        'below': {'counts': below_counts, 'total': below_total, 'probs': probs_below, 'gini': gini_below},
        'above': {'counts': above_counts, 'total': above_total, 'probs': probs_above, 'gini': gini_above},
        'gini_split': gini_split,
        'total': total
    }

print("="*70)
print("KIỂM TRA TÍNH TOÁN GINI CHO TẤT CẢ CÁC THUỘC TÍNH")
print("="*70)

# Các thuộc tính và ngưỡng từ slide
attributes = {
    'AT': {'threshold': 14.80, 'name': 'Nhiệt độ môi trường', 'slide_gini': 0.1600},
    'V': {'threshold': 44.34, 'name': 'Áp suất hơi', 'slide_gini': 0.4000},
    'AP': {'threshold': 1014.58, 'name': 'Áp suất khí quyển', 'slide_gini': 0.4800},
    'RH': {'threshold': 71.94, 'name': 'Độ ẩm tương đối', 'slide_gini': 0.4800}
}

results = {}

for feat, info in attributes.items():
    threshold = info['threshold']
    slide_gini = info['slide_gini']
    
    print(f"\n{'='*70}")
    print(f"Tính Gini cho {feat} ({info['name']})")
    print(f"{'='*70}")
    
    result = calculate_gini_split(feat, threshold)
    results[feat] = result
    
    # Hiển thị phân phối
    print(f"\nNgưỡng: {threshold}")
    print(f"\nKhoảng < {threshold}:")
    below_counts = result['below']['counts']
    below_total = result['below']['total']
    print(f"  Thấp: {below_counts.get('Thấp', 0)}/{below_total}")
    print(f"  Trung bình: {below_counts.get('Trung bình', 0)}/{below_total}")
    print(f"  Cao: {below_counts.get('Cao', 0)}/{below_total}")
    
    print(f"\nKhoảng >= {threshold}:")
    above_counts = result['above']['counts']
    above_total = result['above']['total']
    print(f"  Thấp: {above_counts.get('Thấp', 0)}/{above_total}")
    print(f"  Trung bình: {above_counts.get('Trung bình', 0)}/{above_total}")
    print(f"  Cao: {above_counts.get('Cao', 0)}/{above_total}")
    
    # Tính toán Gini
    gini_below = result['below']['gini']
    gini_above = result['above']['gini']
    gini_split = result['gini_split']
    
    print(f"\nGini(< {threshold}) = {gini_below:.4f}")
    print(f"Gini(>= {threshold}) = {gini_above:.4f}")
    print(f"Gini({feat}) = ({below_total}/{below_total + above_total}) × {gini_below:.4f} + ({above_total}/{below_total + above_total}) × {gini_above:.4f}")
    print(f"           = {gini_split:.4f}")
    
    # So sánh với slide
    match = abs(gini_split - slide_gini) < 0.0001
    print(f"\nSo sánh với slide:")
    print(f"  Slide: {slide_gini:.4f}")
    print(f"  Tính toán: {gini_split:.4f}")
    print(f"  {'✅ Đúng' if match else '❌ Sai'}")

# Tổng kết
print(f"\n{'='*70}")
print("TỔNG KẾT")
print(f"{'='*70}")
print(f"{'Thuộc tính':<10} {'Slide Gini':<15} {'Tính toán':<15} {'Kết quả':<10}")
print("-"*70)

all_correct = True
for feat, info in attributes.items():
    slide_gini = info['slide_gini']
    calculated_gini = results[feat]['gini_split']
    match = abs(calculated_gini - slide_gini) < 0.0001
    if not match:
        all_correct = False
    print(f"{feat:<10} {slide_gini:<15.4f} {calculated_gini:<15.4f} {'✅ Đúng' if match else '❌ Sai'}")

print("-"*70)
if all_correct:
    print("✅ TẤT CẢ CÁC TÍNH TOÁN ĐỀU ĐÚNG!")
else:
    print("❌ CÓ MỘT SỐ TÍNH TOÁN SAI!")

