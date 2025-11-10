"""Script để kiểm tra cách chia dữ liệu trong code"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

X = all_data[['AT', 'V', 'AP', 'RH']]
y = all_data['PE']

print("="*70)
print("KIỂM TRA CÁCH CHIA DỮ LIỆU")
print("="*70)

print(f"\nTổng số phần tử: {len(X):,}")

# Kiểm tra cách chia dữ liệu như trong code
print("\nCách chia dữ liệu trong code:")
print("train_test_split(X, y, test_size=0.2, random_state=42+i, shuffle=True)")

# Thử với random_state=42 (lần chạy đầu tiên, i=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nKết quả với random_state=42 (i=0):")
print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# Kiểm tra với random_state=40 (như trong slide)
print(f"\nKết quả với random_state=40 (như trong slide):")
X_train_40, X_test_40, y_train_40, y_test_40 = train_test_split(
    X, y, test_size=0.2, random_state=40, shuffle=True
)
print(f"  Train: {len(X_train_40):,} ({len(X_train_40)/len(X)*100:.1f}%)")
print(f"  Test: {len(X_test_40):,} ({len(X_test_40)/len(X)*100:.1f}%)")

print("\n" + "="*70)
print("KẾT LUẬN:")
print("="*70)
print("✅ Số phần tử và tỷ lệ đều ĐÚNG:")
print(f"   • Tổng: {len(X):,} phần tử")
print(f"   • Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   • Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
print("\n⚠️  Random state trong slide: random_state=40+i")
print("   Random state trong code: random_state=42+i")
print("   → Cần sửa slide thành: random_state=42+i")

