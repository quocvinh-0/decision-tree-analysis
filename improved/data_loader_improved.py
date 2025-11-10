"""
PHIÊN BẢN CẢI THIỆN CỦA DATA_LOADER.PY

Các cải thiện chính:
1. Loại bỏ scaling không cần thiết cho Decision Tree
2. Chỉ trả về X và y, không có X_scaled
3. Scaling chỉ được thực hiện khi cần (ví dụ: cho KNN)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

def load_and_prepare_data(file_path, use_enhanced_features=False, return_scaler=False):
    """
    Đọc và chuẩn bị dữ liệu từ file Excel
    
    Parameters:
    - file_path: đường dẫn đến file dữ liệu
    - use_enhanced_features: có sử dụng feature engineering không
    - return_scaler: có trả về scaler không (chỉ cần khi dùng cho KNN)
    
    Returns:
    - X: features (KHÔNG được scale - Decision Tree không cần)
    - y: target
    - scaler: (optional) StandardScaler nếu return_scaler=True
    """
    try:
        xls = pd.ExcelFile(file_path)
        df_list = []
        # Lặp qua tên của từng sheet
        print(f"\n📂 Bắt đầu đọc dữ liệu từ file: {file_path}")
        print(f"📋 Tổng số sheets: {len(xls.sheet_names)}")
        for idx, sheet_name in enumerate(xls.sheet_names, 1):
            print(f"   [{idx}/{len(xls.sheet_names)}] Đang đọc sheet: {sheet_name}...")
            sys.stdout.flush()
            df_list.append(pd.read_excel(xls, sheet_name=sheet_name))
        
        # Gộp tất cả các DataFrame từ các sheet lại
        df = pd.concat(df_list, ignore_index=True)
        
        print(f"✅ Đã đọc và gộp {len(xls.sheet_names)} sheets thành công!")
        sys.stdout.flush()
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file '{file_path}'.")
        print("Vui lòng đảm bảo file dữ liệu nằm cùng thư mục với script.")
        exit()
    
    print(f"\n📊 THÔNG TIN DATASET (SAU KHI GỘP):")
    print(f"   • Kích thước: {df.shape} ({df.shape[0]:,} mẫu × {df.shape[1]} đặc trưng)")
    print(f"   • Cột: {list(df.columns)}")
    print(f"   • Kiểm tra NaN (dữ liệu thiếu): {df.isna().sum().sum()} (Nếu > 0 là có lỗi)")
    
    # Phân tích và tiền xử lý
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    
    # Feature engineering (tùy chọn)
    if use_enhanced_features:
        X = create_enhanced_features(X)
        print("   ✅ Đã sử dụng feature engineering")
    else:
        print("   ℹ️  Sử dụng feature gốc")
    
    # LƯU Ý: Decision Tree KHÔNG cần scaling
    # Chỉ tạo scaler nếu được yêu cầu (ví dụ: cho KNN)
    scaler = None
    if return_scaler:
        scaler = StandardScaler()
        # Chỉ fit, không transform (vì Decision Tree không cần)
        scaler.fit(X)
        print("   ℹ️  Scaler đã được tạo (chỉ dùng cho KNN nếu cần)")
    else:
        print("   ℹ️  Không tạo scaler (Decision Tree không cần scaling)")
    
    if return_scaler:
        return X, y, scaler
    else:
        return X, y

def create_enhanced_features(X):
    """
    Tạo các feature mới từ feature gốc
    """
    print("\n🔧 FEATURE ENGINEERING NÂNG CAO")
    
    X_enhanced = X.copy()
    X_enhanced['AT_V'] = X['AT'] * X['V']           # Tương tác nhiệt độ và áp suất hơi
    X_enhanced['AT_RH'] = X['AT'] * X['RH']         # Tương tác nhiệt độ và độ ẩm
    X_enhanced['V_AP'] = X['V'] * X['AP']           # Tương tác áp suất hơi và áp suất khí
    X_enhanced['AT_squared'] = X['AT'] ** 2         # Đa thức bậc 2 cho nhiệt độ
    X_enhanced['V_squared'] = X['V'] ** 2           # Đa thức bậc 2 cho áp suất hơi
    
    print(f"   • Số feature ban đầu: {X.shape[1]}")
    print(f"   • Số feature sau engineering: {X_enhanced.shape[1]}")
    print(f"   • Feature mới: {list(X_enhanced.columns)[X.shape[1]:]}")
    
    return X_enhanced

def get_scaled_features(X, scaler=None):
    """
    Hàm tiện ích để scale features khi cần (ví dụ: cho KNN)
    
    Parameters:
    - X: features gốc
    - scaler: StandardScaler đã được fit (nếu None, sẽ tạo mới)
    
    Returns:
    - X_scaled: features đã được scale
    - scaler: scaler đã được fit
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler

