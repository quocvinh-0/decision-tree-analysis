import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path, use_enhanced_features=False):
   # Đọc dữ liệu từ file Excel và tiền xử lý
    try:
        xls = pd.ExcelFile(file_path)
        df_list = []
        # Lặp qua tên của từng sheet
        for sheet_name in xls.sheet_names:
            print(f"Đang đọc sheet: {sheet_name}...")
            df_list.append(pd.read_excel(xls, sheet_name=sheet_name))
        
        # Gộp tất cả các DataFrame từ các sheet lại
        df = pd.concat(df_list, ignore_index=True)
        
        print(f" Đã đọc và gộp {len(xls.sheet_names)} sheets thành công!")
    except FileNotFoundError:
        print(f" LỖI: Không tìm thấy file '{file_path}'.")
        print("Vui lòng đảm bảo file dữ liệu nằm cùng thư mục với script.")
        exit()
    
    print(f"\n THÔNG TIN DATASET (SAU KHI GỘP):")
    print(f"    . Kích thước: {df.shape} ({df.shape[0]:,} mẫu × {df.shape[1]} đặc trưng)")
    print(f"     Cột: {list(df.columns)}")
    print(f"     Kiểm tra NaN (dữ liệu thiếu): {df.isna().sum().sum()} (Nếu > 0 là có lỗi)")
    
    # Phân tích và tiền xử lý
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    
    # Feature engineering (tùy chọn)
    if use_enhanced_features:
        X = create_enhanced_features(X)
        print("     Đã sử dụng feature engineering")
    else:
        print("     Sử dụng feature gốc (để so sánh công bằng)")
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, y, X_scaled

def create_enhanced_features(X):
    
    #Tạo các feature mới từ feature gốc
    X_enhanced = X.copy()
    X_enhanced['AT_V'] = X['AT'] * X['V']           # Tương tác nhiệt độ và áp suất hơi
    X_enhanced['AT_RH'] = X['AT'] * X['RH']         # Tương tác nhiệt độ và độ ẩm
    X_enhanced['V_AP'] = X['V'] * X['AP']           # Tương tác áp suất hơi và áp suất khí
    X_enhanced['AT_squared'] = X['AT'] ** 2         # Đa thức bậc 2 cho nhiệt độ
    X_enhanced['V_squared'] = X['V'] ** 2           # Đa thức bậc 2 cho áp suất hơi
    
    print(f"    Số feature ban đầu: {X.shape[1]}")
    print(f"    Số feature sau engineering: {X_enhanced.shape[1]}")
    print(f"    Feature mới: {list(X_enhanced.columns)[X.shape[1]:]}")
    
    return X_enhanced