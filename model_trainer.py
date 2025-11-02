import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Tính các metrics đánh giá mô hình
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Xử lý trường hợp y_true = 0 để tránh lỗi chia cho 0
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10))) * 100
    
    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 
        'r2': r2, 'mape': mape
    }

def train_decision_trees(X, y, X_scaled, n_runs=10):
    """
    Huấn luyện nhiều mô hình Decision Tree với các tham số khác nhau
    
    Parameters:
    - X: features gốc
    - y: target
    - X_scaled: features đã chuẩn hóa
    - n_runs: số lần huấn luyện
    
    Returns:
    - train_df: DataFrame chứa metrics tập train
    - test_df: DataFrame chứa metrics tập test  
    - feature_importance_df: DataFrame chứa độ quan trọng đặc trưng
    - best_models: danh sách các mô hình tốt nhất
    - best_model_info: thông tin mô hình tốt nhất
    """
    # Lists để lưu kết quả
    all_train_metrics = []
    all_test_metrics = []
    all_feature_importances = []
    best_models = []
    
    # Định nghĩa các bộ tham số
    param_sets = [
        {'max_depth': 5, 'min_samples_split': 20, 'min_samples_leaf': 10},
        {'max_depth': 7, 'min_samples_split': 15, 'min_samples_leaf': 5},
        {'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 3},
        {'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 8},
        {'max_depth': 12, 'min_samples_split': 8, 'min_samples_leaf': 4},
        {'max_depth': 6, 'min_samples_split': 25, 'min_samples_leaf': 12},
        {'max_depth': 9, 'min_samples_split': 12, 'min_samples_leaf': 6},
        {'max_depth': 4, 'min_samples_split': 30, 'min_samples_leaf': 15}
    ]
    
    for i in range(n_runs):
        print(f"\n🔄 LẦN CHẠY THỨ {i+1}/{n_runs}")
        
        # Chuẩn hóa dữ liệu cho mỗi lần chạy
        scaler = StandardScaler()
        X_scaled_run = scaler.fit_transform(X)
        
        # Phân chia train-test với random_state khác nhau
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_run, y, test_size=0.2, random_state=40 + i, shuffle=True
        )
        
        # Lấy bộ tham số cho lần chạy này
        params = param_sets[i]
        print(f"     Tham số: {params}")
        
        # Tạo và huấn luyện mô hình
        dt_model = DecisionTreeRegressor(
            random_state=40 + i,
            **params
        )
        
        dt_model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred_train = dt_model.predict(X_train)
        y_pred_test = dt_model.predict(X_test)
        
        # Tính metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)
        
        # Lưu kết quả
        all_train_metrics.append(train_metrics)
        all_test_metrics.append(test_metrics)
        all_feature_importances.append(dt_model.feature_importances_)
        
        best_models.append({
            'model': dt_model,
            'params': params,
            'test_r2': test_metrics['r2'],
            'run_id': i,
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'y_pred_test': y_pred_test,
            'scaler': scaler
        })
        
        print(f"    ✓ Train R²: {train_metrics['r2']:.4f}")
        print(f"    ✓ Test R²:  {test_metrics['r2']:.4f}")
        print(f"    ✓ Test RMSE: {test_metrics['rmse']:.4f}")
    
    # Tạo DataFrames từ kết quả
    train_df = pd.DataFrame(all_train_metrics)
    test_df = pd.DataFrame(all_test_metrics)
    
    # Tính độ quan trọng đặc trưng trung bình
    avg_feature_importance = np.mean(all_feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'Đặc trưng': list(X.columns),
        'Độ quan trọng trung bình': avg_feature_importance,
        'Độ lệch chuẩn': np.std(all_feature_importances, axis=0)
    }).sort_values('Độ quan trọng trung bình', ascending=False)
    
    # Chọn mô hình tốt nhất
    best_models.sort(key=lambda x: x['test_r2'], reverse=True)
    best_model_info = best_models[0]
    
    # In kết quả tổng hợp
    print_10_runs_summary(train_df, test_df, feature_importance_df)
    
    return train_df, test_df, feature_importance_df, best_models, best_model_info

def print_10_runs_summary(train_df, test_df, feature_importance_df):
    """In tổng kết kết quả 10 lần chạy"""
    print("\n" + "="*50)
    print("PHÂN TÍCH TỔNG HỢP 10 LẦN CHẠY")
    print("="*50)
    
    print("\n THỐNG KÊ TẬP TRAIN (10 lần):")
    print(f"     R²:     {train_df['r2'].mean():.4f} (±{train_df['r2'].std():.4f})")
    print(f"     RMSE:   {train_df['rmse'].mean():.4f} (±{train_df['rmse'].std():.4f})")
    print(f"     MAE:    {train_df['mae'].mean():.4f} (±{train_df['mae'].std():.4f})")
    print(f"     MAPE:   {train_df['mape'].mean():.2f}% (±{train_df['mape'].std():.2f}%)")
    
    print("\n THỐNG KÊ TẬP TEST (10 lần):")
    print(f"     R²:     {test_df['r2'].mean():.4f} (±{test_df['r2'].std():.4f})")
    print(f"     RMSE:   {test_df['rmse'].mean():.4f} (±{test_df['rmse'].std():.4f})")
    print(f"     MAE:    {test_df['mae'].mean():.4f} (±{test_df['mae'].std():.4f})")
    print(f"     MAPE:   {test_df['mape'].mean():.2f}% (±{test_df['mape'].std():.2f}%)")
    
    print("\n ĐỘ QUAN TRỌNG ĐẶC TRƯNG TRUNG BÌNH:")
    for idx, row in feature_importance_df.iterrows():
        print(f"    ✓ {row['Đặc trưng']}: {row['Độ quan trọng trung bình']:.4f} (±{row['Độ lệch chuẩn']:.4f})")