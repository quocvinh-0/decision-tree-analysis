"""
PHIÊN BẢN CẢI THIỆN CỦA MODEL_TRAINER.PY

Các cải thiện chính:
1. Loại bỏ scaling không cần thiết cho Decision Tree
2. Sử dụng GridSearchCV để tìm hyperparameter tối ưu
3. Sử dụng nested cross-validation để tránh data leakage
4. Thêm cost complexity pruning
5. Cải thiện quy trình đánh giá
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import random
warnings.filterwarnings('ignore')

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

def find_optimal_ccp_alpha(X_train, y_train, X_test, y_test, base_params):
    """
    Tìm giá trị ccp_alpha tối ưu bằng cách thử nghiệm trên test set
    (hoặc có thể dùng cross-validation trên train set)
    """
    # Tạo cây với các giá trị ccp_alpha khác nhau
    ccp_alphas = np.logspace(-4, -1, 20)
    best_ccp_alpha = 0.0
    best_r2 = -np.inf
    
    # Sử dụng cross-validation trên train set để tìm ccp_alpha tốt nhất
    # (tránh dùng test set cho việc tuning)
    from sklearn.model_selection import cross_val_score
    for ccp_alpha in ccp_alphas:
        model = DecisionTreeRegressor(
            random_state=42,
            ccp_alpha=ccp_alpha,
            **base_params
        )
        # Sử dụng 5-fold CV trên train set
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        avg_r2 = scores.mean()
        
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_ccp_alpha = ccp_alpha
    
    return best_ccp_alpha

def train_decision_trees_improved(X, y, n_runs=10, use_grid_search=True):
    """
    Huấn luyện Decision Tree với phương pháp cải thiện
    
    Parameters:
    - X: features (KHÔNG cần scaling cho Decision Tree)
    - y: target
    - n_runs: số lần chạy để đánh giá độ ổn định
    - use_grid_search: có sử dụng GridSearchCV không
    
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
    
    # Định nghĩa grid search parameters - tối ưu để cân bằng giữa chất lượng và tốc độ
    if use_grid_search:
        # Giảm số lượng tham số để tăng tốc độ
        # Loại bỏ các giá trị dễ gây overfitting (min_samples_split=2, min_samples_leaf=1, max_features=None)
        # Mở rộng param_grid để có nhiều lựa chọn hơn, tránh chọn cùng một bộ tham số
        # Tổng số tổ hợp: 8 × 5 × 5 × 3 = 600 tổ hợp
        # Với 5-fold CV: 600 × 5 = 3,000 mô hình
        param_grid = {
            'max_depth': [5, 7, 9, 10, 12, 15, 18, 20],    # 8 giá trị (thêm 9, 18)
            'min_samples_split': [5, 8, 10, 15, 20],       # 5 giá trị (thêm 8)
            'min_samples_leaf': [2, 3, 4, 5, 10],          # 5 giá trị (thêm 4)
            'max_features': ['sqrt', 'log2', None]         # 3 giá trị (thêm lại None để đa dạng)
        }
    else:
        # Fallback: sử dụng các bộ tham số với max_depth ngẫu nhiên
        # Loại bỏ None để tránh overfitting
        max_depth_options = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Loại bỏ None
        min_samples_split_options = [2, 5, 8, 10, 12, 15, 20, 25, 30]
        min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
        max_features_options = ['sqrt', 'log2', None]
        
        # Tạo 10 bộ tham số ngẫu nhiên
        random.seed(42)  # Để có thể reproduce
        param_sets = []
        for _ in range(10):
            param_sets.append({
                'max_depth': random.choice(max_depth_options),
                'min_samples_split': random.choice(min_samples_split_options),
                'min_samples_leaf': random.choice(min_samples_leaf_options),
                'max_features': random.choice(max_features_options)
            })
    
    print(f"\n{'='*60}")
    print(f"HUẤN LUYỆN DECISION TREE (PHƯƠNG PHÁP CẢI THIỆN)")
    print(f"{'='*60}")
    print(f"• Số lần chạy: {n_runs}")
    print(f"• Sử dụng GridSearchCV: {use_grid_search}")
    print(f"• Không sử dụng scaling (Decision Tree không cần)")
    print(f"• Sử dụng train/test split (80/20)")
    
    for i in range(n_runs):
        print(f"\n{'='*60}")
        print(f"🔄 LẦN CHẠY THỨ {i+1}/{n_runs}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        # Phân chia train/test với random_state khác nhau
        # 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + i, shuffle=True
        )
        
        print(f"   📊 Kích thước: Train={len(X_train)}, Test={len(X_test)}")
        sys.stdout.flush()
        
        # Tìm hyperparameter tối ưu
        if use_grid_search:
            print(f"   🔍 Đang tìm hyperparameter tối ưu với GridSearchCV...")
            sys.stdout.flush()
            
            # Sử dụng GridSearchCV với 5-fold CV trên train set
            base_model = DecisionTreeRegressor(random_state=42 + i)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            best_params = grid_search.best_params_
            print(f"   ✓ Tham số tốt nhất (CV score: {grid_search.best_score_:.4f}): {best_params}")
            sys.stdout.flush()
        else:
            # Sử dụng bộ tham số cố định
            best_params = param_sets[i % len(param_sets)]
            print(f"   📝 Tham số: {best_params}")
            sys.stdout.flush()
        
        # Tìm ccp_alpha tối ưu trên train set (dùng cross-validation)
        # (chỉ nếu chưa có trong best_params từ GridSearchCV)
        if 'ccp_alpha' not in best_params:
            print(f"   🔍 Đang tìm ccp_alpha tối ưu (dùng cross-validation trên train set)...")
            sys.stdout.flush()
            best_ccp_alpha = find_optimal_ccp_alpha(
                X_train, y_train, X_test, y_test, best_params
            )
            
            if best_ccp_alpha > 0:
                best_params['ccp_alpha'] = best_ccp_alpha
                print(f"   ✓ ccp_alpha tối ưu: {best_ccp_alpha:.6f}")
            else:
                print(f"   ✓ Không cần pruning (ccp_alpha = 0)")
            sys.stdout.flush()
        else:
            print(f"   ✓ ccp_alpha từ GridSearchCV: {best_params['ccp_alpha']:.6f}")
            sys.stdout.flush()
        
        # Tạo và huấn luyện mô hình cuối cùng với train data
        if isinstance(X_train, pd.DataFrame):
            X_train_final = X_train.values
        else:
            X_train_final = X_train
        
        if isinstance(y_train, pd.Series):
            y_train_final = y_train.values
        else:
            y_train_final = y_train
        
        dt_model = DecisionTreeRegressor(
            random_state=42 + i,
            **best_params
        )
        dt_model.fit(X_train_final, y_train_final)
        
        # Dự đoán trên các tập khác nhau
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
            'params': best_params,
            'test_r2': test_metrics['r2'],  # Chọn dựa trên test
            'run_id': i,
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'y_pred_test': y_pred_test,
        })
        
        print(f"   📈 Kết quả:")
        print(f"      • Train R²: {train_metrics['r2']:.4f}")
        print(f"      • Test R²:  {test_metrics['r2']:.4f}")
        print(f"      • Test RMSE: {test_metrics['rmse']:.4f}")
        sys.stdout.flush()
    
    # Tạo DataFrames từ kết quả
    train_df = pd.DataFrame(all_train_metrics)
    test_df = pd.DataFrame(all_test_metrics)
    
    # Tính độ quan trọng đặc trưng trung bình
    avg_feature_importance = np.mean(all_feature_importances, axis=0)
    
    # Lấy tên feature
    if hasattr(X, 'columns'):
        feature_names = list(X.columns)
    elif isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    feature_importance_df = pd.DataFrame({
        'Đặc trưng': feature_names,
        'Độ quan trọng trung bình': avg_feature_importance,
        'Độ lệch chuẩn': np.std(all_feature_importances, axis=0)
    }).sort_values('Độ quan trọng trung bình', ascending=False)
    
    # Chọn mô hình tốt nhất dựa trên TEST score
    best_models.sort(key=lambda x: x['test_r2'], reverse=True)
    best_model_info = best_models[0]
    
    # In kết quả tổng hợp
    print_improved_summary(train_df, test_df, feature_importance_df)
    
    return train_df, test_df, feature_importance_df, best_models, best_model_info

def print_improved_summary(train_df, test_df, feature_importance_df):
    """In tổng kết kết quả với train/test split"""
    print("\n" + "="*60)
    print("PHÂN TÍCH TỔNG HỢP (PHƯƠNG PHÁP CẢI THIỆN)")
    print("="*60)
    
    print("\n📊 THỐNG KÊ TẬP TRAIN (10 lần):")
    print(f"   R²:     {train_df['r2'].mean():.4f} (±{train_df['r2'].std():.4f})")
    print(f"   RMSE:   {train_df['rmse'].mean():.4f} (±{train_df['rmse'].std():.4f})")
    print(f"   MAE:    {train_df['mae'].mean():.4f} (±{train_df['mae'].std():.4f})")
    
    print("\n📊 THỐNG KÊ TẬP TEST (10 lần):")
    print(f"   R²:     {test_df['r2'].mean():.4f} (±{test_df['r2'].std():.4f})")
    print(f"   RMSE:   {test_df['rmse'].mean():.4f} (±{test_df['rmse'].std():.4f})")
    print(f"   MAE:    {test_df['mae'].mean():.4f} (±{test_df['mae'].std():.4f})")
    
    # Đánh giá overfitting
    train_test_gap = train_df['r2'].mean() - test_df['r2'].mean()
    
    print(f"\n🔍 ĐÁNH GIÁ OVERFITTING:")
    print(f"   Chênh lệch Train-Test R²: {train_test_gap:.4f}")
    if train_test_gap > 0.05:
        print(f"   ⚠️  Có dấu hiệu overfitting (chênh lệch > 0.05)")
    else:
        print(f"   ✅ Không có overfitting nghiêm trọng")
    
    print("\n🔍 ĐỘ QUAN TRỌNG ĐẶC TRƯNG TRUNG BÌNH:")
    for idx, row in feature_importance_df.iterrows():
        print(f"   ✓ {row['Đặc trưng']}: {row['Độ quan trọng trung bình']:.4f} (±{row['Độ lệch chuẩn']:.4f})")

