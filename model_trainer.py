"""Decision Tree training utilities aligned với THB3_Decision_Tree.pdf."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    max_error,
    r2_score,
    explained_variance_score,
)
from sklearn.model_selection import cross_val_score

def calculate_metrics(y_true, y_pred):
    """
    Tính các metrics đánh giá mô hình
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    max_err = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Xử lý trường hợp y_true = 0 để tránh lỗi chia cho 0
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-10))) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'medae': medae,
        'max_error': max_err,
        'r2': r2,
        'explained_variance': explained_var,
        'mape': mape,
    }

def find_optimal_ccp_alpha(X_train, y_train, base_params):
    """Tìm giá trị ccp_alpha tối ưu dựa trên 5-fold CV (không đụng tới test set)."""
    candidate_alphas = np.logspace(-4, -1, 20)
    best_alpha = 0.0
    best_score = -np.inf

    for alpha in candidate_alphas:
        dt = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42, **base_params)
        scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='r2')
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    return best_alpha


def train_decision_trees(X, y, X_scaled=None, n_runs=10, use_grid_search=True):
    """Huấn luyện Decision Tree đúng chuẩn THB3_Decision_Tree (không chuẩn hóa đặc trưng)."""

    _ = X_scaled  # giữ tham số để tương thích, Decision Tree không dùng dữ liệu đã chuẩn hóa

    print("\n==============================================")
    print("HUẤN LUYỆN DECISION TREE THEO THB3_Decision_Tree.pdf")
    print("==============================================")
    print("- Không chuẩn hóa đầu vào (Decision Tree không nhạy thang đo)")
    print("- Sử dụng GridSearchCV + ccp_alpha pruning để chống overfitting")

    train_records = []
    test_records = []
    feature_importances = []
    best_models = []

    param_grid = {
        'max_depth': [5, 7, 9, 12, 15, 18],
        'min_samples_split': [5, 8, 12, 15, 20],
        'min_samples_leaf': [2, 3, 4, 5, 8],
        'max_features': ['sqrt', 'log2', None]
    }

    for run_idx in range(n_runs):
        print(f"\nLẦN CHẠY THỨ {run_idx + 1}/{n_runs}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42 + run_idx, shuffle=True
        )
        print(f"   Kích thước: Train={len(X_train)}, Test={len(X_test)}")

        if use_grid_search:
            base_model = DecisionTreeRegressor(random_state=42 + run_idx, splitter='best')
            grid = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)
            params = grid.best_params_
            params['splitter'] = 'best'
            print(f"   Tham số từ GridSearchCV: {params}")
        else:
            params = {
                'max_depth': 9,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'splitter': 'best'
            }
            print(f"   Tham số mặc định: {params}")

        if 'ccp_alpha' not in params:
            alpha = find_optimal_ccp_alpha(X_train, y_train, params)
            params['ccp_alpha'] = alpha
            if alpha > 0:
                print(f"   Pruning với ccp_alpha={alpha:.5f}")
            else:
                print("   Không cần pruning (ccp_alpha = 0)")

        dt_model = DecisionTreeRegressor(random_state=42 + run_idx, **params)
        dt_model.fit(X_train, y_train)

        y_pred_train = dt_model.predict(X_train)
        y_pred_test = dt_model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        train_records.append(train_metrics)
        test_records.append(test_metrics)
        feature_importances.append(dt_model.feature_importances_)

        best_models.append({
            'model': dt_model,
            'params': params,
            'test_r2': test_metrics['r2'],
            'run_id': run_idx,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'scaler': None,
        })

        print(f"   Train R² = {train_metrics['r2']:.4f} | Test R² = {test_metrics['r2']:.4f}")
        print(f"   Test RMSE = {test_metrics['rmse']:.4f} | Test MAE = {test_metrics['mae']:.4f}")
        print(f"   Test Median AE = {test_metrics['medae']:.4f} | Test Max Error = {test_metrics['max_error']:.4f}")
        print(f"   Test MAPE = {test_metrics['mape']:.2f}% | Explained variance = {test_metrics['explained_variance']:.4f}")

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    if hasattr(X, 'columns'):
        feature_names = list(X.columns)
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    feature_importance_df = pd.DataFrame({
        'Đặc trưng': feature_names,
        'Độ quan trọng trung bình': np.mean(feature_importances, axis=0),
        'Độ lệch chuẩn': np.std(feature_importances, axis=0)
    }).sort_values('Độ quan trọng trung bình', ascending=False)

    best_models.sort(key=lambda info: info['test_r2'], reverse=True)
    best_model_info = best_models[0]

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
    print(f"     Median AE: {train_df['medae'].mean():.4f} (±{train_df['medae'].std():.4f})")
    print(f"     Max Error: {train_df['max_error'].mean():.4f} (±{train_df['max_error'].std():.4f})")
    print(f"     MAPE:   {train_df['mape'].mean():.2f}% (±{train_df['mape'].std():.2f}%)")
    print(f"     Explained Variance: {train_df['explained_variance'].mean():.4f} (±{train_df['explained_variance'].std():.4f})")
    
    print("\n THỐNG KÊ TẬP TEST (10 lần):")
    print(f"     R²:     {test_df['r2'].mean():.4f} (±{test_df['r2'].std():.4f})")
    print(f"     RMSE:   {test_df['rmse'].mean():.4f} (±{test_df['rmse'].std():.4f})")
    print(f"     MAE:    {test_df['mae'].mean():.4f} (±{test_df['mae'].std():.4f})")
    print(f"     Median AE: {test_df['medae'].mean():.4f} (±{test_df['medae'].std():.4f})")
    print(f"     Max Error: {test_df['max_error'].mean():.4f} (±{test_df['max_error'].std():.4f})")
    print(f"     MAPE:   {test_df['mape'].mean():.2f}% (±{test_df['mape'].std():.2f}%)")
    print(f"     Explained Variance: {test_df['explained_variance'].mean():.4f} (±{test_df['explained_variance'].std():.4f})")
    
    print("\n ĐỘ QUAN TRỌNG ĐẶC TRƯNG TRUNG BÌNH:")
    for idx, row in feature_importance_df.iterrows():
        print(f"    - {row['Đặc trưng']}: {row['Độ quan trọng trung bình']:.4f} (±{row['Độ lệch chuẩn']:.4f})")