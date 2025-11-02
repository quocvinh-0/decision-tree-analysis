import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_validate
from model_trainer import calculate_metrics

def compare_with_other_models(X_train_best, X_test_best, y_train_best, y_test_best, best_model):
    """
    So sánh Decision Tree với các mô hình khác
    
    Returns:
    - comparison_results: dictionary chứa kết quả so sánh
    """
    comparison_results = {}
    
    # Tính metrics cho Decision Tree tốt nhất
    y_pred_dt_best = best_model.predict(X_test_best)
    dt_metrics_best = calculate_metrics(y_test_best, y_pred_dt_best)
    comparison_results['decision_tree'] = {
        'metrics': dt_metrics_best,
        'predictions': y_pred_dt_best
    }
    
    # SO SÁNH VỚI RANDOM FOREST
    print("\nSO SÁNH VỚI RANDOM FOREST")
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10,
        min_samples_split=10, n_jobs=-1
    )
    rf_model.fit(X_train_best, y_train_best)
    y_pred_rf = rf_model.predict(X_test_best)
    rf_metrics = calculate_metrics(y_test_best, y_pred_rf)
    comparison_results['random_forest'] = {
        'metrics': rf_metrics,
        'predictions': y_pred_rf,
        'model': rf_model
    }
    
    # SO SÁNH VỚI KNN
    print("\n🔍 SO SÁNH THÊM VỚI KNN REGRESSOR (TỐI ƯU HÓA THAM SỐ)")
    knn_metrics, best_knn = train_optimized_knn(X_train_best, X_test_best, y_train_best, y_test_best)
    comparison_results['knn'] = {
        'metrics': knn_metrics,
        'predictions': knn_metrics.get('predictions'),
        'model': best_knn
    }
    
    # Cross-validation cho mô hình tốt nhất
    print("\n🔄 ĐÁNH GIÁ ĐỘ ỔN ĐỊNH VỚI CROSS-VALIDATION (5-fold)")
    cv_results = perform_cross_validation(best_model, X_train_best, y_train_best)
    comparison_results['cv_results'] = cv_results
    
    # In kết quả so sánh
    print_comparison_results(dt_metrics_best, rf_metrics, knn_metrics, cv_results)
    
    return comparison_results

def train_optimized_knn(X_train, X_test, y_train, y_test):
    """Huấn luyện KNN với tối ưu hóa tham số"""
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    knn_grid = GridSearchCV(
        KNeighborsRegressor(), knn_param_grid, cv=5, 
        scoring='r2', n_jobs=-1, verbose=0
    )
    
    print("Đang tìm tham số tối ưu cho KNN...")
    knn_grid.fit(X_train, y_train)
    
    best_knn = knn_grid.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    knn_metrics = calculate_metrics(y_test, y_pred_knn)
    knn_metrics['predictions'] = y_pred_knn
    
    print(f"\n✅ KNN Regressor (ĐÃ TỐI ƯU):")
    print(f"    Tham số tốt nhất: {knn_grid.best_params_}")
    print(f"    R²:   {knn_metrics['r2']:.4f}")
    print(f"    RMSE: {knn_metrics['rmse']:.4f}")
    
    return knn_metrics, best_knn

def perform_cross_validation(model, X, y):
    """Thực hiện cross-validation"""
    cv_results = cross_validate(
        model, X, y, 
        cv=5, 
        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
        return_train_score=True,
        n_jobs=-1
    )
    
    cv_train_r2 = cv_results['train_r2']
    cv_test_r2 = cv_results['test_r2']
    cv_test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])
    cv_test_mae = -cv_results['test_neg_mean_absolute_error']
    
    return {
        'train_r2': cv_train_r2,
        'test_r2': cv_test_r2,
        'test_rmse': cv_test_rmse,
        'test_mae': cv_test_mae
    }

def print_comparison_results(dt_metrics, rf_metrics, knn_metrics, cv_results):
    """In kết quả so sánh các mô hình"""
    print("\n SO SÁNH HIỆU SUẤT TRÊN TẬP TEST TỐT NHẤT:")
    print(f"    Decision Tree (tốt nhất):")
    print(f"       R²:   {dt_metrics['r2']:.4f}")
    print(f"       RMSE: {dt_metrics['rmse']:.4f}")
    print(f"       MAE:  {dt_metrics['mae']:.4f}")
    print(f"       MAPE: {dt_metrics['mape']:.2f}%")
    
    print(f"    Random Forest:")
    print(f"       R²:   {rf_metrics['r2']:.4f}")
    print(f"       RMSE: {rf_metrics['rmse']:.4f}")
    print(f"       MAE:  {rf_metrics['mae']:.4f}")
    print(f"       MAPE: {rf_metrics['mape']:.2f}%")
    
    print(f"    KNN (tối ưu):")
    print(f"       R²:   {knn_metrics['r2']:.4f}")
    print(f"       RMSE: {knn_metrics['rmse']:.4f}")
    print(f"       MAE:  {knn_metrics['mae']:.4f}")
    print(f"       MAPE: {knn_metrics['mape']:.2f}%")
    
    print(f"\n📊 KẾT QUẢ CROSS-VALIDATION (5-fold):")
    print(f"    Train R²:     {cv_results['train_r2'].mean():.4f} (±{cv_results['train_r2'].std():.4f})")
    print(f"    Test R²:      {cv_results['test_r2'].mean():.4f} (±{cv_results['test_r2'].std():.4f})")
    print(f"    Test RMSE:    {cv_results['test_rmse'].mean():.4f} (±{cv_results['test_rmse'].std():.4f})")
    
    cv_stability = "RẤT ỔN ĐỊNH" if cv_results['test_r2'].std() < 0.02 else "KHÁ ỔN ĐỊNH" if cv_results['test_r2'].std() < 0.05 else "CÓ BIẾN ĐỘNG"
    print(f"    Độ ổn định:    {cv_stability} (độ lệch chuẩn: {cv_results['test_r2'].std():.4f})")