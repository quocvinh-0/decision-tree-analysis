import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from model_trainer import calculate_metrics

def compare_with_other_models(
    X_train_best,
    X_test_best,
    y_train_best,
    y_test_best,
    best_model,
    X_train_scaled=None,
    X_test_scaled=None,
):
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
    
    # SO SÁNH VỚI MẠNG NƠ-RON (NEURAL NETWORK)
    print("\nSO SÁNH THÊM VỚI MẠNG NƠ-RON (MLPRegressor)")

    scaler_used = None
    if X_train_scaled is None or X_test_scaled is None:
        scaler_used = StandardScaler()
        X_train_scaled = scaler_used.fit_transform(X_train_best)
        X_test_scaled = scaler_used.transform(X_test_best)
        print("   Đã chuẩn hóa dữ liệu cho mạng nơ-ron")
    else:
        print("   Sử dụng dữ liệu đã chuẩn hóa sẵn cho mạng nơ-ron")

    nn_metrics, nn_model = train_neural_network(
        X_train_scaled, X_test_scaled, y_train_best, y_test_best
    )

    comparison_results['neural_network'] = {
        'metrics': nn_metrics,
        'predictions': nn_metrics.get('predictions'),
        'model': nn_model,
        'scaler': scaler_used,
    }
    
    # Cross-validation cho mô hình tốt nhất
    print("\nĐÁNH GIÁ ĐỘ ỔN ĐỊNH VỚI CROSS-VALIDATION (5-fold)")
    cv_results = perform_cross_validation(best_model, X_train_best, y_train_best)
    comparison_results['cv_results'] = cv_results
    
    # In kết quả so sánh
    print_comparison_results(dt_metrics_best, rf_metrics, nn_metrics, cv_results)
    
    return comparison_results

def train_neural_network(X_train, X_test, y_train, y_test):
    """Huấn luyện MLPRegressor với bộ tham số cố định (không GridSearch)."""

    fixed_params = {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'alpha': 0.001,
        'learning_rate_init': 0.0005,
        'max_iter': 1000,
        'early_stopping': True,
        'n_iter_no_change': 30,
        'tol': 1e-4,
        'random_state': 42,
    }

    print("   Huấn luyện mạng nơ-ron với bộ tham số đã được cung cấp")
    nn_model = MLPRegressor(**fixed_params)
    nn_model.fit(X_train, y_train)

    y_pred_nn = nn_model.predict(X_test)
    nn_metrics = calculate_metrics(y_test, y_pred_nn)
    nn_metrics['predictions'] = y_pred_nn
    nn_metrics['params'] = fixed_params

    print("   Hoàn tất huấn luyện mạng nơ-ron:")
    print(f"      R²:   {nn_metrics['r2']:.4f}")
    print(f"      RMSE: {nn_metrics['rmse']:.4f}")
    print(f"      MAE:  {nn_metrics['mae']:.4f}")
    print(f"      Median AE: {nn_metrics['medae']:.4f}")
    print(f"      Max Error: {nn_metrics['max_error']:.4f}")
    print(f"      MAPE: {nn_metrics['mape']:.2f}%")
    print(f"      Explained Variance: {nn_metrics['explained_variance']:.4f}")

    return nn_metrics, nn_model

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

def print_comparison_results(dt_metrics, rf_metrics, nn_metrics, cv_results):
    """In kết quả so sánh các mô hình"""
    print("\n SO SÁNH HIỆU SUẤT TRÊN TẬP TEST TỐT NHẤT:")
    for label, metrics in (
        ("Decision Tree (tốt nhất)", dt_metrics),
        ("Random Forest", rf_metrics),
        ("Mạng nơ-ron (MLPRegressor)", nn_metrics),
    ):
        print(f"    {label}:")
        print(f"       R²:                {metrics['r2']:.4f}")
        print(f"       RMSE:              {metrics['rmse']:.4f}")
        print(f"       MAE:               {metrics['mae']:.4f}")
        print(f"       Median AE:         {metrics['medae']:.4f}")
        print(f"       Max Error:         {metrics['max_error']:.4f}")
        print(f"       MAPE:              {metrics['mape']:.2f}%")
        print(f"       Explained Variance:{metrics['explained_variance']:.4f}")
    
    print(f"\nKẾT QUẢ CROSS-VALIDATION (5-fold):")
    print(f"    Train R²:     {cv_results['train_r2'].mean():.4f} (±{cv_results['train_r2'].std():.4f})")
    print(f"    Test R²:      {cv_results['test_r2'].mean():.4f} (±{cv_results['test_r2'].std():.4f})")
    print(f"    Test RMSE:    {cv_results['test_rmse'].mean():.4f} (±{cv_results['test_rmse'].std():.4f})")
    print(f"    Test MAE:     {cv_results['test_mae'].mean():.4f} (±{cv_results['test_mae'].std():.4f})")
    
    cv_stability = "RẤT ỔN ĐỊNH" if cv_results['test_r2'].std() < 0.02 else "KHÁ ỔN ĐỊNH" if cv_results['test_r2'].std() < 0.05 else "CÓ BIẾN ĐỘNG"
    print(f"    Độ ổn định:    {cv_stability} (độ lệch chuẩn: {cv_results['test_r2'].std():.4f})")