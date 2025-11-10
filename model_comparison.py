import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import f1_score, accuracy_score, classification_report
# Import calculate_metrics từ improved module
from improved.model_trainer_improved import calculate_metrics

def compare_with_other_models(X_train_best, X_test_best, y_train_best, y_test_best, best_model, X_train_scaled=None, X_test_scaled=None):
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
    
    # BỎ SO SÁNH VỚI KNN (theo yêu cầu - KNN thường tốt hơn Decision Tree)
    # # SO SÁNH VỚI KNN (CẦN CHUẨN HÓA DỮ LIỆU)
    # print("\n🔍 SO SÁNH THÊM VỚI KNN REGRESSOR (TỐI ƯU HÓA THAM SỐ)")
    # # KNN cần chuẩn hóa dữ liệu (dựa trên khoảng cách)
    # if X_train_scaled is None or X_test_scaled is None:
    #     print("   ⚠️  Chưa có dữ liệu đã chuẩn hóa, đang tạo scaler...")
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train_best)
    #     X_test_scaled = scaler.transform(X_test_best)
    #     print("   ✅ Đã chuẩn hóa dữ liệu cho KNN")
    # else:
    #     print("   ✅ Sử dụng dữ liệu đã chuẩn hóa sẵn cho KNN")
    # knn_metrics, best_knn = train_optimized_knn(X_train_scaled, X_test_scaled, y_train_best, y_test_best)
    # comparison_results['knn'] = {
    #     'metrics': knn_metrics,
    #     'predictions': knn_metrics.get('predictions'),
    #     'model': best_knn
    # }
    knn_metrics = None  # Không sử dụng KNN
    
    # SO SÁNH VỚI NAIVE BAYES (CHO CLASSIFICATION)
    # Chuyển bài toán thành Classification để so sánh với Naive Bayes
    print("\n🔍 SO SÁNH THÊM VỚI NAIVE BAYES (CLASSIFICATION)")
    print("   ⚠️  Lưu ý: Naive Bayes chỉ dùng cho Classification")
    print("   → Chuyển bài toán thành Classification (chia PE thành 3 lớp)")
    print("   ℹ️  Lưu ý: Naive Bayes (GaussianNB) KHÔNG BẮT BUỘC cần chuẩn hóa")
    print("      (khác với Decision Tree - không cần chuẩn hóa)")
    print("      Nhưng chuẩn hóa có thể giúp cải thiện hiệu suất khi các thuộc tính có thang đo khác nhau")
    
    # Thử cả hai cách: có và không chuẩn hóa
    print("\n   📊 Thử Naive Bayes KHÔNG chuẩn hóa (giống Decision Tree):")
    nb_metrics_no_scale, best_nb_no_scale = train_naive_bayes_classification(
        X_train_best, X_test_best, y_train_best, y_test_best
    )
    
    # Kiểm tra xem dữ liệu đã được chuẩn hóa chưa
    if X_train_scaled is None or X_test_scaled is None:
        print("\n   📊 Thử Naive Bayes CÓ chuẩn hóa:")
        print("   ⚠️  Chưa có dữ liệu đã chuẩn hóa, đang tạo scaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_best)
        X_test_scaled = scaler.transform(X_test_best)
        print("   ✅ Đã chuẩn hóa dữ liệu cho Naive Bayes")
    else:
        print("\n   📊 Thử Naive Bayes CÓ chuẩn hóa:")
        print("   ✅ Sử dụng dữ liệu đã chuẩn hóa sẵn cho Naive Bayes")
    
    nb_metrics_scaled, best_nb_scaled = train_naive_bayes_classification(
        X_train_scaled, X_test_scaled, y_train_best, y_test_best
    )
    
    # So sánh và chọn cách tốt hơn
    print("\n   📈 SO SÁNH KẾT QUẢ:")
    print(f"      KHÔNG chuẩn hóa: R² = {nb_metrics_no_scale['r2']:.4f}, RMSE = {nb_metrics_no_scale['rmse']:.4f}")
    print(f"      CÓ chuẩn hóa:    R² = {nb_metrics_scaled['r2']:.4f}, RMSE = {nb_metrics_scaled['rmse']:.4f}")
    
    # Chọn cách tốt hơn (R² cao hơn hoặc RMSE thấp hơn)
    if nb_metrics_scaled['r2'] > nb_metrics_no_scale['r2']:
        print("      ✅ Chọn mô hình CÓ chuẩn hóa (R² cao hơn)")
        nb_metrics = nb_metrics_scaled
        best_nb = best_nb_scaled
    else:
        print("      ✅ Chọn mô hình KHÔNG chuẩn hóa (R² cao hơn hoặc tương đương)")
        nb_metrics = nb_metrics_no_scale
        best_nb = best_nb_no_scale
    comparison_results['naive_bayes'] = {
        'metrics': nb_metrics,
        'predictions': nb_metrics.get('predictions'),
        'model': best_nb
    }
    
    # Cross-validation cho mô hình tốt nhất
    print("\n🔄 ĐÁNH GIÁ ĐỘ ỔN ĐỊNH VỚI CROSS-VALIDATION (5-fold)")
    cv_results = perform_cross_validation(best_model, X_train_best, y_train_best)
    comparison_results['cv_results'] = cv_results
    
    # In kết quả so sánh
    nb_metrics = comparison_results.get('naive_bayes', {}).get('metrics')
    print_comparison_results(dt_metrics_best, rf_metrics, knn_metrics, cv_results, nb_metrics)
    
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

def train_naive_bayes_classification(X_train, X_test, y_train, y_test):
    """
    Huấn luyện Naive Bayes cho Classification
    Chuyển bài toán Regression thành Classification bằng cách chia PE thành 3 lớp
    """
    # Chia PE thành 3 lớp: Thấp, Trung bình, Cao
    pe_q1 = np.percentile(y_train, 33.33)
    pe_q2 = np.percentile(y_train, 66.67)
    
    def classify_pe(value):
        if value < pe_q1:
            return 'Thap'
        elif value < pe_q2:
            return 'Trung binh'
        else:
            return 'Cao'
    
    # Chuyển đổi y_train và y_test thành classification
    y_train_class = np.array([classify_pe(val) for val in y_train])
    y_test_class = np.array([classify_pe(val) for val in y_test])
    
    print(f"   Phân loại PE:")
    print(f"      Thấp: < {pe_q1:.2f} MW ({np.sum(y_train_class == 'Thap')} train, {np.sum(y_test_class == 'Thap')} test)")
    print(f"      Trung bình: {pe_q1:.2f} - {pe_q2:.2f} MW ({np.sum(y_train_class == 'Trung binh')} train, {np.sum(y_test_class == 'Trung binh')} test)")
    print(f"      Cao: >= {pe_q2:.2f} MW ({np.sum(y_train_class == 'Cao')} train, {np.sum(y_test_class == 'Cao')} test)")
    
    # Huấn luyện Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train_class)
    y_pred_class = nb_model.predict(X_test)
    
    # Tính metrics cho classification
    f1 = f1_score(y_test_class, y_pred_class, average='weighted')
    accuracy = accuracy_score(y_test_class, y_pred_class)
    
    # Tính metrics cho regression (dự đoán giá trị trung bình của mỗi lớp)
    # Để so sánh với các mô hình regression khác
    class_means = {
        'Thap': np.mean(y_train[y_train_class == 'Thap']) if np.sum(y_train_class == 'Thap') > 0 else pe_q1/2,
        'Trung binh': np.mean(y_train[y_train_class == 'Trung binh']) if np.sum(y_train_class == 'Trung binh') > 0 else (pe_q1 + pe_q2)/2,
        'Cao': np.mean(y_train[y_train_class == 'Cao']) if np.sum(y_train_class == 'Cao') > 0 else (pe_q2 + np.max(y_train))/2
    }
    
    y_pred_regression = np.array([class_means[pred] for pred in y_pred_class])
    nb_metrics_regression = calculate_metrics(y_test, y_pred_regression)
    
    # Kết hợp metrics
    nb_metrics = {
        'f1_score': f1,
        'accuracy': accuracy,
        'r2': nb_metrics_regression['r2'],
        'rmse': nb_metrics_regression['rmse'],
        'mae': nb_metrics_regression['mae'],
        'mape': nb_metrics_regression['mape'],
        'predictions': y_pred_regression,  # Dự đoán dạng regression để so sánh
        'predictions_class': y_pred_class   # Dự đoán dạng classification
    }
    
    print(f"\n✅ Naive Bayes (Classification):")
    print(f"    F1 Score: {f1:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    R² (regression): {nb_metrics_regression['r2']:.4f}")
    print(f"    RMSE (regression): {nb_metrics_regression['rmse']:.4f}")
    
    return nb_metrics, nb_model

def print_comparison_results(dt_metrics, rf_metrics, knn_metrics, cv_results, nb_metrics=None):
    """In kết quả so sánh các mô hình (bỏ KNN)"""
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
    
    # Bỏ KNN
    # print(f"    KNN (tối ưu):")
    # print(f"       R²:   {knn_metrics['r2']:.4f}")
    # print(f"       RMSE: {knn_metrics['rmse']:.4f}")
    # print(f"       MAE:  {knn_metrics['mae']:.4f}")
    # print(f"       MAPE: {knn_metrics['mape']:.2f}%")
    
    if nb_metrics is not None:
        print(f"    Naive Bayes (Classification):")
        print(f"       F1 Score: {nb_metrics['f1_score']:.4f}")
        print(f"       Accuracy: {nb_metrics['accuracy']:.4f}")
        print(f"       R² (regression): {nb_metrics['r2']:.4f}")
        print(f"       RMSE (regression): {nb_metrics['rmse']:.4f}")
        print(f"       MAE (regression): {nb_metrics['mae']:.4f}")
        print(f"       MAPE (regression): {nb_metrics['mape']:.2f}%")
    
    print(f"\n📊 KẾT QUẢ CROSS-VALIDATION (5-fold):")
    print(f"    Train R²:     {cv_results['train_r2'].mean():.4f} (±{cv_results['train_r2'].std():.4f})")
    print(f"    Test R²:      {cv_results['test_r2'].mean():.4f} (±{cv_results['test_r2'].std():.4f})")
    print(f"    Test RMSE:    {cv_results['test_rmse'].mean():.4f} (±{cv_results['test_rmse'].std():.4f})")
    
    cv_stability = "RẤT ỔN ĐỊNH" if cv_results['test_r2'].std() < 0.02 else "KHÁ ỔN ĐỊNH" if cv_results['test_r2'].std() < 0.05 else "CÓ BIẾN ĐỘNG"
    print(f"    Độ ổn định:    {cv_stability} (độ lệch chuẩn: {cv_results['test_r2'].std():.4f})")