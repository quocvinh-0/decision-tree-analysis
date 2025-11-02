import pandas as pd
import joblib
import os

def save_results(train_df, test_df, feature_importance_df, best_model_info, 
                comparison_results, best_model):
    """
    Lưu tất cả kết quả vào file
    """
    # Lưu mô hình và scaler
    save_models(best_model_info, best_model)
    
    # Lưu kết quả vào Excel
    save_results_to_excel(train_df, test_df, feature_importance_df, 
                         best_model_info, comparison_results)

def save_models(best_model_info, best_model):
    """Lưu mô hình và scaler"""
    model_path = os.path.join('result', 'best_decision_tree_model.pkl')
    scaler_path = os.path.join('result', 'scaler.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(best_model_info['scaler'], scaler_path)
    
    print("\n✅ Đã lưu mô hình và scaler thành công vào thư mục 'result':")
    print(f"   • {model_path}")
    print(f"   • {scaler_path}")

def save_results_to_excel(train_df, test_df, feature_importance_df, 
                         best_model_info, comparison_results):
    """Lưu kết quả vào file Excel"""
    excel_path = os.path.join('result', 'results_summary.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Tổng quan kết quả
        save_summary_sheet(writer, test_df, best_model_info, comparison_results)
        
        # Sheet 2: So sánh mô hình
        save_model_comparison_sheet(writer, comparison_results)
        
        # Sheet 3: Feature Importance
        feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        
        # Sheet 4: Kết quả 10 lần chạy
        save_detailed_results_sheet(writer, test_df)
        
        # Sheet 5: Tham số mô hình tốt nhất
        save_best_model_sheet(writer, best_model_info)
        
        # Sheet 6: Cross-validation results
        save_cv_results_sheet(writer, comparison_results)
    
    print(f"✅ Đã lưu file Excel tổng hợp: {excel_path}")

def save_summary_sheet(writer, test_df, best_model_info, comparison_results):
    """Lưu sheet tổng quan"""
    summary_data = {
        'Metric': ['R² Trung bình', 'RMSE Trung bình', 'MAE Trung bình', 'MAPE Trung bình',
                  'R² Tốt nhất', 'Độ lệch chuẩn R²', 'Số lần chạy', 'Mô hình tốt nhất',
                  'Cross-Val R²', 'Cross-Val RMSE'],
        'Giá trị': [f"{test_df['r2'].mean():.4f}", f"{test_df['rmse'].mean():.4f}", 
                   f"{test_df['mae'].mean():.4f}", f"{test_df['mape'].mean():.2f}%",
                   f"{best_model_info['test_r2']:.4f}", f"{test_df['r2'].std():.4f}",
                   '10', f"Lần {best_model_info['run_id'] + 1}",
                   f"{comparison_results['cv_results']['test_r2'].mean():.4f}", 
                   f"{comparison_results['cv_results']['test_rmse'].mean():.4f}"],
        'Đánh giá': [f"{'✅ Tốt' if test_df['r2'].mean() > 0.9 else '⚠️ Khá'}", 
                    f"{'✅ Tốt' if test_df['rmse'].mean() < 5 else '⚠️ Trung bình'}",
                    f"{'✅ Tốt' if test_df['mae'].mean() < 4 else '⚠️ Trung bình'}",
                    f"{'✅ Tốt' if test_df['mape'].mean() < 5 else '⚠️ Khá'}",
                    '🏆 Tốt nhất', f"{'Ổn định' if test_df['r2'].std() < 0.02 else 'Biến động'}",
                    'Đủ', 'Đã chọn',
                    f"{'✅ Tốt' if comparison_results['cv_results']['test_r2'].mean() > 0.9 else '⚠️ Khá'}",
                    f"{'✅ Tốt' if comparison_results['cv_results']['test_rmse'].mean() < 5 else '⚠️ Trung bình'}"]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Tổng quan', index=False)

def save_model_comparison_sheet(writer, comparison_results):
    """Lưu sheet so sánh mô hình"""
    dt_metrics = comparison_results['decision_tree']['metrics']
    rf_metrics = comparison_results['random_forest']['metrics']
    knn_metrics = comparison_results['knn']['metrics']
    
    model_comparison = {
        'Mô hình': ['Decision Tree', 'Random Forest', 'KNN'],
        'R²': [dt_metrics['r2'], rf_metrics['r2'], knn_metrics['r2']],
        'RMSE': [dt_metrics['rmse'], rf_metrics['rmse'], knn_metrics['rmse']],
        'MAE': [dt_metrics['mae'], rf_metrics['mae'], knn_metrics['mae']],
        'MAPE': [f"{dt_metrics['mape']:.2f}%", f"{rf_metrics['mape']:.2f}%", f"{knn_metrics['mape']:.2f}%"],
        'Đánh giá': [f"{'✅ Tốt' if dt_metrics['r2'] > 0.9 else '⚠️ Khá'}",
                    f"{'✅ Tốt' if rf_metrics['r2'] > 0.9 else '⚠️ Khá'}",
                    f"{'✅ Tốt' if knn_metrics['r2'] > 0.9 else '⚠️ Khá'}"]
    }
    pd.DataFrame(model_comparison).to_excel(writer, sheet_name='So sánh mô hình', index=False)

def save_detailed_results_sheet(writer, test_df):
    """Lưu sheet kết quả chi tiết 10 lần chạy"""
    detailed_results = test_df.copy()
    detailed_results['Lần chạy'] = range(1, 11)
    detailed_results.to_excel(writer, sheet_name='10 Lần chạy', index=False)

def save_best_model_sheet(writer, best_model_info):
    """Lưu sheet thông tin mô hình tốt nhất"""
    best_params_df = pd.DataFrame([best_model_info['params']])
    best_params_df['Test_R2'] = best_model_info['test_r2']
    best_params_df['Lần_chạy'] = best_model_info['run_id'] + 1
    best_params_df.to_excel(writer, sheet_name='Mô hình tốt nhất', index=False)

def save_cv_results_sheet(writer, comparison_results):
    """Lưu sheet kết quả cross-validation"""
    cv_details = pd.DataFrame({
        'Fold': range(1, 6),
        'Train_R2': comparison_results['cv_results']['train_r2'],
        'Test_R2': comparison_results['cv_results']['test_r2'],
        'Test_RMSE': comparison_results['cv_results']['test_rmse'],
        'Test_MAE': comparison_results['cv_results']['test_mae']
    })
    cv_details.to_excel(writer, sheet_name='Cross-Validation', index=False)