import os
import joblib
import pandas as pd


def save_results(train_df, test_df, feature_importance_df, best_model_info,
                comparison_results, best_model):
    """Lưu toàn bộ kết quả chạy mô hình vào mô hình + Excel."""
    save_models(best_model_info, best_model)
    save_results_to_excel(
        train_df, test_df, feature_importance_df, best_model_info, comparison_results
    )


def save_models(best_model_info, best_model):
    """Lưu mô hình Decision Tree và scaler (nếu hiện diện)."""
    model_path = os.path.join('result', 'best_decision_tree_model.pkl')
    joblib.dump(best_model, model_path)
    print("\nĐã lưu mô hình Decision Tree vào thư mục 'result':")
    print(f"   - {model_path}")

    scaler = best_model_info.get('scaler')
    if scaler is not None:
        scaler_path = os.path.join('result', 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"   - {scaler_path} (Scaler đi kèm)")


def save_results_to_excel(train_df, test_df, feature_importance_df,
                         best_model_info, comparison_results):
    """Ghi kết quả ra file Excel nhiều sheet."""
    excel_path = os.path.join('result', 'results_summary.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        save_summary_sheet(writer, test_df, best_model_info, comparison_results)
        save_model_comparison_sheet(writer, comparison_results)
        feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        save_detailed_results_sheet(writer, test_df)
        save_best_model_sheet(writer, best_model_info)
        save_cv_results_sheet(writer, comparison_results)

    print(f"Đã lưu file Excel tổng hợp: {excel_path}")


def save_summary_sheet(writer, test_df, best_model_info, comparison_results):
    """Lưu sheet tổng quan"""
    def status_text(condition, good='Tốt', caution='Cần theo dõi'):
        return good if condition else caution

    summary_data = {
        'Metric': [
            'R² Trung bình',
            'RMSE Trung bình',
            'MAE Trung bình',
            'Median AE Trung bình',
            'Max Error Trung bình',
            'MAPE Trung bình',
            'Explained Variance TB',
            'R² Tốt nhất',
            'Độ lệch chuẩn R²',
            'Số lần chạy',
            'Mô hình tốt nhất',
            'Cross-Val R²',
            'Cross-Val RMSE',
            'Cross-Val MAE',
        ],
        'Giá trị': [
            f"{test_df['r2'].mean():.4f}",
            f"{test_df['rmse'].mean():.4f}",
            f"{test_df['mae'].mean():.4f}",
            f"{test_df['medae'].mean():.4f}",
            f"{test_df['max_error'].mean():.4f}",
            f"{test_df['mape'].mean():.2f}%",
            f"{test_df['explained_variance'].mean():.4f}",
            f"{best_model_info['test_r2']:.4f}",
            f"{test_df['r2'].std():.4f}",
            '10',
            f"Lần {best_model_info['run_id'] + 1}",
            f"{comparison_results['cv_results']['test_r2'].mean():.4f}",
            f"{comparison_results['cv_results']['test_rmse'].mean():.4f}",
            f"{comparison_results['cv_results']['test_mae'].mean():.4f}",
        ],
        'Đánh giá': [
            status_text(test_df['r2'].mean() > 0.9),
            status_text(test_df['rmse'].mean() < 5),
            status_text(test_df['mae'].mean() < 4),
            status_text(test_df['medae'].mean() < 4),
            status_text(test_df['max_error'].mean() < 10),
            status_text(test_df['mape'].mean() < 5),
            status_text(test_df['explained_variance'].mean() > 0.9),
            'Mốc tham chiếu',
            status_text(test_df['r2'].std() < 0.02, good='Ổn định', caution='Biến động'),
            'Số lần chạy cố định',
            'Đã chọn',
            status_text(comparison_results['cv_results']['test_r2'].mean() > 0.9),
            status_text(comparison_results['cv_results']['test_rmse'].mean() < 5),
            status_text(comparison_results['cv_results']['test_mae'].mean() < 4),
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Tổng quan', index=False)

def save_model_comparison_sheet(writer, comparison_results):
    """Lưu sheet so sánh mô hình"""
    dt_metrics = comparison_results['decision_tree']['metrics']
    rf_metrics = comparison_results['random_forest']['metrics']

    models = ['Decision Tree', 'Random Forest']
    metrics = [dt_metrics, rf_metrics]

    if 'neural_network' in comparison_results:
        models.append('Neural Network')
        metrics.append(comparison_results['neural_network']['metrics'])

    model_comparison = {
        'Mô hình': models,
        'R²': [m['r2'] for m in metrics],
        'RMSE': [m['rmse'] for m in metrics],
        'MAE': [m['mae'] for m in metrics],
        'Median AE': [m['medae'] for m in metrics],
        'Max Error': [m['max_error'] for m in metrics],
        'MAPE (%)': [m['mape'] for m in metrics],
        'Explained Variance': [m['explained_variance'] for m in metrics],
        'Đánh giá': ["Tốt" if m['r2'] > 0.9 else "Cần theo dõi" for m in metrics]
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