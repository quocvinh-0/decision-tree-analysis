import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.tree import plot_tree
from sklearn.model_selection import learning_curve
from scipy import stats

def create_all_visualizations(train_df, test_df, feature_importance_df, best_model_info, 
                            comparison_results, X_scaled, y):
    """Tạo tất cả các biểu đồ trực quan"""
    
    # 1. Biểu đồ so sánh mô hình
    create_model_comparison_chart(comparison_results)
    
    # 2. Biểu đồ feature importance
    create_feature_importance_chart(feature_importance_df)
    
    # 3. Biểu đồ Actual vs Predicted
    create_actual_vs_predicted_chart(best_model_info, comparison_results)
    
    # 4. Biểu đồ tổng hợp 10 lần chạy
    create_summary_plots(train_df, test_df, comparison_results)
    
    # 5. Phân tích sai số
    create_residuals_analysis(best_model_info, comparison_results)
    
    # 6. Learning curves
    create_learning_curves(best_model_info, comparison_results, X_scaled, y)
    
    # 7. Biểu đồ so sánh chi tiết 10 lần lặp
    create_detailed_comparison_plots(train_df, test_df, best_model_info)
    
    # 8. Biểu đồ chi tiết từng lần chạy
    create_detailed_runs_analysis(train_df, test_df, best_model_info)
    
    # 9. Vẽ cây quyết định
    plot_decision_tree(best_model_info)

def create_model_comparison_chart(comparison_results):
    """Biểu đồ so sánh các mô hình"""
    print("\n📊 1. Biểu đồ so sánh mô hình")
    comparison_path = os.path.join('img', 'model_comparison.png')
    
    plt.figure(figsize=(10, 6))
    models = ['Decision Tree', 'Random Forest', 'KNN']
    r2_scores = [
        comparison_results['decision_tree']['metrics']['r2'],
        comparison_results['random_forest']['metrics']['r2'],
        comparison_results['knn']['metrics']['r2']
    ]
    colors = ['#2ECC71', '#3498DB', '#9B59B6']
    
    bars = plt.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel('R² Score', fontsize=12)
    plt.title('SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH', fontweight='bold', fontsize=14)
    plt.ylim(0.8, 1.0)
    plt.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {comparison_path}")

def create_feature_importance_chart(feature_importance_df):
    """Biểu đồ feature importance"""
    print("📊 2. Biểu đồ feature importance")
    feature_img_path = os.path.join('img', 'feature_importance.png')
    
    plt.figure(figsize=(10, 6))
    features = feature_importance_df['Đặc trưng']
    importances = feature_importance_df['Độ quan trọng trung bình']
    std_dev = feature_importance_df['Độ lệch chuẩn']
    
    bars = plt.bar(features, importances, yerr=std_dev, capsize=8, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
                   alpha=0.8, edgecolor='black')
    plt.ylabel('Độ quan trọng trung bình', fontsize=12)
    plt.title('ĐỘ QUAN TRỌNG ĐẶC TRƯNG (10 LẦN CHẠY)', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    for bar, importance, std in zip(bars, importances, std_dev):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{importance:.3f} (±{std:.3f})', ha='center', va='bottom', 
                 fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(feature_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {feature_img_path}")

def create_actual_vs_predicted_chart(best_model_info, comparison_results):
    """Biểu đồ so sánh giá trị thực và dự đoán"""
    print("📊 3. Biểu đồ Actual vs Predicted")
    actual_pred_path = os.path.join('img', 'actual_vs_predicted.png')
    
    plt.figure(figsize=(15, 5))
    y_test = best_model_info['y_test']
    
    # Decision Tree
    plt.subplot(1, 3, 1)
    y_pred_dt = best_model_info['y_pred_test']
    dt_r2 = comparison_results['decision_tree']['metrics']['r2']
    plt.scatter(y_test, y_pred_dt, alpha=0.6, s=30, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title(f'Decision Tree\nR² = {dt_r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Random Forest
    plt.subplot(1, 3, 2)
    y_pred_rf = comparison_results['random_forest']['predictions']
    rf_r2 = comparison_results['random_forest']['metrics']['r2']
    plt.scatter(y_test, y_pred_rf, alpha=0.6, s=30, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title(f'Random Forest\nR² = {rf_r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # KNN
    plt.subplot(1, 3, 3)
    y_pred_knn = comparison_results['knn']['predictions']
    knn_r2 = comparison_results['knn']['metrics']['r2']
    plt.scatter(y_test, y_pred_knn, alpha=0.6, s=30, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title(f'KNN\nR² = {knn_r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(actual_pred_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {actual_pred_path}")

def create_summary_plots(train_df, test_df, comparison_results):
    """Biểu đồ tổng hợp kết quả"""
    print("📊 4. Biểu đồ tổng hợp 10 lần chạy")
    summary_plots_path = os.path.join('img', 'summary_plots.png')
    
    plt.figure(figsize=(20, 12))
    plt.suptitle("PHÂN TÍCH TỔNG HỢP 10 LẦN HUẤN LUYỆN DECISION TREE", 
                 fontsize=20, fontweight='bold', y=1.03)
    
    # Biểu đồ 1: So sánh R² qua 10 lần chạy
    plt.subplot(2, 3, 1)
    runs = range(1, 11)
    plt.plot(runs, train_df['r2'], marker='o', linewidth=2, markersize=8, 
             label='Train R²', color='#2ECC71')
    plt.plot(runs, test_df['r2'], marker='s', linewidth=2, markersize=8, 
             label='Test R²', color='#E74C3C')
    plt.axhline(y=test_df['r2'].mean(), color='red', linestyle='--', alpha=0.7, 
                label=f"Test R² TB: {test_df['r2'].mean():.3f}")
    plt.xlabel('Lần chạy')
    plt.ylabel('R² Score')
    plt.title('SO SÁNH R² QUA 10 LẦN CHẠY', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 2: So sánh RMSE qua 10 lần chạy
    plt.subplot(2, 3, 2)
    plt.plot(runs, train_df['rmse'], marker='o', linewidth=2, markersize=8, 
             label='Train RMSE', color='#3498DB')
    plt.plot(runs, test_df['rmse'], marker='s', linewidth=2, markersize=8, 
             label='Test RMSE', color='#F39C12')
    plt.axhline(y=test_df['rmse'].mean(), color='orange', linestyle='--', alpha=0.7, 
                label=f"Test RMSE TB: {test_df['rmse'].mean():.3f}")
    plt.xlabel('Lần chạy')
    plt.ylabel('RMSE')
    plt.title('SO SÁNH RMSE QUA 10 LẦN CHẠY', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 3: Phân bố R² trên tập test
    plt.subplot(2, 3, 3)
    sns.boxplot(data=[train_df['r2'], test_df['r2']], palette=['#AED6F1', '#FAD7A0'])
    plt.xticks([0, 1], ['Train R²', 'Test R²'])
    plt.ylabel('R² Score')
    plt.title('PHÂN BỐ R² SCORE (10 LẦN)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 4: Hiệu suất theo bộ tham số
    plt.subplot(2, 3, 4)
    param_names = [f"Set {i+1}" for i in range(10)]
    test_r2_values = test_df['r2']
    plt.scatter(param_names, test_r2_values, s=100, alpha=0.7, 
                c=test_r2_values, cmap='viridis')
    plt.axhline(y=test_r2_values.mean(), color='red', linestyle='--', label='Trung bình')
    plt.xlabel('Bộ tham số')
    plt.ylabel('Test R²')
    plt.title('HIỆU SUẤT THEO BỘ THAM SỐ', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.colorbar(label='R² Score')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 5: So sánh 3 mô hình
    plt.subplot(2, 3, 5)
    models_compare = ['DT', 'RF', 'KNN']
    r2_compare = [
        comparison_results['decision_tree']['metrics']['r2'],
        comparison_results['random_forest']['metrics']['r2'],
        comparison_results['knn']['metrics']['r2']
    ]
    plt.bar(models_compare, r2_compare, color=['#2ECC71', '#3498DB', '#9B59B6'])
    plt.ylabel('R² Score')
    plt.title('SO SÁNH 3 MÔ HÌNH', fontweight='bold')
    for i, v in enumerate(r2_compare):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 6: Cross-validation results
    plt.subplot(2, 3, 6)
    cv_folds = range(1, 6)
    cv_test_r2 = comparison_results['cv_results']['test_r2']
    plt.plot(cv_folds, cv_test_r2, marker='o', linewidth=2, markersize=8, color='#E74C3C')
    plt.axhline(y=cv_test_r2.mean(), color='red', linestyle='--', 
                label=f'Trung bình: {cv_test_r2.mean():.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Test R²')
    plt.title('CROSS-VALIDATION (5-fold)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(summary_plots_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {summary_plots_path}")
def create_residuals_analysis(best_model_info, comparison_results):
    """Phân tích sai số (residuals analysis)"""
    print("📊 5. Phân tích sai số")
    
    y_test = best_model_info['y_test']
    residuals_dt = y_test - best_model_info['y_pred_test']
    residuals_rf = y_test - comparison_results['random_forest']['predictions']
    residuals_knn = y_test - comparison_results['knn']['predictions']
    
    residuals_path = os.path.join('img', 'residuals_analysis.png')
    plt.figure(figsize=(18, 12))
    
    # Biểu đồ 1: Residuals vs Predicted cho DT
    plt.subplot(2, 3, 1)
    plt.scatter(best_model_info['y_pred_test'], residuals_dt, alpha=0.6, s=30, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Sai số (Residuals)')
    plt.title(f'Decision Tree\nStd: {residuals_dt.std():.3f}')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 2: Residuals vs Predicted cho RF
    plt.subplot(2, 3, 2)
    y_pred_rf = comparison_results['random_forest']['predictions']
    plt.scatter(y_pred_rf, residuals_rf, alpha=0.6, s=30, color='green')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Sai số (Residuals)')
    plt.title(f'Random Forest\nStd: {residuals_rf.std():.3f}')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 3: Residuals vs Predicted cho KNN
    plt.subplot(2, 3, 3)
    y_pred_knn = comparison_results['knn']['predictions']
    plt.scatter(y_pred_knn, residuals_knn, alpha=0.6, s=30, color='purple')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Giá trị dự đoán')
    plt.ylabel('Sai số (Residuals)')
    plt.title(f'KNN\nStd: {residuals_knn.std():.3f}')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 4: Phân phối residuals
    plt.subplot(2, 3, 4)
    plt.hist(residuals_dt, bins=30, alpha=0.7, label=f'DT (std: {residuals_dt.std():.3f})', color='blue')
    plt.hist(residuals_rf, bins=30, alpha=0.7, label=f'RF (std: {residuals_rf.std():.3f})', color='green')
    plt.hist(residuals_knn, bins=30, alpha=0.7, label=f'KNN (std: {residuals_knn.std():.3f})', color='purple')
    plt.xlabel('Sai số (Residuals)')
    plt.ylabel('Tần suất')
    plt.title('PHÂN PHỐI SAI SỐ CỦA CÁC MÔ HÌNH')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 5: Q-Q plot cho Decision Tree
    plt.subplot(2, 3, 5)
    stats.probplot(residuals_dt, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Decision Tree Residuals')
    
    # Biểu đồ 6: So sánh độ lớn sai số
    plt.subplot(2, 3, 6)
    residuals_abs = [np.abs(residuals_dt).mean(), np.abs(residuals_rf).mean(), np.abs(residuals_knn).mean()]
    models_resid = ['Decision Tree', 'Random Forest', 'KNN']
    bars = plt.bar(models_resid, residuals_abs, color=['blue', 'green', 'purple'], alpha=0.7)
    plt.ylabel('Sai số tuyệt đối trung bình (MAE)')
    plt.title('SO SÁNH ĐỘ LỚN SAI SỐ')
    for bar, value in zip(bars, residuals_abs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}', 
                 ha='center', va='bottom', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {residuals_path}")
    
    # Phân tích thống kê residuals
    print(f"\n📊 PHÂN TÍCH THỐNG KÊ SAI SỐ:")
    print(f"    Decision Tree: Mean = {residuals_dt.mean():.4f}, Std = {residuals_dt.std():.4f}")
    print(f"    Random Forest: Mean = {residuals_rf.mean():.4f}, Std = {residuals_rf.std():.4f}")
    print(f"    KNN:           Mean = {residuals_knn.mean():.4f}, Std = {residuals_knn.std():.4f}")

def create_learning_curves(best_model_info, comparison_results, X_scaled, y):
    """Tạo learning curves"""
    print("📊 6. Learning Curves")
    
    print("Đang vẽ và lưu learning curves...")
    plot_and_save_learning_curve(best_model_info['model'], "Decision Tree (Best Model)", 
                                "learning_curve_dt.png", X_scaled, y)
    
    if 'model' in comparison_results['random_forest']:
        plot_and_save_learning_curve(comparison_results['random_forest']['model'], 
                                    "Random Forest", "learning_curve_rf.png", X_scaled, y)

def plot_and_save_learning_curve(estimator, title, filename, X, y, cv=5):
    """Vẽ và lưu learning curve"""
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', n_jobs=-1, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score", linewidth=2)
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score", linewidth=2)
    
    plt.xlabel("Số lượng mẫu training", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.title(f"Learning Curve: {title}", fontweight='bold')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join('img', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Đã lưu: {filepath}")

def create_detailed_comparison_plots(train_df, test_df, best_model_info):
    """Biểu đồ so sánh chi tiết 10 lần lặp"""
    print("📊 7. Biểu đồ so sánh chi tiết 10 lần lặp")
    
    comparison_10_runs_path = os.path.join('img', 'comparison_10_runs.png')
    plt.figure(figsize=(18, 12))
    
    # Biểu đồ 1: So sánh R² train vs test qua 10 lần
    plt.subplot(2, 3, 1)
    runs = range(1, 11)
    plt.plot(runs, train_df['r2'], marker='o', linewidth=3, markersize=8, 
             label=f'Train R² (TB: {train_df["r2"].mean():.3f})', color='#2ECC71')
    plt.plot(runs, test_df['r2'], marker='s', linewidth=3, markersize=8, 
             label=f'Test R² (TB: {test_df["r2"].mean():.3f})', color='#E74C3C')
    plt.axhline(y=train_df['r2'].mean(), color='#2ECC71', linestyle='--', alpha=0.5)
    plt.axhline(y=test_df['r2'].mean(), color='#E74C3C', linestyle='--', alpha=0.5)
    plt.xlabel('Lần chạy')
    plt.ylabel('R² Score')
    plt.title('SO SÁNH R² TRAIN vs TEST QUA 10 LẦN', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.0)
    
    # Thêm annotation cho lần chạy tốt nhất
    best_run = best_model_info['run_id'] + 1
    best_test_r2 = best_model_info['test_r2']
    plt.annotate(f'Tốt nhất\nLần {best_run}\nR² = {best_test_r2:.3f}', 
                 xy=(best_run, best_test_r2), 
                 xytext=(best_run+0.5, best_test_r2-0.02),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontweight='bold', color='red')
    
    # Biểu đồ 2: So sánh RMSE train vs test qua 10 lần
    plt.subplot(2, 3, 2)
    plt.plot(runs, train_df['rmse'], marker='o', linewidth=3, markersize=8, 
             label=f'Train RMSE (TB: {train_df["rmse"].mean():.3f})', color='#3498DB')
    plt.plot(runs, test_df['rmse'], marker='s', linewidth=3, markersize=8, 
             label=f'Test RMSE (TB: {test_df["rmse"].mean():.3f})', color='#F39C12')
    plt.axhline(y=train_df['rmse'].mean(), color='#3498DB', linestyle='--', alpha=0.5)
    plt.axhline(y=test_df['rmse'].mean(), color='#F39C12', linestyle='--', alpha=0.5)
    plt.xlabel('Lần chạy')
    plt.ylabel('RMSE')
    plt.title('SO SÁNH RMSE TRAIN vs TEST QUA 10 LẦN', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 3: So sánh MAE train vs test qua 10 lần
    plt.subplot(2, 3, 3)
    plt.plot(runs, train_df['mae'], marker='o', linewidth=3, markersize=8, 
             label=f'Train MAE (TB: {train_df["mae"].mean():.3f})', color='#9B59B6')
    plt.plot(runs, test_df['mae'], marker='s', linewidth=3, markersize=8, 
             label=f'Test MAE (TB: {test_df["mae"].mean():.3f})', color='#E67E22')
    plt.axhline(y=train_df['mae'].mean(), color='#9B59B6', linestyle='--', alpha=0.5)
    plt.axhline(y=test_df['mae'].mean(), color='#E67E22', linestyle='--', alpha=0.5)
    plt.xlabel('Lần chạy')
    plt.ylabel('MAE')
    plt.title('SO SÁNH MAE TRAIN vs TEST QUA 10 LẦN', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 4: Phân bố chênh lệch R² (Overfitting)
    plt.subplot(2, 3, 4)
    r2_diff = train_df['r2'] - test_df['r2']
    plt.bar(runs, r2_diff, color=np.where(r2_diff > 0.1, '#E74C3C', '#2ECC71'), alpha=0.7)
    plt.axhline(y=r2_diff.mean(), color='red', linestyle='--', 
               label=f'Trung bình: {r2_diff.mean():.3f}')
    plt.xlabel('Lần chạy')
    plt.ylabel('Chênh lệch R² (Train - Test)')
    plt.title('ĐÁNH GIÁ OVERFITTING QUA 10 LẦN', fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Thêm giá trị trên các cột
    for i, v in enumerate(r2_diff):
        plt.text(i+1, v + 0.001, f'{v:.3f}', ha='center', va='bottom', 
                 fontsize=9, fontweight='bold', 
                 color='red' if v > 0.1 else 'green')
    
    # Biểu đồ 5: Hiệu suất theo bộ tham số (Heatmap style)
    plt.subplot(2, 3, 5)
    # Tạo dữ liệu cho heatmap
    param_names = [f"Lần {i+1}" for i in range(10)]
    metrics = ['R²', 'RMSE', 'MAE']
    performance_data = np.array([
        test_df['r2'].values,
        test_df['rmse'].values,
        test_df['mae'].values
    ])
    
    im = plt.imshow(performance_data, cmap='RdYlGn', aspect='auto')
    plt.xticks(range(10), param_names, rotation=45)
    plt.yticks(range(3), metrics)
    plt.title('MA TRẬN HIỆU SUẤT 10 LẦN CHẠY', fontweight='bold', fontsize=14)
    
    # Thêm giá trị vào ô - R² hiển thị 3 chữ số thập phân
    for i in range(3):
        for j in range(10):
            if i == 0:  # R²
                text = f'{performance_data[i, j]:.3f}'  # .3f thay vì .4f
                color = 'white' if performance_data[i, j] < 0.95 else 'black'
            else:  # RMSE, MAE
                text = f'{performance_data[i, j]:.2f}'
                color = 'white' if performance_data[i, j] > performance_data[i].mean() else 'black'
            plt.text(j, i, text, ha='center', va='center', 
                    fontweight='bold', color=color, fontsize=9)
    
    plt.colorbar(im, label='Hiệu suất (Xanh = Tốt, Đỏ = Kém)')
    
    # Biểu đồ 6: Tổng quan độ ổn định
    plt.subplot(2, 3, 6)
    metrics_std = [test_df['r2'].std(), test_df['rmse'].std(), test_df['mae'].std()]
    metrics_names = ['R²', 'RMSE', 'MAE']
    colors_std = ['#2ECC71' if std < 0.02 else '#F39C12' if std < 0.05 else '#E74C3C' for std in metrics_std]
    
    bars = plt.bar(metrics_names, metrics_std, color=colors_std, alpha=0.7, edgecolor='black')
    plt.ylabel('Độ lệch chuẩn')
    plt.title('ĐÁNH GIÁ ĐỘ ỔN ĐỊNH 10 LẦN CHẠY', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Thêm giá trị và đánh giá
    for bar, std, metric in zip(bars, metrics_std, metrics_names):
        if metric == 'R²':
            rating = "Rất ổn định" if std < 0.01 else "Ổn định" if std < 0.02 else "Biến động"
        else:
            rating = "Rất ổn định" if std < 0.5 else "Ổn định" if std < 1.0 else "Biến động"
        
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                 f'{std:.4f}\n{rating}', ha='center', va='bottom', 
                 fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(comparison_10_runs_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {comparison_10_runs_path}")

def create_detailed_runs_analysis(train_df, test_df, best_model_info):
    """Biểu đồ chi tiết từng lần chạy"""
    print("📊 8. Biểu đồ chi tiết từng lần chạy")
    
    # Định nghĩa các bộ tham số (giống trong model_trainer)
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
    
    detailed_runs_path = os.path.join('img', 'detailed_runs_analysis.png')
    plt.figure(figsize=(20, 15))
    
    # Biểu đồ 1: Hiệu suất theo max_depth
    plt.subplot(3, 3, 1)
    max_depths = [params.get('max_depth', 'None') for params in param_sets]
    test_r2_by_depth = test_df['r2'].values
    colors_depth = ['#2ECC71' if r2 > test_df['r2'].mean() else '#E74C3C' for r2 in test_r2_by_depth]
    
    bars = plt.bar(range(1, 11), test_r2_by_depth, color=colors_depth, alpha=0.7)
    plt.xlabel('Lần chạy')
    plt.ylabel('Test R²')
    plt.title('HIỆU SUẤT THEO LẦN CHẠY', fontweight='bold')
    plt.xticks(range(1, 11), [f'Lần {i}' for i in range(1, 11)], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Thêm giá trị R²
    for i, (bar, r2, depth) in enumerate(zip(bars, test_r2_by_depth, max_depths)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                 f'{r2:.3f}\n(depth: {depth})', ha='center', va='bottom', 
                 fontsize=8, fontweight='bold')
    
    # Biểu đồ 2: Phân tích tham số max_depth
    plt.subplot(3, 3, 2)
    unique_depths = list(set(max_depths))
    depth_performance = []
    for depth in unique_depths:
        indices = [i for i, d in enumerate(max_depths) if d == depth]
        avg_r2 = test_df.iloc[indices]['r2'].mean()
        depth_performance.append(avg_r2)
    
    plt.bar([str(d) for d in unique_depths], depth_performance, 
            color='#3498DB', alpha=0.7, edgecolor='black')
    plt.xlabel('Max Depth')
    plt.ylabel('R² Trung bình')
    plt.title('HIỆU SUẤT THEO MAX DEPTH', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (depth, perf) in enumerate(zip(unique_depths, depth_performance)):
        plt.text(i, perf + 0.002, f'{perf:.3f}', ha='center', va='bottom', 
                 fontweight='bold')
    
    # Biểu đồ 3: Phân tích min_samples_split
    plt.subplot(3, 3, 3)
    min_splits = [params.get('min_samples_split', 'N/A') for params in param_sets]
    split_groups = {}
    for i, split in enumerate(min_splits):
        if split not in split_groups:
            split_groups[split] = []
        split_groups[split].append(test_df.iloc[i]['r2'])
    
    split_means = {k: np.mean(v) for k, v in split_groups.items()}
    plt.bar([str(k) for k in split_means.keys()], split_means.values(),
            color='#9B59B6', alpha=0.7, edgecolor='black')
    plt.xlabel('Min Samples Split')
    plt.ylabel('R² Trung bình')
    plt.title('HIỆU SUẤT THEO MIN SAMPLES SPLIT', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 4: Tương quan giữa các metrics
    plt.subplot(3, 3, 4)
    plt.scatter(test_df['rmse'], test_df['r2'], s=100, alpha=0.7, 
               c=test_df['r2'], cmap='RdYlGn')
    plt.xlabel('RMSE')
    plt.ylabel('R²')
    plt.title('TƯƠNG QUAN RMSE vs R²', fontweight='bold')
    plt.colorbar(label='R² Score')
    plt.grid(True, alpha=0.3)
    
    # Thêm annotation cho các điểm
    for i, (rmse, r2) in enumerate(zip(test_df['rmse'], test_df['r2'])):
        plt.annotate(f'Lần {i+1}', (rmse, r2), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    # Biểu đồ 5: Phân bố R² của 10 lần chạy
    plt.subplot(3, 3, 5)
    plt.hist(test_df['r2'], bins=8, color='#2ECC71', alpha=0.7, edgecolor='black')
    plt.axvline(test_df['r2'].mean(), color='red', linestyle='--', 
               label=f'Trung bình: {test_df["r2"].mean():.3f}')
    plt.xlabel('R² Score')
    plt.ylabel('Tần suất')
    plt.title('PHÂN BỐ R² 10 LẦN CHẠY', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 6: Biểu đồ radar so sánh metrics 
    plt.subplot(3, 3, 6, projection='polar')
    
    # Chuẩn hóa dữ liệu cho radar chart
    metrics_to_plot = ['r2', 'rmse', 'mae', 'mape']
    normalized_data = []
    for metric in metrics_to_plot:
        if metric == 'r2':  # R² càng cao càng tốt
            normalized = test_df[metric] / test_df[metric].max()
        else:  # RMSE, MAE, MAPE càng thấp càng tốt
            normalized = 1 - (test_df[metric] / test_df[metric].max())
        normalized_data.append(normalized.values)
    
    normalized_data = np.array(normalized_data)
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng
    
    # Vẽ 3 lần chạy tốt nhất
    best_runs_indices = test_df['r2'].nlargest(3).index
    colors_best = ['#E74C3C', '#3498DB', '#2ECC71']
    
    for idx, color in zip(best_runs_indices, colors_best):
        values = normalized_data[:, idx].tolist()
        values += values[:1]  # Đóng vòng
        plt.plot(angles, values, 'o-', linewidth=2, label=f'Lần {idx+1}', color=color)
        plt.fill(angles, values, alpha=0.1, color=color)
    
    plt.thetagrids(np.degrees(angles[:-1]), metrics_to_plot)
    plt.title('RADAR CHART: 3 LẦN CHẠY TỐT NHẤT', fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.3, 1.1))
    
    # Biểu đồ 7: Trend hiệu suất theo thời gian
    plt.subplot(3, 3, 7)
    # Tính cumulative mean
    cumulative_mean = [test_df['r2'].iloc[:i+1].mean() for i in range(len(test_df))]
    cumulative_std = [test_df['r2'].iloc[:i+1].std() for i in range(len(test_df))]
    
    plt.plot(range(1, 11), cumulative_mean, marker='o', linewidth=2, 
             label='R² trung bình tích lũy', color='#E74C3C')
    plt.fill_between(range(1, 11), 
                     np.array(cumulative_mean) - np.array(cumulative_std),
                     np.array(cumulative_mean) + np.array(cumulative_std),
                     alpha=0.2, color='#E74C3C', label='±1 std')
    plt.xlabel('Số lần chạy')
    plt.ylabel('R² Trung bình tích lũy')
    plt.title('XU HƯỚNG HIỆU SUẤT THEO SỐ LẦN CHẠY', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 8: So sánh độ biến động
    plt.subplot(3, 3, 8)
    metrics_variability = {
        'R²': test_df['r2'].std(),
        'RMSE': test_df['rmse'].std(),
        'MAE': test_df['mae'].std(),
        'MAPE': test_df['mape'].std()
    }
    
    plt.bar(metrics_variability.keys(), metrics_variability.values(),
            color=['#2ECC71', '#3498DB', '#9B59B6', '#E67E22'], alpha=0.7)
    plt.ylabel('Độ lệch chuẩn')
    plt.title('ĐỘ BIẾN ĐỘNG CÁC CHỈ SỐ', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (metric, std) in enumerate(metrics_variability.items()):
        plt.text(i, std + 0.001, f'{std:.4f}', ha='center', va='bottom', 
                 fontweight='bold')
    
    # Biểu đồ 9: Tổng kết ranking
    plt.subplot(3, 3, 9)
    ranking = test_df['r2'].rank(ascending=False)
    colors_rank = ['gold' if rank == 1 else 'silver' if rank == 2 else 'brown' if rank == 3 else '#3498DB' 
                   for rank in ranking]
    
    plt.bar(range(1, 11), test_df['r2'], color=colors_rank, alpha=0.7)
    plt.xlabel('Lần chạy')
    plt.ylabel('R² Score')
    plt.title('RANKING 10 LẦN CHẠY', fontweight='bold')
    plt.xticks(range(1, 11), [f'#{int(r)}' for r in ranking], rotation=45)
    
    for i, (r2, rank) in enumerate(zip(test_df['r2'], ranking)):
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else ''
        plt.text(i+1, r2 + 0.002, f'{r2:.3f}\n{medal}', ha='center', va='bottom', 
                 fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(detailed_runs_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {detailed_runs_path}")

def plot_decision_tree(best_model_info):
    """Vẽ và lưu cây quyết định"""
    print(f"\n🌳 9. Vẽ và lưu cây quyết định")
    
    tree_path = os.path.join('img', 'decision_tree.png')
    plt.figure(figsize=(25, 12))
    plot_tree(
        best_model_info['model'],
        feature_names=['AT', 'V', 'AP', 'RH'],
        filled=True,
        rounded=True,
        impurity=True,
        fontsize=8,
        max_depth=3
    )
    plt.title(f"CÂY QUYẾT ĐỊNH - MÔ HÌNH TỐT NHẤT (Lần {best_model_info['run_id'] + 1})", 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(tree_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {tree_path}")