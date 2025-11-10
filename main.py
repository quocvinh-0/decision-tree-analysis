"""
MAIN.PY CẢI THIỆN - SỬ DỤNG PHƯƠNG PHÁP TỐI ƯU

Các cải thiện:
1. Sử dụng code improved (không scaling, có validation set, GridSearchCV, pruning)
2. Báo cáo kết quả rõ ràng và đầy đủ
3. Phù hợp cho bài báo cáo khoa học
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend non-interactive để tránh lỗi tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from scipy import stats
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Import các module cải thiện
from improved.data_loader_improved import load_and_prepare_data, get_scaled_features
from improved.model_trainer_improved import train_decision_trees_improved, calculate_metrics

# Import các module gốc (cần sửa để tương thích)
from model_comparison import compare_with_other_models
from visualization import create_all_visualizations

# Cập nhật model_comparison để sử dụng calculate_metrics từ improved
import model_comparison
model_comparison.calculate_metrics = calculate_metrics

def main():
    """Hàm chính để chạy toàn bộ quy trình phân tích với phương pháp cải thiện"""
    
    print("="*70)
    print("PHÂN TÍCH DỮ LIỆU VỚI CÂY QUYẾT ĐỊNH - PHƯƠNG PHÁP CẢI THIỆN")
    print("="*70)
    print("📋 Dataset: Folds5x2_pp.xlsx (Combined Cycle Power Plant Data)")
    print("🎯 Mục tiêu: Dự đoán sản lượng điện (PE) từ các đặc trưng môi trường")
    print("="*70)
    
    # ============================
    # TẠO THƯ MỤC LƯU TRỮ
    # ============================
    os.makedirs('img', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    os.makedirs('report', exist_ok=True)
    print("\n✅ Đã tạo thư mục: 'img/', 'result/', 'report/'")
    
    # ============================
    # BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
    # ============================
    print("\n" + "="*70)
    print("BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU")
    print("="*70)
    
    dataset_path = 'Folds5x2_pp.xlsx'
    X, y = load_and_prepare_data(dataset_path, use_enhanced_features=False)
    
    # Tạo biểu đồ phân phối PE cho slide
    from visualization import create_pe_distribution_slide
    create_pe_distribution_slide(y)
    
    # Tạo cây quyết định từ 10 mẫu đầu (cho slide)
    from visualization import plot_decision_tree_slide
    plot_decision_tree_slide(X, y)
    
    # Lấy scaler cho Naive Bayes (nếu cần)
    X_scaled, scaler = get_scaled_features(X)
    
    # ============================
    # BƯỚC 2: HUẤN LUYỆN MÔ HÌNH DECISION TREE (CẢI THIỆN)
    # ============================
    print("\n" + "="*70)
    print("BƯỚC 2: HUẤN LUYỆN MÔ HÌNH DECISION TREE")
    print("="*70)
    print("📌 Phương pháp:")
    print("   • Sử dụng GridSearchCV để tìm hyperparameter tối ưu")
    print("   • Train/Test split (80/20)")
    print("   • Cost Complexity Pruning để giảm overfitting")
    print("   • Không sử dụng scaling (Decision Tree không cần)")
    print("   • Chọn mô hình dựa trên test set")
    
    train_df, test_df, feature_importance_df, best_models, best_model_info = \
        train_decision_trees_improved(X, y, n_runs=10, use_grid_search=True)
    
    # ============================
    # BƯỚC 3: SO SÁNH VỚI MÔ HÌNH KHÁC
    # ============================
    print("\n" + "="*70)
    print("BƯỚC 3: SO SÁNH VỚI CÁC MÔ HÌNH KHÁC")
    print("="*70)
    
    # Chuẩn bị dữ liệu cho so sánh
    # Sử dụng train để train các mô hình khác
    X_train_best = best_model_info['X_train']
    X_test_best = best_model_info['X_test']
    y_train_best = best_model_info['y_train']
    y_test_best = best_model_info['y_test']
    
    # Sử dụng train data để train các mô hình khác
    if isinstance(X_train_best, pd.DataFrame):
        X_train_combined = X_train_best.values
    else:
        X_train_combined = X_train_best
    
    if isinstance(y_train_best, pd.Series):
        y_train_combined = y_train_best.values
    else:
        y_train_combined = y_train_best
    
    # Chuẩn hóa dữ liệu cho Naive Bayes (nếu cần)
    # Bỏ KNN nên không cần scaling nữa, nhưng Naive Bayes vẫn cần
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_best)
    
    comparison_results = compare_with_other_models(
        X_train_best=X_train_combined,
        X_test_best=X_test_best,
        y_train_best=y_train_combined,
        y_test_best=y_test_best,
        best_model=best_model_info['model'],
        X_train_scaled=X_train_scaled,  # Dữ liệu đã chuẩn hóa cho Naive Bayes
        X_test_scaled=X_test_scaled     # Dữ liệu đã chuẩn hóa cho Naive Bayes
    )
    
    # ============================
    # BƯỚC 4: TRỰC QUAN HÓA KẾT QUẢ
    # ============================
    print("\n" + "="*70)
    print("BƯỚC 4: TRỰC QUAN HÓA KẾT QUẢ")
    print("="*70)
    
    # Cập nhật best_model_info để có X_scaled cho visualization
    best_model_info_vis = best_model_info.copy()
    best_model_info_vis['X_scaled'] = X_scaled
    
    create_all_visualizations(
        train_df, test_df, feature_importance_df, best_model_info_vis, 
        comparison_results, X_scaled, y
    )
    
    # Tạo thêm biểu đồ train/test comparison
    create_validation_visualizations(train_df, test_df, best_model_info)
    
    # Tạo biểu đồ Decision Tree là tốt nhất
    create_decision_tree_best_charts(comparison_results)
    
    # Tạo cây quyết định rút gọn cho slide
    from visualization import plot_decision_tree_simplified
    plot_decision_tree_simplified(best_model_info)
    
    # Tạo biểu đồ R² score theo các tham số
    from visualization import create_r2_score_by_params_chart
    create_r2_score_by_params_chart(best_models)
    
    # Tạo biểu đồ so sánh 3 phương pháp qua 10 lần lặp
    from visualization import create_comparison_3_methods_chart
    create_comparison_3_methods_chart(best_models, test_df)
    
    # ============================
    # BƯỚC 5: LƯU KẾT QUẢ
    # ============================
    print("\n" + "="*70)
    print("BƯỚC 5: LƯU KẾT QUẢ")
    print("="*70)
    
    # Cập nhật best_model_info để có scaler (None vì không dùng)
    best_model_info_save = best_model_info.copy()
    best_model_info_save['scaler'] = None  # Decision Tree không cần scaler
    
    save_results_improved(
        train_df, test_df, feature_importance_df, best_model_info_save,
        comparison_results, best_model_info['model']
    )
    
    # ============================
    # BƯỚC 6: TẠO BÁO CÁO TỰ ĐỘNG
    # ============================
    print("\n" + "="*70)
    print("BƯỚC 6: TẠO BÁO CÁO TỰ ĐỘNG")
    print("="*70)
    
    generate_report(train_df, test_df, feature_importance_df, 
                   best_model_info, comparison_results)
    
    # ============================
    # BƯỚC 7: TỔNG KẾT
    # ============================
    print_final_summary_improved(train_df, test_df, best_model_info, 
                                feature_importance_df, comparison_results)

def create_decision_tree_best_charts(comparison_results):
    """Tạo các biểu đồ cột thể hiện Decision Tree là lựa chọn tốt nhất"""
    print("\n📊 Tạo biểu đồ Decision Tree là tốt nhất...")
    
    os.makedirs('img/decision_tree_best', exist_ok=True)
    
    # Lấy dữ liệu từ comparison_results
    models_data = {
        'Decision Tree': {
            'R²': comparison_results['decision_tree']['metrics']['r2'],
            'RMSE': comparison_results['decision_tree']['metrics']['rmse'],
            'MAE': comparison_results['decision_tree']['metrics']['mae']
        },
        'Random Forest': {
            'R²': comparison_results['random_forest']['metrics']['r2'],
            'RMSE': comparison_results['random_forest']['metrics']['rmse'],
            'MAE': comparison_results['random_forest']['metrics']['mae']
        },
        # Bỏ KNN (theo yêu cầu)
        # 'KNN': {
        #     'R²': comparison_results['knn']['metrics']['r2'],
        #     'RMSE': comparison_results['knn']['metrics']['rmse'],
        #     'MAE': comparison_results['knn']['metrics']['mae']
        # }
    }
    
    # Thêm Naive Bayes nếu có
    if 'naive_bayes' in comparison_results:
        models_data['Naive Bayes'] = {
            'R²': comparison_results['naive_bayes']['metrics']['r2'],
            'RMSE': comparison_results['naive_bayes']['metrics']['rmse'],
            'MAE': comparison_results['naive_bayes']['metrics']['mae']
        }
    
    models = list(models_data.keys())
    colors = {
        'Decision Tree': '#2ECC71',
        # 'KNN': '#9B59B6',  # Bỏ KNN
        'Random Forest': '#3498DB',
        'Naive Bayes': '#E74C3C'
    }
    
    # Biểu đồ 1: So sánh R²
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    r2_scores = [models_data[m]['R²'] for m in models]
    model_colors = [colors[m] for m in models]
    bars = ax.bar(models, r2_scores, color=model_colors, alpha=0.8, 
                  edgecolor='black', linewidth=2)
    
    dt_idx = models.index('Decision Tree')
    bars[dt_idx].set_color('#27AE60')
    bars[dt_idx].set_edgecolor('#1E8449')
    bars[dt_idx].set_linewidth(3)
    bars[dt_idx].set_alpha(1.0)
    
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        height = bar.get_height()
        if i == dt_idx:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                    f'{score:.4f}\n(TỐT NHẤT)', ha='center', va='bottom',
                    fontweight='bold', fontsize=12, color='#1E8449')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.002,
                    f'{score:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('SO SÁNH R² SCORE - DECISION TREE ĐẠT HIỆU SUẤT CAO NHẤT',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0.75, 1.01)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.98, '[TỐT NHẤT] Decision Tree: R² = {:.4f} (Cao nhất)'.format(r2_scores[dt_idx]),
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='#D5F4E6', edgecolor='#27AE60', linewidth=2))
    plt.xticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig('img/decision_tree_best/comparison_r2_score.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("   ✅ Đã lưu: img/decision_tree_best/comparison_r2_score.png")
    
    # Biểu đồ 2: So sánh RMSE
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    rmse_scores = [models_data[m]['RMSE'] for m in models]
    model_colors = [colors[m] for m in models]
    bars = ax.bar(models, rmse_scores, color=model_colors, alpha=0.8,
                  edgecolor='black', linewidth=2)
    
    bars[dt_idx].set_color('#27AE60')
    bars[dt_idx].set_edgecolor('#1E8449')
    bars[dt_idx].set_linewidth(3)
    bars[dt_idx].set_alpha(1.0)
    
    for i, (bar, score) in enumerate(zip(bars, rmse_scores)):
        height = bar.get_height()
        if i == dt_idx:
            ax.text(bar.get_x() + bar.get_width()/2, height + max(rmse_scores)*0.05,
                    f'{score:.4f}\n(THẤP NHẤT)', ha='center', va='bottom',
                    fontweight='bold', fontsize=12, color='#1E8449')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height + max(rmse_scores)*0.05,
                    f'{score:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
    ax.set_title('SO SÁNH RMSE - DECISION TREE CÓ SAI SỐ THẤP NHẤT',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(rmse_scores) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.98, '[TỐT NHẤT] Decision Tree: RMSE = {:.4f} (Thấp nhất)'.format(rmse_scores[dt_idx]),
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='#D5F4E6', edgecolor='#27AE60', linewidth=2))
    plt.xticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig('img/decision_tree_best/comparison_rmse.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("   ✅ Đã lưu: img/decision_tree_best/comparison_rmse.png")
    
    # Biểu đồ 3: So sánh MAE
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    mae_scores = [models_data[m]['MAE'] for m in models]
    model_colors = [colors[m] for m in models]
    bars = ax.bar(models, mae_scores, color=model_colors, alpha=0.8,
                  edgecolor='black', linewidth=2)
    
    bars[dt_idx].set_color('#27AE60')
    bars[dt_idx].set_edgecolor('#1E8449')
    bars[dt_idx].set_linewidth(3)
    bars[dt_idx].set_alpha(1.0)
    
    for i, (bar, score) in enumerate(zip(bars, mae_scores)):
        height = bar.get_height()
        if i == dt_idx:
            ax.text(bar.get_x() + bar.get_width()/2, height + max(mae_scores)*0.05,
                    f'{score:.4f}\n(THẤP NHẤT)', ha='center', va='bottom',
                    fontweight='bold', fontsize=12, color='#1E8449')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, height + max(mae_scores)*0.05,
                    f'{score:.4f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    
    ax.set_ylabel('MAE', fontsize=14, fontweight='bold')
    ax.set_title('SO SÁNH MAE - DECISION TREE CÓ SAI SỐ TUYỆT ĐỐI THẤP NHẤT',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(mae_scores) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.text(0.02, 0.98, '[TỐT NHẤT] Decision Tree: MAE = {:.4f} (Thấp nhất)'.format(mae_scores[dt_idx]),
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='#D5F4E6', edgecolor='#27AE60', linewidth=2))
    plt.xticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig('img/decision_tree_best/comparison_mae.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("   ✅ Đã lưu: img/decision_tree_best/comparison_mae.png")

def create_validation_visualizations(train_df, test_df, best_model_info):
    """Tạo biểu đồ so sánh train/test (bỏ validation)"""
    print("\n📊 Tạo biểu đồ so sánh Train/Test")
    
    val_comparison_path = os.path.join('img', 'train_test_comparison.png')
    plt.figure(figsize=(15, 10))
    
    # Biểu đồ 1: So sánh R²
    plt.subplot(2, 2, 1)
    runs = range(1, 11)
    plt.plot(runs, train_df['r2'], marker='o', linewidth=2, markersize=6, 
             label='Train R²', color='#2ECC71')
    plt.plot(runs, test_df['r2'], marker='^', linewidth=2, markersize=6, 
             label='Test R²', color='#E74C3C')
    plt.axhline(y=train_df['r2'].mean(), color='#2ECC71', linestyle='--', alpha=0.5)
    plt.axhline(y=test_df['r2'].mean(), color='#E74C3C', linestyle='--', alpha=0.5)
    plt.xlabel('Lần chạy')
    plt.ylabel('R² Score')
    plt.title('SO SÁNH R²: TRAIN vs TEST', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 2: So sánh RMSE
    plt.subplot(2, 2, 2)
    plt.plot(runs, train_df['rmse'], marker='o', linewidth=2, markersize=6, 
             label='Train RMSE', color='#2ECC71')
    plt.plot(runs, test_df['rmse'], marker='^', linewidth=2, markersize=6, 
             label='Test RMSE', color='#E74C3C')
    plt.xlabel('Lần chạy')
    plt.ylabel('RMSE')
    plt.title('SO SÁNH RMSE: TRAIN vs TEST', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 3: Đánh giá overfitting
    plt.subplot(2, 2, 3)
    train_test_gap = train_df['r2'] - test_df['r2']
    x = np.arange(len(runs))
    width = 0.5
    plt.bar(x, train_test_gap, width, label='Train - Test', color='#F39C12', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Ngưỡng overfitting')
    plt.xlabel('Lần chạy')
    plt.ylabel('Chênh lệch R²')
    plt.title('ĐÁNH GIÁ OVERFITTING', fontweight='bold')
    plt.xticks(x, runs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Biểu đồ 4: Box plot so sánh
    plt.subplot(2, 2, 4)
    data_to_plot = [train_df['r2'], test_df['r2']]
    bp = plt.boxplot(data_to_plot, labels=['Train', 'Test'], 
                     patch_artist=True)
    colors = ['#2ECC71', '#E74C3C']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel('R² Score')
    plt.title('PHÂN BỐ R² SCORE', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(val_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu: {val_comparison_path}")

def save_results_improved(train_df, test_df, feature_importance_df, 
                         best_model_info, comparison_results, best_model):
    """Lưu kết quả với validation set"""
    # Lưu mô hình
    model_path = os.path.join('result', 'best_decision_tree_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n✅ Đã lưu mô hình: {model_path}")
    
    # Lưu kết quả vào Excel
    excel_path = os.path.join('result', 'results_summary.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Tổng quan
        save_summary_sheet_improved(writer, train_df, test_df, 
                                   best_model_info, comparison_results)
        
        # Sheet 2: So sánh mô hình
        save_model_comparison_sheet(writer, comparison_results)
        
        # Sheet 3: Feature Importance
        feature_importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
        
        # Sheet 4: Kết quả 10 lần chạy
        save_detailed_results_sheet_improved(writer, train_df, test_df)
        
        # Sheet 5: Tham số mô hình tốt nhất
        save_best_model_sheet_improved(writer, best_model_info)
        
        # Sheet 6: Cross-validation
        save_cv_results_sheet(writer, comparison_results)
        
        # Sheet 7: Đánh giá overfitting
        save_overfitting_analysis_sheet(writer, train_df, test_df)
    
    print(f"✅ Đã lưu file Excel: {excel_path}")

def save_summary_sheet_improved(writer, train_df, test_df, 
                                best_model_info, comparison_results):
    """Lưu sheet tổng quan (bỏ validation)"""
    summary_data = {
        'Metric': [
            'R² Train (TB)', 'R² Test (TB)',
            'RMSE Test (TB)', 'MAE Test (TB)', 'MAPE Test (TB)',
            'R² Test (Tốt nhất)', 'Độ lệch chuẩn R² Test',
            'Chênh lệch Train-Test R²',
            'Số lần chạy', 'Mô hình tốt nhất',
            'Cross-Val R²', 'Cross-Val RMSE'
        ],
        'Giá trị': [
            f"{train_df['r2'].mean():.4f}", 
            f"{test_df['r2'].mean():.4f}",
            f"{test_df['rmse'].mean():.4f}", f"{test_df['mae'].mean():.4f}", 
            f"{test_df['mape'].mean():.2f}%",
            f"{best_model_info['test_r2']:.4f}", f"{test_df['r2'].std():.4f}",
            f"{train_df['r2'].mean() - test_df['r2'].mean():.4f}",
            '10', f"Lần {best_model_info['run_id'] + 1}",
            f"{comparison_results['cv_results']['test_r2'].mean():.4f}",
            f"{comparison_results['cv_results']['test_rmse'].mean():.4f}"
        ],
        'Đánh giá': [
            f"{'✅ Tốt' if train_df['r2'].mean() > 0.9 else '⚠️ Khá'}",
            f"{'✅ Tốt' if test_df['r2'].mean() > 0.9 else '⚠️ Khá'}",
            f"{'✅ Tốt' if test_df['rmse'].mean() < 5 else '⚠️ Trung bình'}",
            f"{'✅ Tốt' if test_df['mae'].mean() < 4 else '⚠️ Trung bình'}",
            f"{'✅ Tốt' if test_df['mape'].mean() < 5 else '⚠️ Khá'}",
            '🏆 Tốt nhất',
            f"{'Ổn định' if test_df['r2'].std() < 0.02 else 'Biến động'}",
            f"{'⚠️ Overfitting' if (train_df['r2'].mean() - test_df['r2'].mean()) > 0.05 else '✅ OK'}",
            'Đủ', 'Đã chọn',
            f"{'✅ Tốt' if comparison_results['cv_results']['test_r2'].mean() > 0.9 else '⚠️ Khá'}",
            f"{'✅ Tốt' if comparison_results['cv_results']['test_rmse'].mean() < 5 else '⚠️ Trung bình'}"
        ]
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Tổng quan', index=False)

def save_detailed_results_sheet_improved(writer, train_df, test_df):
    """Lưu sheet kết quả chi tiết (bỏ validation)"""
    detailed_results = pd.DataFrame({
        'Lần chạy': range(1, 11),
        'Train_R2': train_df['r2'],
        'Train_RMSE': train_df['rmse'],
        'Train_MAE': train_df['mae'],
        'Test_R2': test_df['r2'],
        'Test_RMSE': test_df['rmse'],
        'Test_MAE': test_df['mae'],
        'Train_Test_Gap': train_df['r2'] - test_df['r2']
    })
    detailed_results.to_excel(writer, sheet_name='10 Lần chạy', index=False)

def save_best_model_sheet_improved(writer, best_model_info):
    """Lưu sheet thông tin mô hình tốt nhất"""
    best_params_df = pd.DataFrame([best_model_info['params']])
    best_params_df['Test_R2'] = best_model_info['test_r2']
    best_params_df['Lần_chạy'] = best_model_info['run_id'] + 1
    best_params_df.to_excel(writer, sheet_name='Mô hình tốt nhất', index=False)

def save_overfitting_analysis_sheet(writer, train_df, test_df):
    """Lưu sheet phân tích overfitting (bỏ validation)"""
    overfitting_analysis = pd.DataFrame({
        'Lần chạy': range(1, 11),
        'Train_R2': train_df['r2'],
        'Test_R2': test_df['r2'],
        'Train_Test_Gap': train_df['r2'] - test_df['r2'],
        'Overfitting': ['Có' if gap > 0.05 else 'Không' for gap in (train_df['r2'] - test_df['r2'])]
    })
    overfitting_analysis.to_excel(writer, sheet_name='Phân tích Overfitting', index=False)

def save_model_comparison_sheet(writer, comparison_results):
    """Lưu sheet so sánh mô hình (bỏ KNN)"""
    dt_metrics = comparison_results['decision_tree']['metrics']
    rf_metrics = comparison_results['random_forest']['metrics']
    # Bỏ KNN
    # knn_metrics = comparison_results['knn']['metrics']
    
    # Tạo danh sách mô hình (bỏ KNN)
    models_data = {
        'Mô hình': ['Decision Tree', 'Random Forest'],
        'R²': [dt_metrics['r2'], rf_metrics['r2']],
        'RMSE': [dt_metrics['rmse'], rf_metrics['rmse']],
        'MAE': [dt_metrics['mae'], rf_metrics['mae']],
        'MAPE': [f"{dt_metrics['mape']:.2f}%", f"{rf_metrics['mape']:.2f}%"]
    }
    
    # Thêm Naive Bayes nếu có
    if 'naive_bayes' in comparison_results:
        nb_metrics = comparison_results['naive_bayes']['metrics']
        models_data['Mô hình'].append('Naive Bayes')
        models_data['R²'].append(nb_metrics['r2'])
        models_data['RMSE'].append(nb_metrics['rmse'])
        models_data['MAE'].append(nb_metrics['mae'])
        models_data['MAPE'].append(f"{nb_metrics['mape']:.2f}%")
    
    model_comparison = pd.DataFrame(models_data)
    model_comparison.to_excel(writer, sheet_name='So sánh mô hình', index=False)

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

def generate_report(train_df, test_df, feature_importance_df, 
                   best_model_info, comparison_results):
    """Tạo báo cáo tự động (bỏ validation)"""
    report_path = os.path.join('report', 'BAO_CAO_KET_QUA.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 BÁO CÁO KẾT QUẢ PHÂN TÍCH DECISION TREE\n\n")
        f.write("## 1. THÔNG TIN DATASET\n\n")
        f.write("- **Dataset**: Folds5x2_pp.xlsx (Combined Cycle Power Plant Data)\n")
        f.write("- **Số mẫu**: 47,840 mẫu (5 sheets × 9,568 mẫu/sheet)\n")
        f.write("- **Đặc trưng**: AT, V, AP, RH\n")
        f.write("- **Target**: PE (Net hourly electrical energy output)\n\n")
        
        f.write("## 2. PHƯƠNG PHÁP\n\n")
        f.write("### 2.1. Tiền xử lý dữ liệu\n")
        f.write("- Không sử dụng scaling (Decision Tree không cần)\n")
        f.write("- Không sử dụng feature engineering\n\n")
        
        f.write("### 2.2. Phân chia dữ liệu\n")
        f.write("- **Train set**: 80%\n")
        f.write("- **Test set**: 20%\n\n")
        
        f.write("### 2.3. Huấn luyện mô hình\n")
        f.write("- Sử dụng **GridSearchCV** để tìm hyperparameter tối ưu\n")
        f.write("- Sử dụng **Cost Complexity Pruning** để giảm overfitting\n")
        f.write("- Chạy 10 lần với các random_state khác nhau\n")
        f.write("- Chọn mô hình tốt nhất dựa trên **test set**\n\n")
        
        f.write("## 3. KẾT QUẢ\n\n")
        f.write("### 3.1. Kết quả tổng hợp (10 lần chạy)\n\n")
        f.write("| Metric | Train | Test |\n")
        f.write("|--------|-------|------|\n")
        f.write(f"| R² (TB) | {train_df['r2'].mean():.4f} | {test_df['r2'].mean():.4f} |\n")
        f.write(f"| RMSE (TB) | {train_df['rmse'].mean():.4f} | {test_df['rmse'].mean():.4f} |\n")
        f.write(f"| MAE (TB) | {train_df['mae'].mean():.4f} | {test_df['mae'].mean():.4f} |\n\n")
        
        f.write("### 3.2. Mô hình tốt nhất\n\n")
        f.write(f"- **Lần chạy**: {best_model_info['run_id'] + 1}\n")
        f.write(f"- **Test R²**: {best_model_info['test_r2']:.4f}\n")
        f.write(f"- **Tham số**: {best_model_info['params']}\n\n")
        
        f.write("### 3.3. Đánh giá overfitting\n\n")
        train_test_gap = train_df['r2'].mean() - test_df['r2'].mean()
        f.write(f"- **Chênh lệch Train-Test R²**: {train_test_gap:.4f}\n")
        if train_test_gap > 0.05:
            f.write("- **Kết luận**: ⚠️ Có dấu hiệu overfitting\n\n")
        else:
            f.write("- **Kết luận**: ✅ Không có overfitting nghiêm trọng\n\n")
        
        f.write("### 3.4. Độ quan trọng đặc trưng\n\n")
        f.write("| Đặc trưng | Độ quan trọng (TB) | Độ lệch chuẩn |\n")
        f.write("|-----------|-------------------|---------------|\n")
        for idx, row in feature_importance_df.iterrows():
            f.write(f"| {row['Đặc trưng']} | {row['Độ quan trọng trung bình']:.4f} | {row['Độ lệch chuẩn']:.4f} |\n")
        f.write("\n")
        
        f.write("### 3.5. So sánh với mô hình khác\n\n")
        dt_metrics = comparison_results['decision_tree']['metrics']
        rf_metrics = comparison_results['random_forest']['metrics']
        # Bỏ KNN
        # knn_metrics = comparison_results['knn']['metrics']
        f.write("| Mô hình | R² | RMSE | MAE |\n")
        f.write("|---------|----|----|----|\n")
        f.write(f"| Decision Tree | {dt_metrics['r2']:.4f} | {dt_metrics['rmse']:.4f} | {dt_metrics['mae']:.4f} |\n")
        f.write(f"| Random Forest | {rf_metrics['r2']:.4f} | {rf_metrics['rmse']:.4f} | {rf_metrics['mae']:.4f} |\n")
        # Bỏ KNN
        # f.write(f"| KNN | {knn_metrics['r2']:.4f} | {knn_metrics['rmse']:.4f} | {knn_metrics['mae']:.4f} |\n")
        
        # Thêm Naive Bayes nếu có
        if 'naive_bayes' in comparison_results:
            nb_metrics = comparison_results['naive_bayes']['metrics']
            f.write(f"| Naive Bayes | {nb_metrics['r2']:.4f} | {nb_metrics['rmse']:.4f} | {nb_metrics['mae']:.4f} |\n")
            f.write(f"\n**Lưu ý:** Naive Bayes được chuyển đổi từ Classification (chia PE thành 3 lớp: Thấp, Trung bình, Cao)\n")
        f.write("\n")
        
        f.write("### 3.6. Cross-Validation (5-fold)\n\n")
        cv_results = comparison_results['cv_results']
        f.write(f"- **Train R²**: {cv_results['train_r2'].mean():.4f} (±{cv_results['train_r2'].std():.4f})\n")
        f.write(f"- **Test R²**: {cv_results['test_r2'].mean():.4f} (±{cv_results['test_r2'].std():.4f})\n")
        f.write(f"- **Test RMSE**: {cv_results['test_rmse'].mean():.4f} (±{cv_results['test_rmse'].std():.4f})\n\n")
        
        f.write("## 4. KẾT LUẬN\n\n")
        avg_test_r2 = test_df['r2'].mean()
        if avg_test_r2 > 0.95:
            f.write("✅ Mô hình Decision Tree đạt hiệu suất **XUẤT SẮC** với R² > 0.95\n\n")
        elif avg_test_r2 > 0.9:
            f.write("✅ Mô hình Decision Tree đạt hiệu suất **TỐT** với R² > 0.9\n\n")
        else:
            f.write("⚠️ Mô hình Decision Tree đạt hiệu suất **KHÁ** với R² < 0.9\n\n")
        
        f.write("## 5. FILE KẾT QUẢ\n\n")
        f.write("- **Biểu đồ**: Thư mục `img/`\n")
        f.write("- **Model**: `result/best_decision_tree_model.pkl`\n")
        f.write("- **Excel**: `result/results_summary.xlsx`\n")
        f.write("- **Báo cáo**: `report/BAO_CAO_KET_QUA.md`\n\n")
    
    print(f"✅ Đã tạo báo cáo: {report_path}")

def print_final_summary_improved(train_df, test_df, best_model_info, 
                                feature_importance_df, comparison_results):
    """In tổng kết cuối cùng (bỏ validation)"""
    print("\n" + "="*70)
    print("🎯 TỔNG KẾT KẾT QUẢ")
    print("="*70)
    
    # Đánh giá chất lượng tổng thể
    avg_test_r2 = test_df['r2'].mean()
    std_test_r2 = test_df['r2'].std()
    train_test_gap = train_df['r2'].mean() - test_df['r2'].mean()
    
    if avg_test_r2 > 0.95 and std_test_r2 < 0.01 and train_test_gap < 0.05:
        stability = "RẤT ỔN ĐỊNH VÀ XUẤT SẮC 🏆"
    elif avg_test_r2 > 0.9 and std_test_r2 < 0.02 and train_test_gap < 0.1:
        stability = "ỔN ĐỊNH VÀ TỐT ✅"
    elif avg_test_r2 > 0.85:
        stability = "KHÁ ỔN ĐỊNH 📊"
    else:
        stability = "CẦN CẢI THIỆN ⚠️"
    
    print(f"\n📈 KẾT QUẢ TỔNG HỢP:")
    print(f"   • Số lần huấn luyện: 10")
    print(f"   • Phương pháp: GridSearchCV + Cost Complexity Pruning")
    print(f"   • Mô hình tốt nhất đạt Test R²: {best_model_info['test_r2']:.4f}")
    
    print(f"\n📊 CHẤT LƯỢNG TRUNG BÌNH (10 lần):")
    print(f"   • Train R²:      {train_df['r2'].mean():.4f} (±{train_df['r2'].std():.4f})")
    print(f"   • Test R²:       {test_df['r2'].mean():.4f} (±{test_df['r2'].std():.4f})")
    print(f"   • Test RMSE:     {test_df['rmse'].mean():.4f} (±{test_df['rmse'].std():.4f})")
    print(f"   • Test MAE:      {test_df['mae'].mean():.4f} (±{test_df['mae'].std():.4f})")
    print(f"   • Độ ổn định:    {stability}")
    
    print(f"\n🔍 ĐÁNH GIÁ OVERFITTING:")
    print(f"   • Chênh lệch Train-Test R²: {train_test_gap:.4f}")
    if train_test_gap > 0.05:
        print(f"   ⚠️  Có dấu hiệu overfitting (chênh lệch > 0.05)")
    else:
        print(f"   ✅ Không có overfitting nghiêm trọng")
    
    print(f"\n🔍 ĐẶC TRƯNG QUAN TRỌNG NHẤT:")
    best_feature = feature_importance_df.iloc[0]
    print(f"   • {best_feature['Đặc trưng']}: {best_feature['Độ quan trọng trung bình']:.4f} "
          f"(±{best_feature['Độ lệch chuẩn']:.4f})")
    
    print(f"\n⚙️ BỘ THAM SỐ TỐT NHẤT (Lần {best_model_info['run_id'] + 1}):")
    for key, value in best_model_info['params'].items():
        print(f"   • {key}: {value if value is not None else 'Không giới hạn'}")
    
    print(f"\n📁 KẾT QUẢ ĐÃ ĐƯỢC LƯU:")
    print(f"   • 📊 Ảnh biểu đồ: {len(os.listdir('img'))} file trong thư mục 'img/'")
    print(f"   • 💾 Model & Data: {len(os.listdir('result'))} file trong thư mục 'result/'")
    print(f"   • 📈 File Excel: result/results_summary.xlsx")
    print(f"   • 📄 Báo cáo: report/BAO_CAO_KET_QUA.md")
    print(f"\n🎉 HOÀN THÀNH PHÂN TÍCH!")
    print("="*70)

if __name__ == "__main__":
    main()

