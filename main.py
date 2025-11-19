import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Import các module tùy chỉnh
from data_loader import load_and_prepare_data
from model_trainer import train_decision_trees, calculate_metrics
from model_comparison import compare_with_other_models
from visualization import create_all_visualizations
from results_saver import save_results

def main():
    """Hàm chính để chạy toàn bộ quy trình phân tích"""
    # TẠO THƯ MỤC LƯU TRỮ
    os.makedirs('img', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    print("Đã tạo thư mục 'img' và 'result'")
    
    
    # BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
    
    print("PHÂN TÍCH DỮ LIỆU VỚI CÂY QUYẾT ĐỊNH")
    
    X, y, X_scaled = load_and_prepare_data('Folds5x2_pp.xlsx', use_enhanced_features=False)
    
    # BƯỚC 2: HUẤN LUYỆN 10 MÔ HÌNH CÂY QUYẾT ĐỊNH
    
    print("\nHUẤN LUYỆN 10 LẦN VÀ TÍNH TRUNG BÌNH")
    
    train_df, test_df, feature_importance_df, best_models, best_model_info = train_decision_trees(
        X, y, X_scaled, n_runs=10
    )
    
    # BƯỚC 3: SO SÁNH VỚI MÔ HÌNH KHÁC
    
    print("\nSO SÁNH VỚI CÁC MÔ HÌNH KHÁC")
    
    comparison_results = compare_with_other_models(
        X_train_best=best_model_info['X_train'],
        X_test_best=best_model_info['X_test'],
        y_train_best=best_model_info['y_train'], 
        y_test_best=best_model_info['y_test'],
        best_model=best_model_info['model']
    )
    
    
    # BƯỚC 4: TRỰC QUAN HÓA KẾT QUẢ
    
    print("\nBẮT ĐẦU TRỰC QUAN HÓA KẾT QUẢ")
    
    create_all_visualizations(
        train_df, test_df, feature_importance_df, best_model_info, 
        comparison_results, X_scaled, y
    )
    
    
    # BƯỚC 5: LƯU KẾT QUẢ
    
    print("\nLƯU KẾT QUẢ VÀO FILE")
    
    save_results(
        train_df, test_df, feature_importance_df, best_model_info,
        comparison_results, best_model_info['model']
    )
    # BƯỚC 6: TỔNG KẾT
    
    print_final_summary(test_df, best_model_info, feature_importance_df)

def print_final_summary(test_df, best_model_info, feature_importance_df):
    # Đánh giá chất lượng tổng thể
    avg_test_r2 = test_df['r2'].mean()
    std_test_r2 = test_df['r2'].std()
    
    if avg_test_r2 > 0.95 and std_test_r2 < 0.01:
        stability = "Rất ổn định và xuất sắc"
    elif avg_test_r2 > 0.9 and std_test_r2 < 0.02:
        stability = "Ổn định và tốt"
    elif avg_test_r2 > 0.85:
        stability = "Khá ổn định"
    else:
        stability = "Cần cải thiện"
    
    print(f"\nKẾT QUẢ TỔNG HỢP:")
    print(f"   - Số lần huấn luyện: 10")
    print(f"   - Số bộ tham số khác nhau: 10")
    print(f"   - Mô hình tốt nhất đạt Test R²: {best_model_info['test_r2']:.4f}")
    
    print(f"\nCHẤT LƯỢNG TRUNG BÌNH (10 lần):")
    print(f"   - R² trung bình: {avg_test_r2:.4f} (±{std_test_r2:.4f})")
    print(f"   - RMSE trung bình: {test_df['rmse'].mean():.4f} (±{test_df['rmse'].std():.4f})")
    print(f"   - MAE trung bình: {test_df['mae'].mean():.4f} (±{test_df['mae'].std():.4f})")
    print(f"   - Median AE trung bình: {test_df['medae'].mean():.4f} (±{test_df['medae'].std():.4f})")
    print(f"   - Max Error trung bình: {test_df['max_error'].mean():.4f} (±{test_df['max_error'].std():.4f})")
    print(f"   - MAPE trung bình: {test_df['mape'].mean():.2f}% (±{test_df['mape'].std():.2f}%)")
    print(f"   - Explained variance trung bình: {test_df['explained_variance'].mean():.4f} (±{test_df['explained_variance'].std():.4f})")
    print(f"   - Độ ổn định: {stability}")
    
    print(f"\nĐẶC TRƯNG QUAN TRỌNG NHẤT:")
    best_feature = feature_importance_df.iloc[0]
    print(f"   - {best_feature['Đặc trưng']}: {best_feature['Độ quan trọng trung bình']:.4f} "
          f"(±{best_feature['Độ lệch chuẩn']:.4f})")
    
    print(f"\nBỘ THAM SỐ TỐT NHẤT (Lần {best_model_info['run_id'] + 1}):")
    for key, value in best_model_info['params'].items():
        print(f"   - {key}: {value if value is not None else 'Không giới hạn'}")
    
    print(f"\nKẾT QUẢ ĐÃ ĐƯỢC LƯU:")
    print(f"   - Ảnh biểu đồ: {len(os.listdir('img'))} file trong thư mục 'img/'")
    print(f"   - Model & Data: {len(os.listdir('result'))} file trong thư mục 'result/'")
    print(f"   - File Excel: result/results_summary.xlsx")

if __name__ == "__main__":
    main()