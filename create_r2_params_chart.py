"""Script để tạo biểu đồ R² score theo các tham số"""
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from improved.data_loader_improved import load_and_prepare_data
from improved.model_trainer_improved import train_decision_trees_improved
from visualization import create_r2_score_by_params_chart

def main():
    print("="*70)
    print("TẠO BIỂU ĐỒ R² SCORE THEO CÁC THAM SỐ")
    print("="*70)
    
    # Đọc dữ liệu
    dataset_path = 'Folds5x2_pp.xlsx'
    X, y = load_and_prepare_data(dataset_path, use_enhanced_features=False)
    
    # Huấn luyện mô hình
    print("\n📊 Đang huấn luyện mô hình...")
    train_df, test_df, feature_importance_df, best_models, best_model_info = \
        train_decision_trees_improved(X, y, n_runs=10, use_grid_search=True)
    
    # Tạo biểu đồ
    print("\n📊 Đang tạo biểu đồ R² score theo các tham số...")
    create_r2_score_by_params_chart(best_models)
    
    print("\n✅ Hoàn thành! Biểu đồ đã được lưu tại: img/r2_score_by_params.png")
    print("="*70)

if __name__ == "__main__":
    main()

