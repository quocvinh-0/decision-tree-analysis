"""Script test để kiểm tra biểu đồ so sánh 3 phương pháp"""
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from improved.data_loader_improved import load_and_prepare_data
from improved.model_trainer_improved import train_decision_trees_improved
from visualization import create_comparison_3_methods_chart

def main():
    print("="*70)
    print("TEST BIỂU ĐỒ SO SÁNH 3 PHƯƠNG PHÁP")
    print("="*70)
    
    # Đọc dữ liệu
    dataset_path = 'Folds5x2_pp.xlsx'
    X, y = load_and_prepare_data(dataset_path, use_enhanced_features=False)
    
    # Huấn luyện mô hình (chỉ cần 10 lần chạy)
    print("\n📊 Đang huấn luyện Decision Tree (10 lần)...")
    train_df, test_df, feature_importance_df, best_models, best_model_info = \
        train_decision_trees_improved(X, y, n_runs=10, use_grid_search=True)
    
    # Tạo biểu đồ
    print("\n📊 Đang tạo biểu đồ so sánh 3 phương pháp...")
    create_comparison_3_methods_chart(best_models, test_df)
    
    print("\n✅ Hoàn thành!")
    print("="*70)

if __name__ == "__main__":
    main()

