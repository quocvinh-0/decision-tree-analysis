import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# BƯỚC 1: Load mô hình Decision Tree

def load_decision_tree_model():
    # Load mô hình Decision Tree đã lưu
    try:
        if not os.path.exists("./result/best_decision_tree_model.pkl"):
            print("[LỖI] Không tìm thấy file 'best_decision_tree_model.pkl'")
            print("   Vui lòng chạy script huấn luyện trước!")
            return None

        model = joblib.load("./result/best_decision_tree_model.pkl")
        
        print("[OK] Đã tải mô hình Decision Tree thành công!")
        print(f"   Model type: {type(model).__name__}")
        return model
        
    except Exception as e:
        print(f"[LỖI] Không thể tải model: {e}")
        return None


# BƯỚC 2: Nhập dữ liệu với validation

def validate_input(prompt, input_type=float, min_val=None, max_val=None):
    """Validate input với range kiểm tra"""
    while True:
        try:
            value = input_type(input(prompt))
            
            if min_val is not None and value < min_val:
                print(f"   [CẢNH BÁO] Giá trị phải >= {min_val}")
                continue
                
            if max_val is not None and value > max_val:
                print(f"   [CẢNH BÁO] Giá trị phải <= {max_val}") 
                continue
                
            return value
            
        except ValueError:
            print("   [CẢNH BÁO] Vui lòng nhập số hợp lệ!")

def get_user_input():
    # Nhận input từ user với validation
    print("NHẬP DỮ LIỆU DỰ ĐOÁN NHÀ MÁY ĐIỆN")
    # Hiển thị hướng dẫn phạm vi giá trị (dựa trên dataset thực tế)
    print("\nHƯỚNG DẪN PHẠM VI GIÁ TRỊ THỰC TẾ:")
    print("   • Nhiệt độ (AT): 1-37°C")
    print("   • Tốc độ gió (V): 25-81 m/s") 
    print("   • Áp suất (AP): 992-1033 hPa")
    print("   • Độ ẩm (RH): 25-100%")
    print("-" * 50)
    
    records = []
    record_count = 1
    
    while True:
        print(f"\nNhập dữ liệu cho bản ghi thứ {record_count}:")
        AT = validate_input("   Nhập nhiệt độ môi trường (AT, °C): ", float, -50, 50)
        V = validate_input("   Nhập tốc độ gió (V, m/s): ", float, 0, 100)
        AP = validate_input("   Nhập áp suất khí quyển (AP, hPa): ", float, 900, 1100)
        RH = validate_input("   Nhập độ ẩm (RH, %): ", float, 0, 100)
        
        records.append({
            "STT": record_count,
            "AT": AT, 
            "V": V, 
            "AP": AP, 
            "RH": RH,
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        record_count += 1
        
        print("\n" + "-"*30)
        more = input("   Nhập thêm bản ghi? (y/n): ").strip().lower()
        if more not in ['y', 'yes', 'co', 'có']:
            break
            
    return records


# BƯỚC 3: Phân tích và dự đoán

def analyze_predictions(original_df, predictions):
    # Phân tích kết quả dự đoán
    df = original_df.copy()
    df["PE_Predicted"] = predictions
    
    # Phân loại hiệu suất
    conditions = [
        df["PE_Predicted"] >= 500,
        df["PE_Predicted"] >= 450,
        df["PE_Predicted"] >= 400,
        df["PE_Predicted"] < 400
    ]
    choices = ["RẤT CAO", "CAO", "TRUNG BÌNH", "THẤP"]
    df["Mức_hiệu_suất"] = np.select(conditions, choices, default="TRUNG BÌNH")
    
    # Đánh giá tổng quan
    avg_pe = df["PE_Predicted"].mean()
    if avg_pe >= 480:
        overall = "Hiệu suất cao - vận hành tối ưu"
    elif avg_pe >= 430:
        overall = "Hiệu suất trung bình - vận hành ổn định"
    else:
        overall = "Hiệu suất thấp - cần kiểm tra"
    
    return df, overall, avg_pe


# BƯỚC 4: Hiển thị kết quả 

def display_results(results_df, overall_rating, avg_pe):
    print("KẾT QUẢ DỰ ĐOÁN HIỆU SUẤT NHÀ MÁY ĐIỆN")
    
    # Hiển thị bảng kết quả
    display_df = results_df[["STT", "AT", "V", "AP", "RH", "PE_Predicted", "Mức_hiệu_suất"]].copy()
    display_df.columns = ["STT", "Nhiệt độ", "Gió", "Áp suất", "Độ ẩm", "PE Dự đoán", "Đánh giá"]
    
    # Format số
    display_df["PE Dự đoán"] = display_df["PE Dự đoán"].round(2)
    display_df["Nhiệt độ"] = display_df["Nhiệt độ"].round(1)
    display_df["Gió"] = display_df["Gió"].round(1)
    display_df["Áp suất"] = display_df["Áp suất"].round(1)
    display_df["Độ ẩm"] = display_df["Độ ẩm"].round(1)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(display_df.to_string(index=False))
    
    # Hiển thị tổng quan
    print("\nTỔNG QUAN HIỆU SUẤT:")
    print(f"   • Đánh giá tổng: {overall_rating}")
    print(f"   • PE trung bình: {avg_pe:.2f} MW")
    print(f"   • Số lượng dự đoán: {len(results_df)}")
    
    # Thống kê phân phối
    performance_counts = results_df["Mức_hiệu_suất"].value_counts()
    print(f"   • Phân bố hiệu suất:")
    for level, count in performance_counts.items():
        print(f"     {level}: {count} mẫu")


# BƯỚC 5: Vẽ cây quyết định

def visualize_decision_tree(model, X_new):
    """Vẽ cây quyết định với dữ liệu mới"""
    print("\n" + "="*60)
    print("VẼ CÂY QUYẾT ĐỊNH")
    print("="*60)
    
    try:
        # Tạo thư mục img nếu chưa có
        os.makedirs('img', exist_ok=True)
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tree_path = os.path.join('img', f'decision_tree_test_{timestamp}.png')
        
        # Vẽ cây quyết định
        plt.figure(figsize=(25, 12))
        plot_tree(
            model,
            feature_names=['AT', 'V', 'AP', 'RH'],
            filled=True,
            rounded=True,
            impurity=True,
            fontsize=10,
            max_depth=4  # Hiển thị 4 tầng đầu tiên để dễ nhìn
        )
        plt.title(f"CÂY QUYẾT ĐỊNH - DỰ ĐOÁN DỮ LIỆU MỚI", 
                  fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(tree_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Đã lưu cây quyết định: {tree_path}")
        print(f"   • Số lượng node: {model.tree_.node_count}")
        print(f"   • Chiều sâu cây: {model.tree_.max_depth}")
        print(f"   • Số lá (leaf nodes): {model.tree_.n_leaves}")
        print(f"   • Số mẫu dự đoán: {len(X_new)}")
        
        return tree_path
        
    except Exception as e:
        print(f"[LỖI] Không thể vẽ cây quyết định: {e}")
        return None


# BƯỚC 6: Lưu kết quả (tùy chọn)

def save_results(results_df, filename=None):
    """Lưu kết quả ra file CSV"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pe_predictions_{timestamp}.csv"
    
    try:
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nĐã lưu kết quả vào file: {filename}")
        return True
    except Exception as e:
        print(f"\nKhông thể lưu file: {e}")
        return False


# HÀM CHÍNH

def main():
    print("DỰ ĐOÁN HIỆU SUẤT NHÀ MÁY ĐIỆN")
    print("   Sử dụng duy nhất mô hình Decision Tree")
    
    # Load model
    model = load_decision_tree_model()
    if model is None:
        return
    
    # Nhập dữ liệu
    records = get_user_input()
    if not records:
        print("Không có dữ liệu để dự đoán!")
        return
    
    # Chuyển sang DataFrame
    new_df = pd.DataFrame(records)
    
    # Dự đoán
    try:
        X_new = new_df[['AT', 'V', 'AP', 'RH']]
        predictions = model.predict(X_new)
        
        # Phân tích kết quả
        results_df, overall_rating, avg_pe = analyze_predictions(new_df, predictions)
        
        # Hiển thị kết quả
        display_results(results_df, overall_rating, avg_pe)
        
        # Vẽ cây quyết định
        print("\n" + "="*60)
        visualize_option = input("Vẽ cây quyết định? (y/n): ").strip().lower()
        if visualize_option in ['y', 'yes', 'co', 'có']:
            tree_path = visualize_decision_tree(model, X_new)
            if tree_path:
                print(f"\nBạn có thể xem cây quyết định tại: {tree_path}")
        
        # Hỏi lưu kết quả
        save_option = input("\nLưu kết quả ra file CSV? (y/n): ").strip().lower()
        if save_option in ['y', 'yes', 'co', 'có']:
            save_results(results_df)
            
        print("\nĐã hoàn thành dự đoán!")
        
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")

if __name__ == "__main__":
    main()