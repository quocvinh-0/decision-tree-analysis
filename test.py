import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# ============================
# BƯỚC 1: Load mô hình và scaler với kiểm tra kỹ hơn
# ============================
def load_model_and_scaler():
    """Load mô hình và scaler với kiểm tra lỗi chi tiết"""
    try:
        if not os.path.exists("./result/best_decision_tree_model.pkl"):
            print("❌ LỖI: Không tìm thấy file 'best_decision_tree_model.pkl'")
            print("   Vui lòng chạy script huấn luyện trước!")
            return None, None

        if not os.path.exists("./result/scaler.pkl"):
            print("❌ LỖI: Không tìm thấy file 'scaler.pkl'")
            return None, None

        model = joblib.load("./result/best_decision_tree_model.pkl")
        scaler = joblib.load("./result/scaler.pkl")
        
        print("✅ Đã tải mô hình và scaler thành công!")
        print(f"   Model type: {type(model).__name__}")
        return model, scaler
        
    except Exception as e:
        print(f"❌ LỖI khi tải model: {e}")
        return None, None

# ============================
# BƯỚC 2: Nhập dữ liệu với validation
# ============================
def validate_input(prompt, input_type=float, min_val=None, max_val=None):
    """Validate input với range kiểm tra"""
    while True:
        try:
            value = input_type(input(prompt))
            
            if min_val is not None and value < min_val:
                print(f"   ⚠️ Giá trị phải >= {min_val}")
                continue
                
            if max_val is not None and value > max_val:
                print(f"   ⚠️ Giá trị phải <= {max_val}") 
                continue
                
            return value
            
        except ValueError:
            print("   ⚠️ Vui lòng nhập số hợp lệ!")

def get_user_input():
    """Nhận input từ user với validation"""
    print("\n" + "="*50)
    print("🎯 NHẬP DỮ LIỆU DỰ ĐOÁN NHÀ MÁY ĐIỆN")
    print("="*50)
    
    # Hiển thị hướng dẫn phạm vi giá trị (dựa trên dataset thực tế)
    print("\n📋 HƯỚNG DẪN PHẠM VI GIÁ TRỊ THỰC TẾ:")
    print("   • Nhiệt độ (AT): 1-37°C")
    print("   • Tốc độ gió (V): 25-81 m/s") 
    print("   • Áp suất (AP): 992-1033 hPa")
    print("   • Độ ẩm (RH): 25-100%")
    print("-" * 50)
    
    records = []
    record_count = 1
    
    while True:
        print(f"\n--- Bản ghi #{record_count} ---")
        
        AT = validate_input("   🌡️ Nhiệt độ môi trường (AT, °C): ", float, -50, 50)
        V = validate_input("   💨 Tốc độ gió (V, m/s): ", float, 0, 100)
        AP = validate_input("   📊 Áp suất khí quyển (AP, hPa): ", float, 900, 1100)
        RH = validate_input("   💧 Độ ẩm (RH, %): ", float, 0, 100)
        
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
        more = input("   ➕ Nhập thêm bản ghi? (y/n): ").strip().lower()
        if more not in ['y', 'yes', 'có', 'co']:
            break
            
    return records

# ============================
# BƯỚC 3: Phân tích và dự đoán
# ============================
def analyze_predictions(original_df, predictions):
    """Phân tích kết quả dự đoán"""
    df = original_df.copy()
    df["PE_Predicted"] = predictions
    
    # Phân loại hiệu suất
    conditions = [
        df["PE_Predicted"] >= 500,
        df["PE_Predicted"] >= 450,
        df["PE_Predicted"] >= 400,
        df["PE_Predicted"] < 400
    ]
    choices = ["🔴 RẤT CAO", "🟡 CAO", "🟢 TRUNG BÌNH", "🔵 THẤP"]
    df["Mức_hiệu_suất"] = np.select(conditions, choices, default="🟢 TRUNG BÌNH")
    
    # Đánh giá tổng quan
    avg_pe = df["PE_Predicted"].mean()
    if avg_pe >= 480:
        overall = "🔴 HIỆU SUẤT CAO - VẬN HÀNH TỐI ƯU"
    elif avg_pe >= 430:
        overall = "🟡 HIỆU SUẤT TRUNG BÌNH - ỔN ĐỊNH"
    else:
        overall = "🟢 HIỆU SUẤT THẤP - CẦN KIỂM TRA"
    
    return df, overall, avg_pe

# ============================
# BƯỚC 4: Hiển thị kết quả đẹp mắt
# ============================
def display_results(results_df, overall_rating, avg_pe):
    """Hiển thị kết quả định dạng đẹp"""
    print("\n" + "="*60)
    print("📊 KẾT QUẢ DỰ ĐOÁN HIỆU SUẤT NHÀ MÁY ĐIỆN")
    print("="*60)
    
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
    print("\n" + "📈 TỔNG QUAN HIỆU SUẤT:")
    print(f"   • Đánh giá tổng: {overall_rating}")
    print(f"   • PE trung bình: {avg_pe:.2f} MW")
    print(f"   • Số lượng dự đoán: {len(results_df)}")
    
    # Thống kê phân phối
    performance_counts = results_df["Mức_hiệu_suất"].value_counts()
    print(f"   • Phân bố hiệu suất:")
    for level, count in performance_counts.items():
        print(f"     {level}: {count} mẫu")

# ============================
# BƯỚC 5: Lưu kết quả (tùy chọn)
# ============================
def save_results(results_df, filename=None):
    """Lưu kết quả ra file CSV"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pe_predictions_{timestamp}.csv"
    
    try:
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n💾 Đã lưu kết quả vào file: {filename}")
        return True
    except Exception as e:
        print(f"\n⚠️ Không thể lưu file: {e}")
        return False

# ============================
# HÀM CHÍNH
# ============================
def main():
    print("🔮 DỰ ĐOÁN HIỆU SUẤT NHÀ MÁY ĐIỆN")
    print("   Sử dụng mô hình Decision Tree")
    
    # Load model
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    # Nhập dữ liệu
    records = get_user_input()
    if not records:
        print("❌ Không có dữ liệu để dự đoán!")
        return
    
    # Chuyển sang DataFrame
    new_df = pd.DataFrame(records)
    
    # Dự đoán
    try:
        X_new = new_df[['AT', 'V', 'AP', 'RH']]
        X_new_scaled = scaler.transform(X_new)
        predictions = model.predict(X_new_scaled)
        
        # Phân tích kết quả
        results_df, overall_rating, avg_pe = analyze_predictions(new_df, predictions)
        
        # Hiển thị kết quả
        display_results(results_df, overall_rating, avg_pe)
        
        # Hỏi lưu kết quả
        save_option = input("\n💾 Lưu kết quả ra file CSV? (y/n): ").strip().lower()
        if save_option in ['y', 'yes', 'có', 'co']:
            save_results(results_df)
            
        print("\n✅ Hoàn thành dự đoán!")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình dự đoán: {e}")

if __name__ == "__main__":
    main()