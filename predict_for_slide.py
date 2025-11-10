"""Script để dự đoán và truy vết đường đi trong cây quyết định từ mô hình tốt nhất"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def trace_decision_path(model, sample_input, feature_names, max_depth=None):
    """
    Truy vết đường đi của một mẫu dữ liệu trong cây quyết định.
    Trả về giá trị dự đoán và danh sách các bước trong đường đi.
    
    Parameters:
    - max_depth: Nếu None, truy vết toàn bộ. Nếu là số, chỉ truy vết đến độ sâu đó.
    """
    tree = model.tree_
    node_id = 0
    path = []
    depth = 0

    while tree.children_left[node_id] != tree.children_right[node_id]:  # Khi chưa phải là nút lá
        if max_depth is not None and depth >= max_depth:
            break
            
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]
        sample_value = sample_input[feature_name]

        if sample_value <= threshold:
            path.append({
                'condition': f"{feature_name} <= {threshold:.3f}",
                'result': True,
                'value': sample_value,
                'node_id': node_id,
                'depth': depth
            })
            node_id = tree.children_left[node_id]
        else:
            path.append({
                'condition': f"{feature_name} <= {threshold:.3f}",
                'result': False,
                'value': sample_value,
                'node_id': node_id,
                'depth': depth
            })
            node_id = tree.children_right[node_id]
        
        depth += 1
    
    predicted_value = tree.value[node_id][0][0]
    samples_at_leaf = tree.n_node_samples[node_id]
    mse_at_leaf = tree.impurity[node_id] * samples_at_leaf
    
    return predicted_value, path, {
        'node_id': node_id,
        'samples': samples_at_leaf,
        'mse': mse_at_leaf,
        'depth': depth
    }

def main():
    print("="*70)
    print("DỰ ĐOÁN CHO MỘT PHẦN TỬ MỚI SỬ DỤNG MÔ HÌNH TỐT NHẤT")
    print("="*70)

    # 1. Tải mô hình tốt nhất
    model_path = os.path.join('result', 'best_decision_tree_model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: Không tìm thấy mô hình tại {model_path}.")
        print("Vui lòng chạy main.py trước để huấn luyện và lưu mô hình.")
        return

    print(f"\n📂 Đang tải mô hình từ: {model_path}")
    best_model = joblib.load(model_path)
    print("   ✅ Đã tải mô hình thành công.")

    # 2. Định nghĩa phần tử mới (có thể thay đổi giá trị này)
    new_sample_data = {
        'AT': 18.00,  # Nhiệt độ môi trường
        'V': 50.00,   # Áp suất hơi
        'AP': 1015.00,  # Áp suất khí quyển
        'RH': 70.00   # Độ ẩm tương đối
    }
    
    # Hoặc lấy từ dòng lệnh nếu có
    if len(sys.argv) > 1:
        try:
            new_sample_data['AT'] = float(sys.argv[1]) if len(sys.argv) > 1 else new_sample_data['AT']
            new_sample_data['V'] = float(sys.argv[2]) if len(sys.argv) > 2 else new_sample_data['V']
            new_sample_data['AP'] = float(sys.argv[3]) if len(sys.argv) > 3 else new_sample_data['AP']
            new_sample_data['RH'] = float(sys.argv[4]) if len(sys.argv) > 4 else new_sample_data['RH']
        except:
            print("⚠️  Lỗi khi đọc tham số từ dòng lệnh, sử dụng giá trị mặc định")

    new_sample_df = pd.DataFrame([new_sample_data])
    feature_names = ['AT', 'V', 'AP', 'RH']

    print("\n📋 Phần tử mới cần dự đoán:")
    print(f"   • AT (Nhiệt độ môi trường): {new_sample_data['AT']:.2f} °C")
    print(f"   • V (Áp suất hơi): {new_sample_data['V']:.2f} cmHg")
    print(f"   • AP (Áp suất khí quyển): {new_sample_data['AP']:.2f} mbar")
    print(f"   • RH (Độ ẩm tương đối): {new_sample_data['RH']:.2f} %")

    # 3. Thực hiện dự đoán
    predicted_pe = best_model.predict(new_sample_df[feature_names])[0]
    print(f"\n✨ Giá trị PE dự đoán từ mô hình tốt nhất: {predicted_pe:.2f} MW")

    # 4. Truy vết đường đi trong cây quyết định (toàn bộ)
    print("\n" + "="*70)
    print("➡️ ĐƯỜNG ĐI TRONG CÂY QUYẾT ĐỊNH (TOÀN BỘ):")
    print("="*70)
    
    predicted_value_from_path, decision_path, leaf_info = trace_decision_path(
        best_model, new_sample_data, feature_names
    )
    
    # 5. Truy vết đường đi rút gọn (chỉ 3-5 bước đầu)
    print("\n" + "="*70)
    print("➡️ ĐƯỜNG ĐI RÚT GỌN CHO SLIDE (3-5 BƯỚC ĐẦU):")
    print("="*70)
    
    # Lấy 5 bước đầu hoặc đến độ sâu 3
    simplified_path = decision_path[:5] if len(decision_path) > 5 else decision_path
    
    for i, step in enumerate(decision_path, 1):
        result_text = "True" if step['result'] else "False"
        print(f"\nBước {i}: {step['condition']}")
        print(f"   → Giá trị mẫu: {step['value']:.2f}")
        print(f"   → Kết quả: {result_text}")
    
    print(f"\n🏁 Kết thúc tại nút lá:")
    print(f"   • Giá trị dự đoán: {predicted_value_from_path:.2f} MW")
    print(f"   • Số mẫu tại nút lá: {leaf_info['samples']}")
    print(f"   • MSE tại nút lá: {leaf_info['mse']:.3f}")
    
    # Kiểm tra tính nhất quán
    if abs(predicted_pe - predicted_value_from_path) < 1e-6:
        print("\n✅ Giá trị dự đoán từ hàm predict() và từ đường đi khớp nhau.")
    else:
        print(f"\n⚠️  Giá trị dự đoán từ hàm predict() ({predicted_pe:.2f}) và từ đường đi ({predicted_value_from_path:.2f}) KHÔNG khớp nhau.")
        print("   (Có thể do làm tròn hoặc cách tính toán)")

    # Hiển thị đường đi rút gọn
    for i, step in enumerate(simplified_path, 1):
        result_text = "True" if step['result'] else "False"
        print(f"\nBước {i}: {step['condition']}")
        print(f"   → Giá trị mẫu: {step['value']:.2f}")
        print(f"   → Kết quả: {result_text}")
    
    if len(decision_path) > len(simplified_path):
        print(f"\n... (còn {len(decision_path) - len(simplified_path)} bước nữa)")
        print(f"🏁 Kết thúc tại nút lá với giá trị dự đoán: {predicted_value_from_path:.2f} MW")
    else:
        print(f"\n🏁 Kết thúc tại nút lá:")
        print(f"   • Giá trị dự đoán: {predicted_value_from_path:.2f} MW")
        print(f"   • Số mẫu tại nút lá: {leaf_info['samples']}")
        print(f"   • MSE tại nút lá: {leaf_info['mse']:.3f}")

    print("\n" + "="*70)
    print("💡 THÔNG TIN CHO SLIDE (RÚT GỌN):")
    print("="*70)
    print(f"• Giá trị PE dự đoán: {predicted_pe:.2f} MW")
    print(f"• Đường đi trong cây quyết định (rút gọn):")
    for i, step in enumerate(simplified_path, 1):
        result_text = "True" if step['result'] else "False"
        print(f"  {i}. {step['condition']} → {result_text} (Giá trị: {step['value']:.2f})")
    if len(decision_path) > len(simplified_path):
        print(f"  ... (còn {len(decision_path) - len(simplified_path)} bước nữa)")
    print("="*70)
    
    # Ghi vào file để dễ copy
    output_file = 'prediction_result.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("KẾT QUẢ DỰ ĐOÁN CHO SLIDE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Phần tử mới:\n")
        f.write(f"  AT: {new_sample_data['AT']:.2f} °C\n")
        f.write(f"  V: {new_sample_data['V']:.2f} cmHg\n")
        f.write(f"  AP: {new_sample_data['AP']:.2f} mbar\n")
        f.write(f"  RH: {new_sample_data['RH']:.2f} %\n\n")
        f.write(f"Giá trị PE dự đoán: {predicted_pe:.2f} MW\n\n")
        f.write("Đường đi trong cây quyết định (toàn bộ):\n")
        for i, step in enumerate(decision_path, 1):
            result_text = "True" if step['result'] else "False"
            f.write(f"  {i}. {step['condition']} → {result_text} (Giá trị: {step['value']:.2f})\n")
        f.write("\nĐường đi rút gọn cho slide (3-5 bước đầu):\n")
        simplified_path = decision_path[:5] if len(decision_path) > 5 else decision_path
        for i, step in enumerate(simplified_path, 1):
            result_text = "True" if step['result'] else "False"
            f.write(f"  {i}. {step['condition']} → {result_text} (Giá trị: {step['value']:.2f})\n")
        if len(decision_path) > len(simplified_path):
            f.write(f"  ... (còn {len(decision_path) - len(simplified_path)} bước nữa)\n")
    
    print(f"\n✅ Đã lưu kết quả vào file: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()

