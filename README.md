# 🌳 Decision Tree Analysis - Phân tích dữ liệu với Cây quyết định

Dự án phân tích dữ liệu sử dụng phương pháp Decision Tree để dự đoán sản lượng điện (Power Output) từ các đặc trưng môi trường.

## 📋 Mô tả

Dự án này sử dụng Decision Tree Regressor để phân tích dataset `Folds5x2_pp.xlsx` (Combined Cycle Power Plant Data) với các đặc trưng:
- **AT**: Ambient Temperature (Nhiệt độ môi trường)
- **V**: Exhaust Vacuum (Áp suất hơi)
- **AP**: Ambient Pressure (Áp suất khí quyển)
- **RH**: Relative Humidity (Độ ẩm tương đối)

**Target**: **PE** - Net hourly electrical energy output (Sản lượng điện)

## 🚀 Cài đặt

### 1. Cài đặt packages

```bash
pip install -r requirements.txt
```

### 2. Chạy dự án

```bash
python main.py
```

## 📁 Cấu trúc dự án

```
decision-tree-analysis/
├── main.py                    # File chính
├── data_loader.py            # Load và xử lý dữ liệu
├── model_trainer.py          # Huấn luyện mô hình
├── model_comparison.py       # So sánh với mô hình khác
├── visualization.py          # Tạo biểu đồ
├── results_saver.py          # Lưu kết quả
│
├── improved/                 # Code cải thiện (tùy chọn)
│   ├── README.md
│   ├── data_loader_improved.py
│   └── model_trainer_improved.py
│
├── docs/                     # Tài liệu đánh giá
│   ├── README.md
│   ├── README_DANH_GIA.md    # ⭐ Đánh giá dự án
│   └── ...
│
├── img/                      # Biểu đồ kết quả
├── result/                   # Model và kết quả
└── Folds5x2_pp.xlsx         # Dataset
```

**Xem `PROJECT_STRUCTURE.md` để biết chi tiết cấu trúc.**

## 📊 Kết quả

Sau khi chạy, bạn sẽ có:
- **Biểu đồ**: Trong thư mục `img/`
- **Model**: Trong thư mục `result/`
- **Báo cáo Excel**: `result/results_summary.xlsx`

## 🔧 Cải thiện

Dự án đã được đánh giá và có các phiên bản cải thiện:

### ⚠️ Vấn đề đã phát hiện:
- Data leakage (chọn mô hình dựa trên test set)
- Scaling không cần thiết cho Decision Tree
- Hyperparameter tuning thủ công
- Thiếu cost complexity pruning

### ✅ Giải pháp:
- Xem `docs/README_DANH_GIA.md` để biết chi tiết
- Sử dụng code trong thư mục `improved/` để cải thiện kết quả
- Xem `improved/README.md` để biết cách sử dụng

## 📚 Tài liệu

- **Đánh giá dự án**: Xem `docs/README_DANH_GIA.md`
- **Hướng dẫn cải thiện**: Xem `docs/HUONG_DAN_CAI_THIEN.md`
- **So sánh phương pháp**: Xem `docs/COMPARISON_OLD_VS_IMPROVED.md`
- **Cấu trúc dự án**: Xem `PROJECT_STRUCTURE.md`

## 📦 Requirements

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- joblib >= 1.1.0
- openpyxl >= 3.0.0

## 👥 Tác giả

Nhóm học tập - Môn: Máy học ứng dụng

## 📝 License

Dự án học tập

---

**Lưu ý**: Để có kết quả tốt nhất, nên sử dụng code cải thiện trong thư mục `improved/`. Xem `docs/README_DANH_GIA.md` để biết chi tiết.
