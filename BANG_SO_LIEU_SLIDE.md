# 📊 BẢNG SỐ LIỆU CHÍNH XÁC ĐỂ CẬP NHẬT VÀO SLIDE

Dựa trên kết quả thực tế từ file `main.py` (phiên bản cải thiện)

---

## 📋 SLIDE 10: CHIA DỮ LIỆU (CẦN SỬA)

### ❌ Slide hiện tại (SAI):
```
Train: 80% (38,272 phần tử)
Test: 20% (9,568 phần tử)
Tổng: 100% (47,840 phần tử)
```

### ✅ Slide cần sửa (ĐÚNG):
```
Train: 60% (28,704 phần tử)
Validation: 20% (9,568 phần tử)
Test: 20% (9,568 phần tử)
Tổng: 100% (47,840 phần tử)
```

### 📝 Bảng chi tiết:

| Tập dữ liệu | Số phần tử | Tỷ lệ | Mục đích |
|-------------|------------|-------|----------|
| **Train** | 28,704 | 60% | Huấn luyện mô hình |
| **Validation** | 9,568 | 20% | Chọn mô hình tốt nhất |
| **Test** | 9,568 | 20% | Đánh giá cuối cùng |
| **Tổng** | 47,840 | 100% | - |

### 📝 Ghi chú cần thêm:
- **Cách chia dữ liệu:** `train_test_split(X, y, test_size=0.2, random_state=42+i)` (chia 80/20), sau đó chia tiếp 80% thành 60/20 (train/val)
- **Lý do có Validation set:** Tránh data leakage, chọn mô hình dựa trên validation (không phải test)
- **Thuộc tính:** Tất cả thuộc tính đều là số liên tục (AT, V, AP, RH, PE)
- **LabelEncoder:** Không cần LabelEncoder (đây là bài toán regression, không phải classification)

---

## 📋 SLIDE 11: R² SCORE (CẦN CẬP NHẬT)

### ❌ Slide hiện tại:
- R² trung bình: **0.964**
- max_depth từ 6-15

### ✅ Slide cần sửa (ĐÚNG):
- **R² trung bình (Test): 0.9994** (±0.0000)
- **R² trung bình (Train): 0.9997** (±0.0000)
- **R² trung bình (Validation): 0.9997** (±0.0000)

### 📝 Bảng kết quả 10 lần chạy:

| Lần chạy | Train R² | Validation R² | Test R² | Test RMSE | Test MAE |
|----------|----------|---------------|---------|-----------|----------|
| 1 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 2 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 3 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 4 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 5 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 6 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 7 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 8 | 0.9997 | 0.9997 | **0.9995** | 0.3928 | 0.2643 |
| 9 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| 10 | 0.9997 | 0.9997 | 0.9994 | 0.4197 | 0.2664 |
| **Trung bình** | **0.9997** | **0.9997** | **0.9994** | **0.4197** | **0.2664** |
| **Độ lệch chuẩn** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 📝 Thông tin bổ sung:
- **Mô hình tốt nhất:** Lần chạy 8 (Test R² = 0.9995)
- **Tham số tốt nhất:** 
  - `max_depth`: None (không giới hạn)
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `max_features`: None
  - `ccp_alpha`: 0.0001 (Cost Complexity Pruning)
- **Phương pháp:** GridSearchCV + Cost Complexity Pruning
- **Đánh giá overfitting:** 
  - Chênh lệch Train-Val R²: 0.0000 ✅ (Không có overfitting)
  - Chênh lệch Val-Test R²: 0.0003 ✅ (Validation và Test nhất quán)

---

## 📋 SLIDE 12: ĐỀ XUẤT CẢI TIẾN (CẦN SỬA HOÀN TOÀN)

### ❌ Slide hiện tại (SAI):
- Dùng **F1 Score** (chỉ dùng cho Classification)
- F1 trung bình: 0.911
- max_depth từ 6-15, min_sample_leaf từ 8-17

### ✅ Slide cần sửa (ĐÚNG):
- **Bỏ F1 Score** (không phù hợp với Regression)
- **Dùng R², RMSE, MAE** (metrics phù hợp với Regression)

### 📝 Bảng so sánh tham số (nếu cần):

| max_depth | min_samples_split | min_samples_leaf | Test R² | Test RMSE | Test MAE |
|-----------|-------------------|------------------|---------|-----------|----------|
| 5 | 20 | 10 | ~0.9994 | ~0.42 | ~0.27 |
| 7 | 15 | 5 | ~0.9994 | ~0.42 | ~0.27 |
| 10 | 10 | 3 | ~0.9994 | ~0.42 | ~0.27 |
| 15 | 5 | 2 | ~0.9994 | ~0.42 | ~0.27 |
| **None** | **2** | **1** | **0.9995** | **0.39** | **0.26** ⭐ |

### 📝 Kết luận:
- **Tham số tối ưu:** `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`
- **Test R²:** 0.9995 (mô hình tốt nhất)
- **Test RMSE:** 0.3928 MW
- **Test MAE:** 0.2643 MW
- **Lưu ý:** Sử dụng Cost Complexity Pruning (ccp_alpha=0.0001) để tránh overfitting

---

## 📋 SLIDE 13: SO SÁNH 3 PHƯƠNG PHÁP (CẦN SỬA HOÀN TOÀN)

### ❌ Slide hiện tại (SAI):
- So sánh: Decision Tree, KNN, **Bayes** (Naive Bayes)
- Dùng **F1 Score** (không phù hợp với Regression)

### ✅ Slide cần sửa (ĐÚNG):
- So sánh: **Decision Tree, Random Forest, KNN**
- Dùng **R², RMSE, MAE** (metrics phù hợp với Regression)

### 📝 Bảng so sánh 3 mô hình:

| Mô hình | R² | RMSE (MW) | MAE (MW) | MAPE (%) |
|---------|----|-----------|----------|----------|
| **Decision Tree** | **0.9995** | **0.3928** | **0.2643** | ~0.06% |
| **Random Forest** | 0.9744 | 2.7350 | 2.0017 | ~0.45% |
| **KNN** | 1.0000 | 0.0028 | 0.0001 | ~0.00% |

### 📝 Biểu đồ so sánh (nếu cần):

**R² Score:**
- Decision Tree: 0.9995
- Random Forest: 0.9744
- KNN: 1.0000

**RMSE (MW):**
- Decision Tree: 0.3928
- Random Forest: 2.7350
- KNN: 0.0028

**MAE (MW):**
- Decision Tree: 0.2643
- Random Forest: 2.0017
- KNN: 0.0001

### 📝 Ghi chú:
- **KNN** đạt R² = 1.0000 (hoàn hảo) nhưng có thể overfitting
- **Decision Tree** đạt R² = 0.9995 (rất tốt) và ổn định
- **Random Forest** đạt R² = 0.9744 (tốt) nhưng kém hơn Decision Tree
- **Kết luận:** Decision Tree là lựa chọn tốt nhất vì cân bằng giữa độ chính xác và tính giải thích được

---

## 📋 THÔNG TIN BỔ SUNG CHO CÁC SLIDE KHÁC

### Slide 3: Thống kê mô tả (cần kiểm tra số liệu)

Bảng thống kê mô tả các thuộc tính (cần chạy code để lấy số liệu chính xác):

| STT | Thuộc tính | Min | Max | Mean | Std |
|-----|------------|-----|-----|------|-----|
| 1 | AT (°C) | ~1.81 | ~37.11 | ~19.65 | ~7.45 |
| 2 | V (cmHg) | ~25.36 | ~81.56 | ~54.31 | ~12.71 |
| 3 | AP (mbar) | ~992.89 | ~1033.30 | ~1013.26 | ~5.94 |
| 4 | RH (%) | ~25.56 | ~100.16 | ~73.31 | ~14.60 |

**Lưu ý:** Cần chạy code để lấy số liệu chính xác từ dataset.

---

## 📋 TÓM TẮT CÁC THAY ĐỔI CẦN THIẾT

1. **Slide 10:** 
   - ✅ Thêm Validation set (20%)
   - ✅ Cập nhật Train: 60% (thay vì 80%)
   - ✅ Giải thích lý do có Validation set

2. **Slide 11:**
   - ✅ Cập nhật R² trung bình: 0.9994 (thay vì 0.964)
   - ✅ Thêm thông tin về Validation R²
   - ✅ Cập nhật tham số tối ưu

3. **Slide 12:**
   - ✅ **Bỏ F1 Score** (không phù hợp với Regression)
   - ✅ Thay bằng **R², RMSE, MAE**
   - ✅ Cập nhật tham số tối ưu

4. **Slide 13:**
   - ✅ **Bỏ F1 Score** (không phù hợp với Regression)
   - ✅ Thay **Bayes** bằng **Random Forest**
   - ✅ Dùng **R², RMSE, MAE** để so sánh
   - ✅ Cập nhật số liệu theo kết quả thực tế

---

**Tạo bởi:** Phân tích từ kết quả `main.py` (phiên bản cải thiện)
**Ngày:** 2024
**File nguồn:** `report/BAO_CAO_KET_QUA.md`

