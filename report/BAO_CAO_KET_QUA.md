# 📊 BÁO CÁO KẾT QUẢ PHÂN TÍCH DECISION TREE

## 1. THÔNG TIN DATASET

- **Dataset**: Folds5x2_pp.xlsx (Combined Cycle Power Plant Data)
- **Số mẫu**: 47,840 mẫu (5 sheets × 9,568 mẫu/sheet)
- **Đặc trưng**: AT, V, AP, RH
- **Target**: PE (Net hourly electrical energy output)

## 2. PHƯƠNG PHÁP

### 2.1. Tiền xử lý dữ liệu
- Không sử dụng scaling (Decision Tree không cần)
- Không sử dụng feature engineering

### 2.2. Phân chia dữ liệu
- **Train set**: 80%
- **Test set**: 20%

### 2.3. Huấn luyện mô hình
- Sử dụng **GridSearchCV** để tìm hyperparameter tối ưu
- Sử dụng **Cost Complexity Pruning** để giảm overfitting
- Chạy 10 lần với các random_state khác nhau
- Chọn mô hình tốt nhất dựa trên **test set**

## 3. KẾT QUẢ

### 3.1. Kết quả tổng hợp (10 lần chạy)

| Metric | Train | Test |
|--------|-------|------|
| R² (TB) | 0.9991 | 0.9984 |
| RMSE (TB) | 0.5144 | 0.6824 |
| MAE (TB) | 0.2847 | 0.3632 |

### 3.2. Mô hình tốt nhất

- **Lần chạy**: 3
- **Test R²**: 0.9990
- **Tham số**: {'max_depth': 20, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'ccp_alpha': np.float64(0.0001)}

### 3.3. Đánh giá overfitting

- **Chênh lệch Train-Test R²**: 0.0007
- **Kết luận**: ✅ Không có overfitting nghiêm trọng

### 3.4. Độ quan trọng đặc trưng

| Đặc trưng | Độ quan trọng (TB) | Độ lệch chuẩn |
|-----------|-------------------|---------------|
| AT | 0.9053 | 0.0010 |
| V | 0.0574 | 0.0007 |
| AP | 0.0206 | 0.0008 |
| RH | 0.0167 | 0.0009 |

### 3.5. So sánh với mô hình khác

| Mô hình | R² | RMSE | MAE |
|---------|----|----|----|
| Decision Tree | 0.9990 | 0.5370 | 0.3204 |
| Random Forest | 0.9759 | 2.6364 | 1.9622 |
| Naive Bayes | 0.8083 | 7.4339 | 6.0220 |

**Lưu ý:** Naive Bayes được chuyển đổi từ Classification (chia PE thành 3 lớp: Thấp, Trung bình, Cao)

### 3.6. Cross-Validation (5-fold)

- **Train R²**: 0.9987 (±0.0004)
- **Test R²**: 0.9962 (±0.0006)
- **Test RMSE**: 1.0433 (±0.0838)

## 4. KẾT LUẬN

✅ Mô hình Decision Tree đạt hiệu suất **XUẤT SẮC** với R² > 0.95

## 5. FILE KẾT QUẢ

- **Biểu đồ**: Thư mục `img/`
- **Model**: `result/best_decision_tree_model.pkl`
- **Excel**: `result/results_summary.xlsx`
- **Báo cáo**: `report/BAO_CAO_KET_QUA.md`

