# THÔNG TIN CHIA DỮ LIỆU CHO SLIDE

## Bảng chia dữ liệu:

| Tập dữ liệu | Số phần tử | Tỷ lệ (%) |
|-------------|------------|-----------|
| **Train** | 38,272 | 80.0% |
| **Test** | 9,568 | 20.0% |
| **Tổng** | 47,840 | 100% |

## Thông tin chi tiết:

- **Tổng số phần tử:** 47,840
- **Chia dữ liệu:** `train_test_split(X, y, test_size=0.2, random_state=42+i, shuffle=True)`
  - `test_size=0.2` → 20% cho test set
  - `random_state=42+i` → i từ 0 đến 9 (10 lần chạy)
  - `shuffle=True` → xáo trộn dữ liệu trước khi chia
- **Tất cả thuộc tính đều là số liên tục:** AT, V, AP, RH, PE
- **Không cần LabelEncoder:** Không có thuộc tính phân lớp (tất cả đều là số liên tục)

## Lưu ý:

⚠️ **Sửa trong slide:**
- Slide hiện tại: `random_state=40+i`
- Code thực tế: `random_state=42+i`
- → **Cần sửa slide thành:** `random_state=42+i`

