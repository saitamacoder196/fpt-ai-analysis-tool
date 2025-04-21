# FPT AI Analysis Tool

Công cụ phân tích dữ liệu FPT AI được xây dựng bằng Streamlit, giúp thu thập và phân tích dữ liệu từ FPT.AI, Maya Portal và tích hợp chat với BAchan workplace.

## Tính năng chính

- **Thu thập dữ liệu từ FPT.AI**: Kết nối với API của FPT.AI để lấy dữ liệu và phân tích.
- **Thu thập dữ liệu từ Maya Portal**: Truy cập và phân tích dữ liệu từ Maya Portal.
- **Chat với BAchan workplace**: Tương tác với trợ lý ảo BAchan để thực hiện các tác vụ và truy vấn dữ liệu.
- **Phân tích dữ liệu**: Cung cấp nhiều công cụ trực quan hóa và phân tích dữ liệu.
- **Tạo báo cáo**: Tạo và xuất báo cáo tự động với nhiều định dạng khác nhau.

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- Các thư viện được liệt kê trong file `requirements.txt`

### Cài đặt với conda

```bash
# Tạo môi trường conda mới
conda create -n fpt-ai-analysis python=3.8
conda activate fpt-ai-analysis

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Cài đặt thông thường

```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
venv\Scripts\activate  # Trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## Cách sử dụng

### Chạy ứng dụng

```bash
streamlit run app.py
```

Sau khi chạy lệnh trên, ứng dụng sẽ được khởi động và bạn có thể truy cập qua trình duyệt web tại địa chỉ: http://localhost:8501

### Cấu hình API keys

Để sử dụng đầy đủ tính năng của ứng dụng, bạn cần cung cấp các API keys:

1. **FPT.AI API Key**: Lấy từ [FPT.AI Developer Portal](https://fpt.ai/developer)
2. **Maya Portal**: Sử dụng tài khoản đăng nhập Maya Portal của bạn
3. **BAchan Workplace API Token**: Lấy từ cài đặt tài khoản trong BAchan Workplace

## Cấu trúc dự án

```
fpt-ai-analysis-tool/
│
├── src/
│   ├── pages/           # Chứa các trang của ứng dụng
│   ├── components/      # Chứa các component tái sử dụng
│   ├── utils/           # Chứa các hàm tiện ích
│   ├── services/        # Chứa các service kết nối API
│   └── assets/          # Chứa tài nguyên như hình ảnh, styles
│
├── data/                # Thư mục lưu trữ dữ liệu tạm thời
│
├── app.py               # File chính để chạy ứng dụng
├── requirements.txt     # Danh sách các thư viện cần thiết
├── README.md            # Tài liệu hướng dẫn
└── .gitignore           # File cấu hình Git ignore
```

## Phát triển

### Thêm trang mới

Để thêm một trang mới vào ứng dụng:

1. Tạo file Python mới trong thư mục `src/pages/`, ví dụ: `my_new_page.py`
2. Định nghĩa biến `title` và hàm `render()` trong file
3. Ứng dụng sẽ tự động phát hiện và thêm trang mới vào menu

Ví dụ:

```python
import streamlit as st

title = "My New Page"

def render():
    st.title("This is my new page")
    st.write("Hello world!")
```

### Thêm service mới

Để thêm một service mới để kết nối với API bên ngoài:

1. Tạo file Python mới trong thư mục `src/services/`, ví dụ: `my_service.py`
2. Định nghĩa class service với các phương thức cần thiết
3. Import và sử dụng service trong các trang

## Đóng góp

Mọi đóng góp cho dự án đều được hoan nghênh. Vui lòng tạo pull request hoặc báo cáo lỗi qua mục Issues.

## Giấy phép

© 2025 FPT AI Analysis Tool