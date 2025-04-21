import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import requests
import json
import os
from datetime import datetime, timedelta
import sys

# Import các service và utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.fptai_service import FPTAIService
from utils.data_processor import DataProcessor

title = "FPT.AI Data"

def render():
    st.title("Thu thập và phân tích dữ liệu từ FPT.AI")
    
    # Kiểm tra xem đã có API key trong session state chưa
    if "fptai_api_key" not in st.session_state:
        st.session_state.fptai_api_key = ""
    
    # Form nhập API key
    with st.expander("Cấu hình API", expanded=not bool(st.session_state.fptai_api_key)):
        api_key = st.text_input(
            "FPT.AI API Key",
            value=st.session_state.fptai_api_key,
            type="password",
            help="Nhập API key của bạn từ FPT.AI Developer Portal"
        )
        
        if st.button("Lưu API Key"):
            st.session_state.fptai_api_key = api_key
            st.success("Đã lưu API key thành công!")
    
    # Nếu chưa có API key, hiển thị hướng dẫn
    if not st.session_state.fptai_api_key:
        st.info("Vui lòng nhập API key để tiếp tục")
        st.markdown("""
        ### Hướng dẫn lấy API key FPT.AI
        
        1. Đăng nhập vào [FPT.AI Developer Portal](https://fpt.ai/developer)
        2. Vào phần quản lý tài khoản
        3. Tìm và sao chép API key của bạn
        """)
        return
    
    # Tạo instance của service
    fptai_service = FPTAIService(st.session_state.fptai_api_key)
    
    # Tabs cho các tính năng khác nhau
    tab1, tab2, tab3 = st.tabs(["Dữ liệu API", "Phân tích", "Báo cáo"])
    
    with tab1:
        st.header("Thu thập dữ liệu API")
        
        # Chọn loại API
        api_type = st.selectbox(
            "Chọn loại API",
            ["Text To Speech", "Speech To Text", "Face Recognition", "OCR", "Other Services"]
        )
        
        # Khoảng thời gian
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Từ ngày",
                value=datetime.now().date() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "Đến ngày",
                value=datetime.now().date()
            )
        
        # Nút để lấy dữ liệu
        if st.button("Lấy dữ liệu", key="get_data"):
            with st.spinner("Đang lấy dữ liệu..."):
                try:
                    # Giả lập lấy dữ liệu thực tế từ service
                    # Trong thực tế, bạn sẽ gọi API thực sự từ FPT.AI
                    data = generate_sample_data(api_type, start_date, end_date)
                    
                    # Lưu dữ liệu vào session state
                    st.session_state.fptai_data = data
                    
                    # Hiển thị bảng dữ liệu
                    st.dataframe(data)
                    
                    # Tùy chọn xuất dữ liệu
                    if st.download_button(
                        "Tải dữ liệu (CSV)",
                        data=data.to_csv(index=False).encode('utf-8'),
                        file_name=f"fptai_{api_type.lower().replace(' ', '_')}_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    ):
                        st.success("Đã tải xuống dữ liệu thành công!")
                        
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {str(e)}")
        
        # Hiển thị dữ liệu nếu đã có trong session state
        if "fptai_data" in st.session_state:
            st.subheader("Dữ liệu đã thu thập")
            st.dataframe(st.session_state.fptai_data)
    
    with tab2:
        st.header("Phân tích dữ liệu")
        
        if "fptai_data" not in st.session_state:
            st.info("Vui lòng thu thập dữ liệu trước khi phân tích")
            return
        
        # Phân tích dữ liệu đã có
        data = st.session_state.fptai_data
        
        # Hiển thị tổng quan
        st.subheader("Tổng quan")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số API calls", f"{len(data):,}")
        col2.metric("Tổng chi phí", f"{data['cost'].sum():,.2f} VND")
        col3.metric("Tỷ lệ thành công", f"{(data['status'] == 'success').mean() * 100:.1f}%")
        
        # Biểu đồ theo thời gian
        st.subheader("Số lượng API calls theo thời gian")
        time_chart = alt.Chart(data).mark_line().encode(
            x='date:T',
            y='count:Q',
            tooltip=['date:T', 'count:Q']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(time_chart, use_container_width=True)
        
        # Biểu đồ phân phối theo trạng thái
        st.subheader("Phân phối theo trạng thái")
        status_data = data['status'].value_counts().reset_index()
        status_data.columns = ['status', 'count']
        
        status_chart = alt.Chart(status_data).mark_bar().encode(
            x=alt.X('status:N', sort='-y'),
            y='count:Q',
            color='status:N',
            tooltip=['status:N', 'count:Q']
        ).properties(
            width=500,
            height=300
        )
        st.altair_chart(status_chart, use_container_width=True)
        
        # Phân tích chi tiết hơn dựa trên loại API
        st.subheader("Phân tích chi tiết")
        
        # Filter options
        filter_option = st.selectbox(
            "Lọc theo",
            ["Tất cả", "Thành công", "Thất bại"]
        )
        
        filtered_data = data
        if filter_option == "Thành công":
            filtered_data = data[data['status'] == 'success']
        elif filter_option == "Thất bại":
            filtered_data = data[data['status'] == 'error']
        
        st.dataframe(filtered_data)
    
    with tab3:
        st.header("Báo cáo")
        
        if "fptai_data" not in st.session_state:
            st.info("Vui lòng thu thập dữ liệu trước khi tạo báo cáo")
            return
        
        # Tạo báo cáo từ dữ liệu đã có
        data = st.session_state.fptai_data
        
        st.subheader("Tổng hợp báo cáo")
        
        report_type = st.radio(
            "Loại báo cáo",
            ["Tổng quan", "Chi tiết theo ngày", "Chi tiết theo trạng thái"]
        )
        
        if report_type == "Tổng quan":
            st.write("### Báo cáo tổng quan sử dụng API FPT.AI")
            
            # Summary metrics
            total_calls = len(data)
            total_cost = data['cost'].sum()
            success_rate = (data['status'] == 'success').mean() * 100
            avg_response_time = data['response_time'].mean()
            
            st.markdown(f"""
            #### Thông tin tổng quan
            
            - **Thời gian**: {data['date'].min()} đến {data['date'].max()}
            - **Tổng số API calls**: {total_calls:,}
            - **Tổng chi phí**: {total_cost:,.2f} VND
            - **Tỷ lệ thành công**: {success_rate:.1f}%
            - **Thời gian phản hồi trung bình**: {avg_response_time:.2f} ms
            """)
            
            # Create some visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily trend
                daily_data = data.groupby('date').size().reset_index(name='count')
                daily_chart = alt.Chart(daily_data).mark_line().encode(
                    x='date:T',
                    y='count:Q',
                    tooltip=['date:T', 'count:Q']
                ).properties(
                    title="Số lượng API calls theo ngày",
                    width=350,
                    height=250
                )
                st.altair_chart(daily_chart, use_container_width=True)
            
            with col2:
                # Status distribution
                status_data = data['status'].value_counts().reset_index()
                status_data.columns = ['status', 'count']
                
                status_chart = alt.Chart(status_data).mark_arc().encode(
                    theta='count:Q',
                    color='status:N',
                    tooltip=['status:N', 'count:Q']
                ).properties(
                    title="Phân phối theo trạng thái",
                    width=350,
                    height=250
                )
                st.altair_chart(status_chart, use_container_width=True)
            
            # Recommendations
            st.subheader("Nhận xét và đề xuất")
            st.markdown("""
            - Nên tối ưu hóa việc sử dụng API để giảm chi phí
            - Cần cải thiện tỷ lệ thành công của các cuộc gọi API
            - Theo dõi thời gian phản hồi và xử lý các trường hợp ngoại lệ
            """)
        
        elif report_type == "Chi tiết theo ngày":
            st.write("### Báo cáo chi tiết theo ngày")
            
            # Group by date
            daily_data = data.groupby('date').agg({
                'cost': 'sum',
                'response_time': 'mean',
                'status': lambda x: (x == 'success').mean() * 100
            }).reset_index()
            
            daily_data.columns = ['date', 'total_cost', 'avg_response_time', 'success_rate']
            
            # Display the data
            st.dataframe(daily_data)
            
            # Create chart
            line_chart = alt.Chart(daily_data).mark_line().encode(
                x='date:T',
                y='total_cost:Q',
                tooltip=['date:T', 'total_cost:Q', 'success_rate:Q', 'avg_response_time:Q']
            ).properties(
                title="Chi phí theo ngày",
                width=700,
                height=400
            )
            st.altair_chart(line_chart, use_container_width=True)
            
        elif report_type == "Chi tiết theo trạng thái":
            st.write("### Báo cáo chi tiết theo trạng thái")
            
            # Group by status
            status_data = data.groupby('status').agg({
                'cost': ['sum', 'mean'],
                'response_time': ['mean', 'min', 'max'],
                'date': 'count'
            }).reset_index()
            
            # Flatten multi-level columns
            status_data.columns = ['status', 'total_cost', 'avg_cost', 'avg_response_time', 'min_response_time', 'max_response_time', 'count']
            
            # Display the data
            st.dataframe(status_data)
            
            # Create chart
            bar_chart = alt.Chart(status_data).mark_bar().encode(
                x='status:N',
                y='count:Q',
                color='status:N',
                tooltip=['status:N', 'count:Q', 'total_cost:Q', 'avg_response_time:Q']
            ).properties(
                title="Số lượng API calls theo trạng thái",
                width=700,
                height=400
            )
            st.altair_chart(bar_chart, use_container_width=True)
        
        # Tùy chọn xuất báo cáo
        report_format = st.selectbox(
            "Định dạng báo cáo",
            ["PDF", "Excel", "CSV"]
        )
        
        if st.button("Tạo báo cáo"):
            st.success(f"Đã tạo báo cáo dạng {report_format} thành công!")
            
            # Trong thực tế, bạn sẽ tạo file báo cáo thực sự và cho phép tải xuống
            # Đây chỉ là ví dụ minh họa
            if report_format == "CSV":
                if report_type == "Tổng quan":
                    download_data = data
                elif report_type == "Chi tiết theo ngày":
                    download_data = daily_data
                else:
                    download_data = status_data
                
                st.download_button(
                    "Tải xuống báo cáo",
                    data=download_data.to_csv(index=False).encode('utf-8'),
                    file_name=f"fptai_report_{report_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# Hàm tạo dữ liệu mẫu để demo
def generate_sample_data(api_type, start_date, end_date):
    # Tạo dãy ngày
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    
    # Tạo số lượng API calls ngẫu nhiên cho mỗi ngày
    np.random.seed(42)  # Đặt seed để có kết quả nhất quán
    
    n_samples = delta * 20  # 20 mẫu mỗi ngày
    
    data = {
        'date': np.random.choice(dates, n_samples),
        'api_type': api_type,
        'status': np.random.choice(['success', 'error'], n_samples, p=[0.95, 0.05]),
        'response_time': np.random.normal(200, 50, n_samples),
        'cost': np.random.uniform(1000, 5000, n_samples),
        'request_id': [f"req_{i}" for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Sắp xếp theo ngày
    df = df.sort_values(by='date')
    
    # Thêm cột count cho biểu đồ theo thời gian
    df['count'] = 1
    
    return df