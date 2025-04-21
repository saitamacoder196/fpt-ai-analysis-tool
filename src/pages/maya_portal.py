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
from services.maya_service import MayaPortalService
from utils.data_processor import DataProcessor

title = "Maya Portal"

def render():
    st.title("Thu thập và phân tích dữ liệu từ Maya Portal")
    
    # Kiểm tra xem đã có thông tin đăng nhập trong session state chưa
    if "maya_username" not in st.session_state:
        st.session_state.maya_username = ""
    if "maya_password" not in st.session_state:
        st.session_state.maya_password = ""
    
    # Form đăng nhập
    with st.expander("Đăng nhập Maya Portal", expanded=not (bool(st.session_state.maya_username) and bool(st.session_state.maya_password))):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input(
                "Username",
                value=st.session_state.maya_username,
                help="Nhập tên đăng nhập Maya Portal của bạn"
            )
        with col2:
            password = st.text_input(
                "Password",
                value=st.session_state.maya_password,
                type="password",
                help="Nhập mật khẩu Maya Portal của bạn"
            )
        
        if st.button("Đăng nhập"):
            if username and password:
                # Trong thực tế, bạn sẽ thực hiện xác thực thực sự ở đây
                # Đây chỉ là mô phỏng
                with st.spinner("Đang đăng nhập..."):
                    # Giả lập đăng nhập thành công
                    st.session_state.maya_username = username
                    st.session_state.maya_password = password
                    st.session_state.maya_logged_in = True
                    st.success("Đăng nhập thành công!")
            else:
                st.error("Vui lòng nhập đầy đủ thông tin đăng nhập")
    
    # Kiểm tra đăng nhập
    if not st.session_state.get("maya_logged_in", False):
        if st.session_state.maya_username and st.session_state.maya_password:
            # Nếu đã có thông tin đăng nhập nhưng chưa đăng nhập
            st.warning("Vui lòng nhấn Đăng nhập để tiếp tục")
        else:
            # Nếu chưa có thông tin đăng nhập
            st.info("Vui lòng đăng nhập vào Maya Portal để tiếp tục")
        return
    
    # Tạo instance của service
    maya_service = MayaPortalService(st.session_state.maya_username, st.session_state.maya_password)
    
    # Tabs cho các tính năng khác nhau
    tab1, tab2, tab3 = st.tabs(["Dữ liệu Maya", "Phân tích", "Báo cáo"])
    
    with tab1:
        st.header("Thu thập dữ liệu Maya Portal")
        
        # Chọn loại dữ liệu
        data_type = st.selectbox(
            "Chọn loại dữ liệu",
            ["Dự án", "Nhân sự", "Khách hàng", "Báo cáo tài chính", "Báo cáo hoạt động"]
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
        
        # Các tùy chọn bổ sung tùy thuộc vào loại dữ liệu được chọn
        if data_type == "Dự án":
            project_status = st.multiselect(
                "Trạng thái dự án",
                ["Tất cả", "Đang thực hiện", "Hoàn thành", "Tạm dừng", "Hủy bỏ"],
                default=["Tất cả"]
            )
            
            project_type = st.multiselect(
                "Loại dự án",
                ["Tất cả", "AI", "Blockchain", "Web/App", "IoT", "Khác"],
                default=["Tất cả"]
            )
        
        elif data_type == "Nhân sự":
            department = st.multiselect(
                "Phòng ban",
                ["Tất cả", "AI Lab", "R&D", "Sales", "Marketing", "HR", "Admin"],
                default=["Tất cả"]
            )
            
            position = st.multiselect(
                "Vị trí",
                ["Tất cả", "Developer", "Researcher", "Manager", "Director", "Admin"],
                default=["Tất cả"]
            )
        
        # Nút để lấy dữ liệu
        if st.button("Lấy dữ liệu", key="get_maya_data"):
            with st.spinner("Đang lấy dữ liệu..."):
                try:
                    # Giả lập lấy dữ liệu thực tế từ service
                    # Trong thực tế, bạn sẽ gọi API thực sự từ Maya Portal
                    data = generate_maya_sample_data(data_type, start_date, end_date)
                    
                    # Lưu dữ liệu vào session state
                    st.session_state.maya_data = data
                    
                    # Hiển thị bảng dữ liệu
                    st.dataframe(data)
                    
                    # Tùy chọn xuất dữ liệu
                    if st.download_button(
                        "Tải dữ liệu (CSV)",
                        data=data.to_csv(index=False).encode('utf-8'),
                        file_name=f"maya_{data_type.lower().replace(' ', '_')}_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    ):
                        st.success("Đã tải xuống dữ liệu thành công!")
                        
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {str(e)}")
        
        # Hiển thị dữ liệu nếu đã có trong session state
        if "maya_data" in st.session_state:
            st.subheader("Dữ liệu đã thu thập")
            st.dataframe(st.session_state.maya_data)
    
    with tab2:
        st.header("Phân tích dữ liệu Maya Portal")
        
        if "maya_data" not in st.session_state:
            st.info("Vui lòng thu thập dữ liệu trước khi phân tích")
            return
        
        # Phân tích dữ liệu đã có
        data = st.session_state.maya_data
        
        # Kiểm tra loại dữ liệu đã thu thập
        if "type" in data.columns:
            data_type = data["type"].iloc[0]
        else:
            data_type = "Dữ liệu chung"
        
        # Hiển thị tổng quan
        st.subheader(f"Phân tích dữ liệu {data_type}")
        
        # Phân tích khác nhau tùy thuộc vào loại dữ liệu
        if data_type == "Dự án":
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng số dự án", f"{len(data):,}")
            col2.metric("Dự án đang thực hiện", f"{(data['status'] == 'Đang thực hiện').sum():,}")
            col3.metric("Dự án hoàn thành", f"{(data['status'] == 'Hoàn thành').sum():,}")
            
            # Biểu đồ phân phối dự án theo trạng thái
            st.subheader("Phân phối dự án theo trạng thái")
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
            
            # Biểu đồ dự án theo thời gian
            st.subheader("Số lượng dự án theo thời gian")
            time_chart = alt.Chart(data).mark_line().encode(
                x='start_date:T',
                y='count():Q',
                tooltip=['start_date:T', 'count():Q']
            ).properties(
                width=700,
                height=400
            )
            st.altair_chart(time_chart, use_container_width=True)
            
            # Biểu đồ ngân sách dự án
            st.subheader("Phân phối ngân sách dự án")
            budget_chart = alt.Chart(data).mark_bar().encode(
                x=alt.X('budget:Q', bin=True),
                y='count():Q',
                tooltip=['count():Q', 'budget:Q']
            ).properties(
                width=700,
                height=300
            )
            st.altair_chart(budget_chart, use_container_width=True)
            
        elif data_type == "Nhân sự":
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng số nhân sự", f"{len(data):,}")
            col2.metric("Nhân sự AI Lab", f"{(data['department'] == 'AI Lab').sum():,}")
            col3.metric("Developer", f"{(data['position'] == 'Developer').sum():,}")
            
            # Biểu đồ phân phối theo phòng ban
            st.subheader("Phân phối nhân sự theo phòng ban")
            dept_data = data['department'].value_counts().reset_index()
            dept_data.columns = ['department', 'count']
            
            dept_chart = alt.Chart(dept_data).mark_pie().encode(
                theta='count:Q',
                color='department:N',
                tooltip=['department:N', 'count:Q']
            ).properties(
                width=500,
                height=500
            )
            st.altair_chart(dept_chart, use_container_width=True)
            
            # Biểu đồ phân phối theo vị trí
            st.subheader("Phân phối nhân sự theo vị trí")
            pos_data = data['position'].value_counts().reset_index()
            pos_data.columns = ['position', 'count']
            
            pos_chart = alt.Chart(pos_data).mark_bar().encode(
                x=alt.X('position:N', sort='-y'),
                y='count:Q',
                color='position:N',
                tooltip=['position:N', 'count:Q']
            ).properties(
                width=700,
                height=400
            )
            st.altair_chart(pos_chart, use_container_width=True)
        
        else:
            # Phân tích chung cho các loại dữ liệu khác
            st.write("Phân tích tổng quan")
            
            # Hiển thị các cột dữ liệu
            st.write("Các trường dữ liệu có sẵn:")
            st.write(", ".join(data.columns))
            
            # Thống kê cơ bản
            st.write("Thống kê mô tả:")
            st.write(data.describe())
            
            # Chọn cột để biểu diễn
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Chọn trường dữ liệu để phân tích:", numeric_cols)
                
                # Biểu đồ phân phối
                st.subheader(f"Phân phối {selected_col}")
                hist_chart = alt.Chart(data).mark_bar().encode(
                    x=alt.X(f'{selected_col}:Q', bin=True),
                    y='count():Q',
                    tooltip=['count():Q', f'{selected_col}:Q']
                ).properties(
                    width=700,
                    height=300
                )
                st.altair_chart(hist_chart, use_container_width=True)
    
    with tab3:
        st.header("Báo cáo Maya Portal")
        
        if "maya_data" not in st.session_state:
            st.info("Vui lòng thu thập dữ liệu trước khi tạo báo cáo")
            return
        
        # Tạo báo cáo từ dữ liệu đã có
        data = st.session_state.maya_data
        
        st.subheader("Tạo báo cáo Maya Portal")
        
        # Kiểm tra loại dữ liệu đã thu thập
        if "type" in data.columns:
            data_type = data["type"].iloc[0]
        else:
            data_type = "Dữ liệu chung"
        
        # Tùy chọn báo cáo
        report_type = st.selectbox(
            "Loại báo cáo",
            ["Tổng quan", "Chi tiết", "So sánh với kỳ trước"]
        )
        
        # Tùy chọn định dạng
        report_format = st.selectbox(
            "Định dạng báo cáo",
            ["PDF", "Excel", "PowerPoint"]
        )
        
        # Thông tin báo cáo
        col1, col2 = st.columns(2)
        with col1:
            report_title = st.text_input("Tiêu đề báo cáo", f"Báo cáo {data_type} - {report_type}")
        with col2:
            author = st.text_input("Tác giả", "Người dùng Maya Portal")
        
        include_charts = st.checkbox("Bao gồm biểu đồ", value=True)
        include_raw_data = st.checkbox("Bao gồm dữ liệu thô", value=False)
        
        # Tạo báo cáo
        if st.button("Tạo báo cáo", key="generate_maya_report"):
            with st.spinner("Đang tạo báo cáo..."):
                # Trong thực tế, bạn sẽ tạo báo cáo thực sự
                # Đây chỉ là mô phỏng
                st.success(f"Đã tạo báo cáo {report_format} thành công!")
                
                if report_format == "Excel":
                    st.download_button(
                        "Tải xuống báo cáo Excel",
                        data=data.to_csv(index=False).encode('utf-8'),  # Trong thực tế, sẽ là file Excel
                        file_name=f"maya_report_{data_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"Báo cáo dạng {report_format} đã được tạo. Vui lòng kiểm tra thư mục tải xuống của bạn.")
        
        # Preview báo cáo
        if report_type == "Tổng quan":
            st.subheader("Xem trước báo cáo")
            
            st.write(f"## {report_title}")
            st.write(f"Tác giả: {author}")
            st.write(f"Ngày tạo: {datetime.now().strftime('%d/%m/%Y')}")
            
            st.write("### Tổng quan")
            
            if data_type == "Dự án":
                st.write(f"Tổng số dự án: {len(data):,}")
                st.write(f"Dự án đang thực hiện: {(data['status'] == 'Đang thực hiện').sum():,}")
                st.write(f"Dự án hoàn thành: {(data['status'] == 'Hoàn thành').sum():,}")
                
                if include_charts:
                    # Biểu đồ trạng thái
                    status_data = data['status'].value_counts().reset_index()
                    status_data.columns = ['status', 'count']
                    
                    status_chart = alt.Chart(status_data).mark_bar().encode(
                        x=alt.X('status:N', sort='-y'),
                        y='count:Q',
                        color='status:N'
                    ).properties(
                        title="Phân phối dự án theo trạng thái",
                        width=400,
                        height=300
                    )
                    st.altair_chart(status_chart, use_container_width=True)
            
            elif data_type == "Nhân sự":
                st.write(f"Tổng số nhân sự: {len(data):,}")
                
                if include_charts:
                    # Biểu đồ phòng ban
                    dept_data = data['department'].value_counts().reset_index()
                    dept_data.columns = ['department', 'count']
                    
                    dept_chart = alt.Chart(dept_data).mark_bar().encode(
                        x=alt.X('department:N', sort='-y'),
                        y='count:Q',
                        color='department:N'
                    ).properties(
                        title="Phân phối nhân sự theo phòng ban",
                        width=400,
                        height=300
                    )
                    st.altair_chart(dept_chart, use_container_width=True)
            
            if include_raw_data:
                st.subheader("Dữ liệu chi tiết")
                st.dataframe(data)

# Hàm tạo dữ liệu mẫu Maya Portal để demo
def generate_maya_sample_data(data_type, start_date, end_date):
    # Tạo dãy ngày
    delta = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(delta)]
    
    # Tạo dữ liệu ngẫu nhiên tùy thuộc vào loại dữ liệu
    np.random.seed(42)  # Đặt seed để có kết quả nhất quán
    
    if data_type == "Dự án":
        n_samples = 50  # 50 dự án
        
        project_names = [f"Project {chr(65 + i)}" for i in range(n_samples)]
        project_types = np.random.choice(["AI", "Blockchain", "Web/App", "IoT", "Khác"], n_samples)
        statuses = np.random.choice(["Đang thực hiện", "Hoàn thành", "Tạm dừng", "Hủy bỏ"], n_samples, p=[0.5, 0.3, 0.1, 0.1])
        start_dates = [start_date + timedelta(days=np.random.randint(0, delta)) for _ in range(n_samples)]
        end_dates = [d + timedelta(days=np.random.randint(30, 180)) for d in start_dates]
        budgets = np.random.uniform(50000000, 500000000, n_samples)
        progress = np.random.uniform(0, 100, n_samples)
        
        data = {
            'project_name': project_names,
            'type': project_types,
            'status': statuses,
            'start_date': start_dates,
            'end_date': end_dates,
            'budget': budgets,
            'progress': progress,
            'project_manager': [f"Manager {i % 10 + 1}" for i in range(n_samples)]
        }
        
        # Đặt kiểu dữ liệu
        for i in range(len(data['project_name'])):
            data['type'] = "Dự án"
        
    elif data_type == "Nhân sự":
        n_samples = 100  # 100 nhân sự
        
        employee_ids = [f"EMP{i:04d}" for i in range(n_samples)]
        names = [f"Employee {i}" for i in range(n_samples)]
        departments = np.random.choice(["AI Lab", "R&D", "Sales", "Marketing", "HR", "Admin"], n_samples)
        positions = np.random.choice(["Developer", "Researcher", "Manager", "Director", "Admin"], n_samples)
        join_dates = [start_date - timedelta(days=np.random.randint(0, 1000)) for _ in range(n_samples)]
        salaries = np.random.normal(25000000, 5000000, n_samples)
        
        data = {
            'employee_id': employee_ids,
            'name': names,
            'department': departments,
            'position': positions,
            'join_date': join_dates,
            'salary': salaries,
            'status': np.random.choice(["Active", "On leave", "Terminated"], n_samples, p=[0.9, 0.05, 0.05])
        }
        
        # Đặt kiểu dữ liệu
        for i in range(len(data['employee_id'])):
            data['type'] = "Nhân sự"
        
    elif data_type == "Khách hàng":
        n_samples = 80  # 80 khách hàng
        
        customer_ids = [f"CUS{i:04d}" for i in range(n_samples)]
        names = [f"Customer {i}" for i in range(n_samples)]
        types = np.random.choice(["Enterprise", "SME", "Startup", "Government"], n_samples)
        join_dates = [start_date - timedelta(days=np.random.randint(0, 500)) for _ in range(n_samples)]
        contract_values = np.random.lognormal(mean=20, sigma=1, size=n_samples) * 1000000
        
        data = {
            'customer_id': customer_ids,
            'name': names,
            'type': types,
            'join_date': join_dates,
            'contract_value': contract_values,
            'status': np.random.choice(["Active", "Inactive"], n_samples, p=[0.8, 0.2]),
            'projects_count': np.random.randint(1, 10, n_samples)
        }
        
    elif data_type == "Báo cáo tài chính":
        n_samples = 12  # 12 tháng
        
        months = [start_date.replace(day=1) + timedelta(days=30*i) for i in range(n_samples)]
        revenues = np.random.normal(5000000000, 1000000000, n_samples)
        expenses = np.random.normal(4000000000, 800000000, n_samples)
        profits = revenues - expenses
        
        data = {
            'month': months,
            'revenue': revenues,
            'expense': expenses,
            'profit': profits,
            'growth_rate': np.random.normal(0.05, 0.02, n_samples),
            'project_count': np.random.randint(5, 20, n_samples)
        }
        
    else:  # Báo cáo hoạt động hoặc khác
        n_samples = 20  # 20 báo cáo
        
        report_ids = [f"REP{i:04d}" for i in range(n_samples)]
        dates = [start_date + timedelta(days=np.random.randint(0, delta)) for _ in range(n_samples)]
        types = np.random.choice(["Monthly", "Quarterly", "Annual", "Special"], n_samples)
        departments = np.random.choice(["AI Lab", "R&D", "Sales", "Marketing", "HR", "Admin"], n_samples)
        
        data = {
            'report_id': report_ids,
            'date': dates,
            'type': types,
            'department': departments,
            'status': np.random.choice(["Draft", "Submitted", "Approved", "Rejected"], n_samples),
            'author': [f"User {i % 10 + 1}" for i in range(n_samples)]
        }
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    
    return df