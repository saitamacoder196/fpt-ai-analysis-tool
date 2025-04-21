import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import os
import sys
from datetime import datetime

# Import các service và utility
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.bachan_service import BAchanService
from utils.data_processor import DataProcessor

title = "BAchan Workplace"

def render():
    st.title("Chat với BAchan Workplace")
    
    # Kiểm tra xem đã có token trong session state chưa
    if "bachan_token" not in st.session_state:
        st.session_state.bachan_token = ""
    
    # Kiểm tra chat history
    if "bachan_messages" not in st.session_state:
        st.session_state.bachan_messages = []
    
    # Form nhập API token
    with st.expander("Cấu hình BAchan API", expanded=not bool(st.session_state.bachan_token)):
        token = st.text_input(
            "BAchan API Token",
            value=st.session_state.bachan_token,
            type="password",
            help="Nhập token API cho BAchan Workplace"
        )
        
        if st.button("Kết nối với BAchan"):
            if token:
                # Trong thực tế, bạn sẽ kiểm tra token có hợp lệ không
                # Đây chỉ là mô phỏng
                with st.spinner("Đang kết nối..."):
                    # Giả lập kết nối thành công
                    time.sleep(1)
                    st.session_state.bachan_token = token
                    st.session_state.bachan_connected = True
                    st.success("Kết nối với BAchan thành công!")
            else:
                st.error("Vui lòng nhập token API")
    
    # Nếu chưa có token, hiển thị hướng dẫn
    if not st.session_state.get("bachan_connected", False):
        if st.session_state.bachan_token:
            # Nếu đã có token nhưng chưa kết nối
            st.warning("Vui lòng nhấn 'Kết nối với BAchan' để tiếp tục")
        else:
            # Nếu chưa có token
            st.info("Vui lòng nhập token API để kết nối với BAchan Workplace")
            
            st.markdown("""
            ### Hướng dẫn lấy BAchan API Token
            
            1. Đăng nhập vào BAchan Workplace
            2. Truy cập phần cài đặt tài khoản
            3. Tạo và sao chép API token
            """)
        return
    
    # Tạo instance của service
    bachan_service = BAchanService(st.session_state.bachan_token)
    
    # Giao diện chat
    st.subheader("Chat với BAchan")
    
    # Container để hiển thị chat
    chat_container = st.container()
    
    # Hiển thị tin nhắn từ lịch sử
    with chat_container:
        for message in st.session_state.bachan_messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant", avatar="🤖").write(message["content"])
    
    # Input cho người dùng
    user_input = st.chat_input("Nhập tin nhắn của bạn...")
    
    # Xử lý tin nhắn nếu có
    if user_input:
        # Hiển thị tin nhắn người dùng
        st.chat_message("user").write(user_input)
        
        # Lưu tin nhắn vào lịch sử
        st.session_state.bachan_messages.append({"role": "user", "content": user_input})
        
        # Xử lý và gửi tin nhắn đến BAchan
        with st.spinner("BAchan đang trả lời..."):
            # Trong thực tế, bạn sẽ gọi API thực sự
            # Đây chỉ là mô phỏng phản hồi
            response = generate_bachan_response(user_input)
            
            # Hiển thị phản hồi
            st.chat_message("assistant", avatar="🤖").write(response)
            
            # Lưu phản hồi vào lịch sử
            st.session_state.bachan_messages.append({"role": "assistant", "content": response})
    
    # Công cụ bổ sung
    st.sidebar.subheader("Công cụ BAchan")
    
    # Tùy chọn xóa lịch sử
    if st.sidebar.button("Xóa lịch sử chat"):
        st.session_state.bachan_messages = []
        st.experimental_rerun()
    
    # Chọn chế độ
    mode = st.sidebar.radio(
        "Chế độ BAchan",
        ["Chatbot", "Trợ lý tác vụ", "Phân tích dữ liệu"]
    )
    
    # Hiển thị các công cụ dựa trên chế độ
    if mode == "Trợ lý tác vụ":
        st.sidebar.subheader("Tác vụ nhanh")
        
        task_type = st.sidebar.selectbox(
            "Chọn loại tác vụ",
            ["Lên lịch họp", "Tạo báo cáo", "Quản lý nhiệm vụ", "Nhắc nhở"]
        )
        
        if task_type == "Lên lịch họp":
            with st.sidebar.form("schedule_meeting"):
                st.subheader("Lên lịch họp")
                meeting_title = st.text_input("Tiêu đề cuộc họp")
                meeting_date = st.date_input("Ngày")
                meeting_time = st.time_input("Giờ")
                participants = st.text_area("Người tham gia (mỗi người một dòng)")
                
                submitted = st.form_submit_button("Tạo lịch họp")
                if submitted:
                    st.success("Đã tạo lịch họp thành công!")
                    
                    # Thêm tin nhắn vào lịch sử chat
                    meeting_info = f"""
                    Đã tạo lịch họp:
                    - Tiêu đề: {meeting_title}
                    - Thời gian: {meeting_date} {meeting_time}
                    - Người tham gia: {participants}
                    """
                    
                    # Hiển thị tin nhắn
                    st.chat_message("assistant", avatar="🤖").write(meeting_info)
                    
                    # Lưu vào lịch sử
                    st.session_state.bachan_messages.append({"role": "assistant", "content": meeting_info})
        
        elif task_type == "Tạo báo cáo":
            with st.sidebar.form("create_report"):
                st.subheader("Tạo báo cáo")
                report_title = st.text_input("Tiêu đề báo cáo")
                report_type = st.selectbox("Loại báo cáo", ["Hàng ngày", "Hàng tuần", "Hàng tháng"])
                department = st.selectbox("Phòng ban", ["AI Lab", "R&D", "Sales", "Marketing", "All"])
                include_charts = st.checkbox("Bao gồm biểu đồ")
                
                submitted = st.form_submit_button("Tạo báo cáo")
                if submitted:
                    st.success("Đã tạo báo cáo thành công!")
                    
                    # Thêm tin nhắn vào lịch sử chat
                    report_info = f"""
                    Đã tạo báo cáo:
                    - Tiêu đề: {report_title}
                    - Loại: {report_type}
                    - Phòng ban: {department}
                    - Biểu đồ: {"Có" if include_charts else "Không"}
                    """
                    
                    # Hiển thị tin nhắn
                    st.chat_message("assistant", avatar="🤖").write(report_info)
                    
                    # Lưu vào lịch sử
                    st.session_state.bachan_messages.append({"role": "assistant", "content": report_info})
    
    elif mode == "Phân tích dữ liệu":
        st.sidebar.subheader("Phân tích dữ liệu với BAchan")
        
        analysis_type = st.sidebar.selectbox(
            "Loại phân tích",
            ["Dữ liệu kinh doanh", "Dữ liệu người dùng", "Dữ liệu hiệu suất"]
        )
        
        data_source = st.sidebar.selectbox(
            "Nguồn dữ liệu",
            ["FPT.AI Data", "Maya Portal", "Upload file Excel/CSV"]
        )
        
        if data_source == "Upload file Excel/CSV":
            uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                st.sidebar.success("Đã tải lên file thành công!")
                
                if st.sidebar.button("Phân tích dữ liệu"):
                    with st.spinner("Đang phân tích dữ liệu..."):
                        # Giả lập phân tích dữ liệu
                        time.sleep(2)
                        
                        # Tạo kết quả phân tích giả lập
                        analysis_result = f"""
                        # Kết quả phân tích {analysis_type}
                        
                        Đã phân tích file: {uploaded_file.name}
                        
                        ## Tổng quan
                        - Số lượng bản ghi: 1,253
                        - Thời gian phân tích: 2.3 giây
                        
                        ## Các phát hiện chính
                        1. Tăng trưởng ổn định 15% so với tháng trước
                        2. Phát hiện 3 điểm dữ liệu ngoại lệ cần kiểm tra
                        3. Xu hướng tăng mạnh trong nhóm khách hàng doanh nghiệp
                        
                        ## Đề xuất
                        - Tập trung vào phân khúc khách hàng doanh nghiệp
                        - Kiểm tra lại các điểm dữ liệu ngoại lệ
                        - Cần thu thập thêm dữ liệu về [chi tiết A] để phân tích sâu hơn
                        """
                        
                        # Hiển thị kết quả trong chat
                        st.chat_message("assistant", avatar="🤖").write(analysis_result)
                        
                        # Lưu vào lịch sử
                        st.session_state.bachan_messages.append({"role": "assistant", "content": analysis_result})
        
        else:
            st.sidebar.info(f"Sử dụng dữ liệu từ {data_source}")
            
            if st.sidebar.button("Lấy và phân tích dữ liệu"):
                with st.spinner(f"Đang lấy và phân tích dữ liệu từ {data_source}..."):
                    # Giả lập phân tích dữ liệu
                    time.sleep(2)
                    
                    # Tạo kết quả phân tích giả lập
                    analysis_result = f"""
                    # Kết quả phân tích từ {data_source}
                    
                    ## Tổng quan
                    - Số lượng bản ghi: 2,561
                    - Thời gian: 30 ngày gần đây
                    
                    ## Các phát hiện chính
                    1. Tăng trưởng 23% so với tháng trước
                    2. Hiệu suất API đạt 98.7%
                    3. Chi phí trung bình giảm 5%
                    
                    ## Đề xuất
                    - Tiếp tục tối ưu hóa API calls
                    - Theo dõi mẫu sử dụng mới của người dùng
                    - Chuẩn bị cho đợt tăng lưu lượng dự kiến vào tháng sau
                    """
                    
                    # Hiển thị kết quả trong chat
                    st.chat_message("assistant", avatar="🤖").write(analysis_result)
                    
                    # Lưu vào lịch sử
                    st.session_state.bachan_messages.append({"role": "assistant", "content": analysis_result})

# Hàm tạo phản hồi giả lập từ BAchan
def generate_bachan_response(user_input):
    # Trong thực tế, bạn sẽ gọi API BAchan và nhận phản hồi thực tế
    # Đây chỉ là mô phỏng phản hồi dựa trên các từ khóa
    
    # Chuyển đổi input sang chữ thường để dễ dàng so sánh
    input_lower = user_input.lower()
    
    # Dictionary chứa các mẫu câu hỏi và phản hồi
    responses = {
        "xin chào": "Xin chào! Tôi là BAchan, trợ lý ảo của FPT AI. Tôi có thể giúp gì cho bạn hôm nay?",
        "giúp": "Tôi có thể giúp bạn với nhiều tác vụ khác nhau như:\n- Thu thập và phân tích dữ liệu\n- Lên lịch họp và quản lý nhiệm vụ\n- Tạo báo cáo tự động\n- Trả lời các câu hỏi về FPT AI\nBạn cần hỗ trợ về vấn đề cụ thể nào?",
        "dữ liệu": "Tôi có thể giúp bạn truy cập và phân tích dữ liệu từ nhiều nguồn khác nhau bao gồm FPT.AI, Maya Portal hoặc từ các file bạn tải lên. Bạn muốn làm việc với dữ liệu nào?",
        "báo cáo": "Tôi có thể giúp bạn tạo các loại báo cáo khác nhau như báo cáo hiệu suất, báo cáo dự án, báo cáo tài chính... Bạn cần loại báo cáo nào?",
        "lịch họp": "Tôi có thể giúp bạn lên lịch họp mới. Vui lòng cung cấp thông tin về tiêu đề, thời gian và người tham gia.",
        "fpt ai": "FPT AI là bộ phận nghiên cứu và phát triển trí tuệ nhân tạo của Tập đoàn FPT. Chúng tôi cung cấp nhiều sản phẩm và dịch vụ AI như xử lý ngôn ngữ tự nhiên, computer vision, chatbot và nhiều giải pháp AI tùy chỉnh khác.",
        "trợ giúp": "Tôi đang ở đây để hỗ trợ bạn! Vui lòng cho tôi biết bạn cần trợ giúp về vấn đề gì, và tôi sẽ cố gắng hết sức để giúp bạn.",
    }
    
    # Kiểm tra từng từ khóa trong input
    for keyword, response in responses.items():
        if keyword in input_lower:
            return response
    
    # Phản hồi mặc định nếu không tìm thấy từ khóa phù hợp
    return "Cảm ơn bạn đã liên hệ. Tôi đang xử lý yêu cầu của bạn. Tôi có thể giúp bạn với các tác vụ liên quan đến dữ liệu FPT AI, quản lý lịch họp, tạo báo cáo và nhiều việc khác. Vui lòng cho tôi biết chi tiết hơn về nhu cầu của bạn."