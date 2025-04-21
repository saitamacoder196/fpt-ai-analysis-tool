import streamlit as st

title = "Home"

def render():
    st.title("FPT AI Analysis Tool")
    
    st.markdown("""
    ## Chào mừng bạn đến với công cụ phân tích dữ liệu FPT AI!
    
    Công cụ này giúp bạn:
    
    - Thu thập và phân tích dữ liệu từ FPT.AI
    - Thu thập và phân tích dữ liệu từ Maya Portal
    - Chat với BAchan workplace
    
    ### Hướng dẫn sử dụng
    
    1. Chọn chức năng cần sử dụng từ menu bên trái
    2. Nhập thông tin xác thực nếu được yêu cầu
    3. Thực hiện truy vấn và phân tích dữ liệu
    
    ### Tính năng chính
    """)
    
    # Grid layout với 3 cột cho 3 tính năng chính
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### FPT.AI Data")
        st.markdown("""
        - Thu thập dữ liệu từ API của FPT.AI
        - Phân tích xu hướng và hiệu suất
        - Tạo báo cáo tự động
        """)
        st.button("Đi đến FPT.AI Data", key="goto_fptai", on_click=lambda: st.session_state.update({"page": "FPT.AI Data"}))
    
    with col2:
        st.success("### Maya Portal")
        st.markdown("""
        - Kết nối với Maya Portal
        - Trích xuất và phân tích dữ liệu
        - Trực quan hóa thông tin
        """)
        st.button("Đi đến Maya Portal", key="goto_maya", on_click=lambda: st.session_state.update({"page": "Maya Portal"}))
    
    with col3:
        st.warning("### BAchan Workplace")
        st.markdown("""
        - Chat với BAchan workplace
        - Truy vấn thông tin
        - Thực hiện các tác vụ tự động
        """)
        st.button("Đi đến BAchan Chat", key="goto_bachan", on_click=lambda: st.session_state.update({"page": "BAchan Workplace"}))
    
    st.markdown("---")
    
    # Thông tin bổ sung
    st.subheader("Thông tin hệ thống")
    st.markdown("""
    - **Phiên bản**: 1.0.0
    - **Cập nhật cuối**: Tháng 4, 2025
    - **Yêu cầu**: Python 3.8+, Streamlit 1.8+
    """)
    
    # Hiển thị một số số liệu thống kê mẫu
    st.subheader("Tổng quan")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="API Calls Today", value="247", delta="23%")
    col2.metric(label="Dữ liệu đã phân tích", value="1.3 GB", delta="-5%")
    col3.metric(label="Báo cáo đã tạo", value="12", delta="2")