import streamlit as st
from pathlib import Path
import importlib
import os
import sys

# Thêm thư mục hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình trang
st.set_page_config(
    page_title="FPT AI Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
def load_css():
    css_file = Path("src/assets/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load các module trang
def load_pages():
    pages = {}
    pages_path = Path("src/pages")
    for file in pages_path.glob("*.py"):
        if file.name.startswith("_"):
            continue
        
        module_name = f"src.pages.{file.stem}"
        page_module = importlib.import_module(module_name)
        
        # Lấy tên hiển thị từ module (nếu có)
        page_title = getattr(page_module, "title", file.stem.replace("_", " ").title())
        pages[page_title] = page_module
    
    return pages

def main():
    # Load CSS
    load_css()
    
    # Hiển thị sidebar
    st.sidebar.title("FPT AI Analysis Tool")
    st.sidebar.image("src/assets/fpt_logo.png", width=150)
    st.sidebar.markdown("---")
    
    # Load các trang
    pages = load_pages()
    
    # Thêm trang Home nếu không có trong danh sách pages
    if "Home" not in pages:
        from src.pages import home
        pages["Home"] = home
    
    # Menu chọn trang
    page_names = list(pages.keys())
    default_page_index = page_names.index("Home") if "Home" in page_names else 0
    selected_page = st.sidebar.selectbox("Chọn chức năng:", page_names, index=default_page_index)
    
    # Hiển thị trang được chọn
    selected_module = pages[selected_page]
    if hasattr(selected_module, "render"):
        selected_module.render()
    else:
        st.error(f"Trang {selected_page} không có hàm render()")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 FPT AI Analysis Tool")

if __name__ == "__main__":
    main()