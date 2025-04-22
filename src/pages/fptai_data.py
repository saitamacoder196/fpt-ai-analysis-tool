import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import requests
import json
import os
import re
import sys
import glob
import warnings
import unicodedata
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import time

# Thư viện cho phương pháp vector
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Tải các tài nguyên NLTK cần thiết
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    warnings.warn("Các thư viện sklearn và nltk chưa được cài đặt. Chỉ sử dụng phương pháp so sánh chuỗi cơ bản.")

# Bỏ qua cảnh báo từ openpyxl
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


title = "FPT.AI Data"

class AzureOpenAIVerifier:
    """
    Lớp để tích hợp với Azure OpenAI để xác minh độ tương đồng dựa trên ngữ nghĩa
    """
    
    def __init__(self, api_key=None, endpoint=None, deployment_name=None):
        """
        Khởi tạo với API key và endpoint của Azure OpenAI
        
        Args:
            api_key (str): API key của Azure OpenAI
            endpoint (str): URL endpoint của Azure OpenAI
            deployment_name (str): Tên deployment model
        """
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.api_version = "2023-05-15"  # Có thể điều chỉnh tùy theo phiên bản Azure OpenAI
        
        self._is_configured = self.api_key and self.endpoint and self.deployment_name
    
    def is_configured(self):
        """Kiểm tra xem API đã được cấu hình chưa"""
        return self._is_configured
    
    def verify_similarity(self, content1, content2):
        """
        Sử dụng Azure OpenAI để xác minh độ tương đồng ngữ nghĩa
        
        Args:
            content1 (str): Nội dung thứ nhất
            content2 (str): Nội dung thứ hai
            
        Returns:
            dict: Kết quả từ Azure OpenAI
        """
        if not self._is_configured:
            return {
                "error": "Azure OpenAI chưa được cấu hình. Vui lòng cung cấp API key, endpoint và deployment name."
            }
        
        # Chuẩn bị URL
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        
        # Chuẩn bị headers
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        # Chuẩn bị prompt
        prompt = self._create_similarity_prompt(content1, content2)
        
        # Chuẩn bị payload
        payload = {
            "messages": [
                {"role": "system", "content": "Bạn là một hệ thống phân tích so sánh văn bản. Nhiệm vụ của bạn là đánh giá độ tương đồng về nội dung và ngữ nghĩa giữa hai đoạn văn bản."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        # Gửi request
        try:
            # Thêm verify=False để bỏ qua xác thực SSL
            response = requests.post(url, headers=headers, json=payload, verify=False)
            response.raise_for_status()
            
            # Xử lý response
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                result_text = response_data["choices"][0]["message"]["content"]
                
                # Trích xuất thông tin từ kết quả
                similarity_info = self._extract_similarity_info(result_text)
                
                return {
                    "success": True,
                    "similarity_score": similarity_info.get("similarity_score", 0),
                    "explanation": similarity_info.get("explanation", ""),
                    "raw_response": result_text
                }
            else:
                return {
                    "success": False,
                    "error": "Không nhận được phản hồi hợp lệ từ Azure OpenAI",
                    "raw_response": response_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_similarity_prompt(self, content1, content2):
        """Tạo prompt cho việc so sánh tương đồng"""
        prompt = f"""
Hãy phân tích hai đoạn văn bản dưới đây và đánh giá độ tương đồng về nội dung và ngữ nghĩa giữa chúng.

Đoạn văn 1:
```
{content1}
```

Đoạn văn 2:
```
{content2}
```

Phân tích theo các tiêu chí sau:
1. Các thông tin chính có khớp nhau không?
2. Các chi tiết quan trọng có được đề cập đầy đủ không?
3. Ý nghĩa tổng thể có tương đồng không?
4. Có sự khác biệt quan trọng nào không?

Sau đó đưa ra:
- Điểm tương đồng (từ 0 đến 100, trong đó 100 là hoàn toàn giống nhau về nội dung)
- Giải thích chi tiết về các điểm tương đồng và khác biệt

Định dạng phản hồi:
ĐIỂM TƯƠNG ĐỒNG: [Điểm số từ 0-100]
GIẢI THÍCH: [Giải thích chi tiết]
"""
        return prompt
    
    def _extract_similarity_info(self, response_text):
        """Trích xuất thông tin từ phản hồi"""
        similarity_score = 0
        explanation = ""
        
        # Tìm điểm tương đồng
        score_match = re.search(r"ĐIỂM TƯƠNG ĐỒNG:\s*(\d+)", response_text)
        if score_match:
            try:
                similarity_score = int(score_match.group(1))
            except:
                similarity_score = 0
        
        # Tìm giải thích
        explanation_match = re.search(r"GIẢI THÍCH:\s*([\s\S]+)", response_text)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        return {
            "similarity_score": similarity_score,
            "explanation": explanation
        }

def render():
    st.title("Thu thập và phân tích dữ liệu từ FPT.AI")
    
    # Chọn bot
    bot_selection = st.selectbox(
        "Chọn Bot",
        ["FJP_VN_Production", "FJP_JP_Production"]
    )
    
    # Tạo đường dẫn đến file JSON dựa trên bot được chọn
    json_file_path = f"data/{bot_selection}.json"
    
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists(json_file_path):
        st.error(f"File {json_file_path} không tồn tại. Vui lòng kiểm tra lại thư mục data/")
        return
    
    # Tab cấu hình
    setup_tab, main_tab = st.tabs(["Cấu hình", "Phân tích dữ liệu"])
    
    with setup_tab:
        # Hiển thị cài đặt so sánh
        st.subheader("Cài đặt so sánh độ tương đồng")
        
        col1, col2 = st.columns(2)
        
        with col1:
            similarity_method = st.selectbox(
                "Phương pháp so sánh",
                ["hybrid", "sequence", "vector", "substring"] if VECTOR_AVAILABLE else ["sequence", "substring"],
                help="hybrid: kết hợp cả phương pháp, sequence: so sánh chuỗi truyền thống, vector: biểu diễn văn bản dưới dạng vector, substring: tìm chuỗi con chung"
            )
        
        with col2:
            similarity_threshold = st.slider(
                "Ngưỡng tương đồng (%)",
                min_value=10,
                max_value=100,
                value=70,
                step=5,
                help="Ngưỡng phần trăm tương đồng để xác định một nội dung khớp với chunks"
            )
        
        # Cấu hình Azure OpenAI
        st.subheader("Cấu hình Azure OpenAI")
        
        st.checkbox(
            "Sử dụng Azure OpenAI để xác minh tương đồng", 
            value=st.session_state.get("use_azure_openai", False),
            key="use_azure_openai",
            help="Sử dụng Azure OpenAI để phân tích ngữ nghĩa và xác minh độ tương đồng giữa step và chunk"
        )
        
        if st.session_state.get("use_azure_openai", False):
            # Đọc giá trị từ session state hoặc sử dụng giá trị mặc định
            azure_api_key = st.session_state.get("azure_openai_api_key", "")
            azure_endpoint = st.session_state.get("azure_openai_endpoint", "")
            azure_deployment = st.session_state.get("azure_openai_deployment", "gpt-4")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_api_key = st.text_input(
                    "Azure OpenAI API Key", 
                    value=azure_api_key,
                    type="password",
                    help="API key của Azure OpenAI"
                )
            
            with col2:
                new_endpoint = st.text_input(
                    "Azure OpenAI Endpoint",
                    value=azure_endpoint,
                    help="URL endpoint của Azure OpenAI (ví dụ: https://your-resource.openai.azure.com/)"
                )
            
            new_deployment = st.text_input(
                "Azure OpenAI Deployment Name",
                value=azure_deployment,
                help="Tên deployment model (ví dụ: gpt-4, gpt-35-turbo)"
            )
            
            # Lưu cấu hình vào session state nếu có thay đổi
            if (new_api_key != azure_api_key or 
                new_endpoint != azure_endpoint or 
                new_deployment != azure_deployment):
                
                st.session_state.azure_openai_api_key = new_api_key
                st.session_state.azure_openai_endpoint = new_endpoint
                st.session_state.azure_openai_deployment = new_deployment
                
                # Kiểm tra cấu hình
                verifier = AzureOpenAIVerifier(
                    api_key=new_api_key,
                    endpoint=new_endpoint,
                    deployment_name=new_deployment
                )
                
                if verifier.is_configured():
                    st.success("Đã lưu cấu hình Azure OpenAI")
                else:
                    st.warning("Cấu hình chưa đầy đủ. Vui lòng cung cấp API key, endpoint và deployment name.")
    
    with main_tab:
        # Đọc dữ liệu chunks từ file Excel mới nhất
        chunks_data = None
        try:
            chunks_file = get_latest_chunks_file()
            if chunks_file:
                st.info(f"Đang đọc dữ liệu chunks từ file: {os.path.basename(chunks_file)}")
                chunks_data = pd.read_excel(chunks_file)
                st.success(f"Đã đọc dữ liệu chunks với {len(chunks_data)} dòng")
            else:
                st.warning("Không tìm thấy file chunks trong thư mục data/chunks/")
        except Exception as e:
            st.warning(f"Không thể đọc file chunks: {str(e)}")
        
        # Đọc dữ liệu từ file JSON
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Kiểm tra xem có dm.scenarios không
            if "dm" not in data or "scenarios" not in data["dm"]:
                st.error("Dữ liệu không đúng định dạng. Không tìm thấy dm.scenarios trong file JSON.")
                return
            
            # Lấy danh sách scenarios
            scenarios = data["dm"]["scenarios"]
            
            # Filter scenarios theo regex pattern
            pattern = r'^\[\d+_[a-zA-Z]+\] \d+\. .+'
            filtered_scenarios = []
            
            for scenario in scenarios:
                # Kiểm tra tên scenario có match với pattern không
                if re.match(pattern, scenario.get("name", "")):
                    filtered_scenarios.append(scenario)
            
            # Hiển thị số lượng scenarios tìm thấy
            st.success(f"Tìm thấy {len(filtered_scenarios)} scenarios thỏa mãn trong {bot_selection}")
            
            # Nút Apply AI cho toàn bộ scenarios
            if st.session_state.get("use_azure_openai", False):
                all_scenarios_container = st.container()
                
                with all_scenarios_container:
                    st.subheader("Xác minh AI cho toàn bộ Bot")
                    
                    if st.button("Verify All Scenarios with AI", type="primary"):
                        # Kiểm tra cấu hình Azure OpenAI
                        if not st.session_state.get("use_azure_openai", False):
                            st.error("Vui lòng kích hoạt và cấu hình Azure OpenAI trong tab 'Cấu hình' trước!")
                        else:
                            # Tạo đối tượng verifier
                            verifier = AzureOpenAIVerifier(
                                api_key=st.session_state.get("azure_openai_api_key"),
                                endpoint=st.session_state.get("azure_openai_endpoint"),
                                deployment_name=st.session_state.get("azure_openai_deployment")
                            )
                            
                            if not verifier.is_configured():
                                st.error("Cấu hình Azure OpenAI chưa đầy đủ. Vui lòng cấu hình trong tab 'Cấu hình'.")
                            else:
                                # Chuẩn bị dữ liệu để làm việc với tất cả scenarios
                                all_steps_data = []
                                total_steps = 0
                                
                                # Đếm tổng số steps để hiển thị tiến trình
                                for s in filtered_scenarios:
                                    active_steps = [step for step in s.get("steps", []) if step.get("activate", 0) == 1]
                                    total_steps += len(active_steps)
                                
                                # Tạo progress bar cho việc xác minh AI
                                progress_bar = st.progress(0, text="Đang phân tích tất cả scenarios...")
                                
                                # Tạo bảng thông báo để hiển thị trạng thái
                                status_container = st.empty()
                                status_container.info("Bắt đầu xác minh toàn bộ dữ liệu...")
                                
                                # Đếm số lượng steps đã xác minh
                                verified_count = 0
                                current_step = 0
                                
                                # Duyệt qua từng scenario
                                for scenario_index, scenario in enumerate(filtered_scenarios):
                                    # Cập nhật trạng thái
                                    scenario_name = scenario.get("name", "")
                                    status_container.info(f"Đang xử lý scenario {scenario_index+1}/{len(filtered_scenarios)}: {scenario_name}")
                                    
                                    # Trích xuất department từ tên scenario
                                    department_match = re.search(r'^\[(\d+)_([a-zA-Z]+)\]', scenario_name)
                                    department = department_match.group(2) if department_match else "Unknown"
                                    
                                    # Lấy steps có activate=1
                                    active_steps = [step for step in scenario.get("steps", []) if step.get("activate", 0) == 1]
                                    
                                    # Duyệt qua từng step
                                    for step in active_steps:
                                        # Cập nhật progress bar
                                        current_step += 1
                                        progress = current_step / total_steps
                                        progress_bar.progress(progress, text=f"Đang xác minh... ({current_step}/{total_steps} steps)")
                                        
                                        # Tạo nội dung từ text và quick_reply cards
                                        content = get_step_content(step)
                                        
                                        # Xác định kiểu step
                                        step_type = determine_step_type(content)
                                        
                                        # Tìm kiếm trong chunks (nếu có)
                                        matching_chunk = None
                                        if chunks_data is not None and content.strip() != "":
                                            matching_chunk = find_matching_chunk(
                                                content, chunks_data, bot_selection, 
                                                similarity_method=similarity_method,
                                                similarity_threshold=similarity_threshold
                                            )
                                        
                                        # Tạo dictionary cho step
                                        step_dict = {
                                            "name": step.get("name", "Không có tên"),
                                            "code": step["code"],
                                            "position": step.get("position", 0),
                                            "content": content,
                                            "type": step_type,
                                            "has_match": matching_chunk is not None,
                                            "document_name": matching_chunk["Document name(*)"] if matching_chunk is not None else "",
                                            "similarity_score": matching_chunk["similarity_score"] if matching_chunk is not None else 0,
                                            "similarity_details": matching_chunk.get("similarity_details", {}) if matching_chunk is not None else {},
                                            "scenario_name": scenario_name,
                                            "department": department,
                                            "ai_verified": False,
                                            "ai_score": 0,
                                            "ai_explanation": ""
                                        }
                                        
                                        # Xác minh AI nếu có match
                                        if step_dict["has_match"] and chunks_data is not None:
                                            # Tìm chunk tương ứng
                                            matching_rows = chunks_data[chunks_data["Document name(*)"] == step_dict["document_name"]]
                                            
                                            if not matching_rows.empty:
                                                chunk_content = matching_rows.iloc[0]["Content(*)"]
                                                
                                                # Tiến hành xác minh
                                                try:
                                                    result = verifier.verify_similarity(content, chunk_content)
                                                    
                                                    if result.get("success", False):
                                                        # Cập nhật thông tin
                                                        step_dict["ai_verified"] = True
                                                        step_dict["ai_score"] = result["similarity_score"]
                                                        step_dict["ai_explanation"] = result["explanation"]
                                                        verified_count += 1
                                                    else:
                                                        step_dict["ai_verified"] = False
                                                        step_dict["ai_score"] = 0
                                                        step_dict["ai_explanation"] = f"Lỗi: {result.get('error', 'Unknown error')}"
                                                except Exception as e:
                                                    step_dict["ai_verified"] = False
                                                    step_dict["ai_score"] = 0
                                                    step_dict["ai_explanation"] = f"Lỗi ngoại lệ: {str(e)}"
                                                
                                                # Thêm độ trễ nhỏ để tránh vượt quá rate limit
                                                time.sleep(0.5)
                                        
                                        # Thêm step vào danh sách
                                        all_steps_data.append(step_dict)
                                
                                # Hoàn thành progress bar
                                progress_bar.progress(100, text="Xác minh toàn bộ dữ liệu hoàn tất!")
                                status_container.success(f"Đã xác minh xong {verified_count}/{len(all_steps_data)} steps!")
                                
                                # Tạo DataFrame từ dữ liệu
                                all_steps_df = pd.DataFrame(all_steps_data)
                                
                                # Lưu dữ liệu vào session state để sử dụng ở nơi khác
                                st.session_state.all_verified_steps = all_steps_df
                                
                                # Tạo các nút Export
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Export tất cả steps
                                    export_all_df = pd.DataFrame({
                                        "Department": all_steps_df["department"],
                                        "Scenario": all_steps_df["scenario_name"],
                                        "Step Name": all_steps_df["name"],
                                        "Step Code": all_steps_df["code"],
                                        "Step Type": all_steps_df["type"],
                                        "Position": all_steps_df["position"],
                                        "Has Match in Chunks": ["Có" if has_match else "Không" for has_match in all_steps_df["has_match"]],
                                        "Document Name": all_steps_df["document_name"],
                                        "Config Similarity": [f"{score:.2f}%" if score > 0 else "" for score in all_steps_df["similarity_score"]],
                                        "AI Verified": ["Có" if verified else "Không" for verified in all_steps_df["ai_verified"]],
                                        "AI Score": [f"{score}%" if score > 0 else "" for score in all_steps_df["ai_score"]]
                                    })
                                    
                                    csv = export_all_df.to_csv(index=False).encode('utf-8')
                                    filename = f"{bot_selection}_all_steps_with_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                    
                                    st.download_button(
                                        label="Export All Steps",
                                        data=csv,
                                        file_name=filename,
                                        mime="text/csv",
                                    )
                                
                                with col2:
                                    # Export chỉ FAQs không có trong chunks
                                    faqs_missing = all_steps_df[(all_steps_df["type"] == "Answer") & (~all_steps_df["has_match"])]
                                    
                                    if not faqs_missing.empty:
                                        export_missing_df = pd.DataFrame({
                                            "Department": faqs_missing["department"],
                                            "Scenario": faqs_missing["scenario_name"],
                                            "Step Name": faqs_missing["name"],
                                            "Step Code": faqs_missing["code"],
                                            "Content": faqs_missing["content"],
                                            "Config Highest Similarity": [get_highest_similarity(content, chunks_data, bot_selection, similarity_method)[0] if chunks_data is not None else 0 for content in faqs_missing["content"]],
                                            "Best Matching Document": [get_highest_similarity(content, chunks_data, bot_selection, similarity_method)[1] if chunks_data is not None else "" for content in faqs_missing["content"]]
                                        })
                                        
                                        csv = export_missing_df.to_csv(index=False).encode('utf-8')
                                        filename = f"{bot_selection}_missing_faqs_with_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                        
                                        st.download_button(
                                            label="Export Missing FAQs",
                                            data=csv,
                                            file_name=filename,
                                            mime="text/csv",
                                        )
                                    else:
                                        st.info("Không có FAQs nào chưa có trong chunks.")
                                
                                with col3:
                                    # Export dữ liệu đã verify AI đầy đủ
                                    verified_steps = all_steps_df[all_steps_df["ai_verified"] == True]
                                    if not verified_steps.empty:
                                        export_verified_df = pd.DataFrame({
                                            "Department": verified_steps["department"],
                                            "Scenario": verified_steps["scenario_name"],
                                            "Step Name": verified_steps["name"],
                                            "Step Code": verified_steps["code"],
                                            "Step Type": verified_steps["type"],
                                            "Has Match in Chunks": ["Có" if has_match else "Không" for has_match in verified_steps["has_match"]],
                                            "Document Name": verified_steps["document_name"],
                                            "Config Similarity": [f"{score:.2f}%" if score > 0 else "" for score in verified_steps["similarity_score"]],
                                            "AI Score": [f"{score}%" if score > 0 else "" for score in verified_steps["ai_score"]],
                                            "AI Explanation": verified_steps["ai_explanation"]
                                        })
                                        
                                        csv = export_verified_df.to_csv(index=False).encode('utf-8')
                                        filename = f"{bot_selection}_verified_steps_with_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                        
                                        st.download_button(
                                            label="Export AI Verified Steps",
                                            data=csv,
                                            file_name=filename,
                                            mime="text/csv",
                                        )
                                    else:
                                        st.info("Chưa có step nào được verify bởi AI.")
                                
                                # Hiển thị thống kê
                                st.subheader("Thống kê kết quả xác minh AI")
                                
                                # 1. Phân bố điểm AI
                                verified_steps = all_steps_df[all_steps_df["ai_verified"] == True]
                                
                                if not verified_steps.empty:
                                    # Phân loại theo điểm số
                                    high_quality = len(verified_steps[verified_steps["ai_score"] >= 80])
                                    medium_quality = len(verified_steps[(verified_steps["ai_score"] >= 50) & (verified_steps["ai_score"] < 80)])
                                    low_quality = len(verified_steps[verified_steps["ai_score"] < 50])
                                    
                                    # Tạo DataFrame cho biểu đồ
                                    quality_df = pd.DataFrame({
                                        "Chất lượng": ["Cao (80-100%)", "Trung bình (50-79%)", "Thấp (0-49%)"],
                                        "Số lượng": [high_quality, medium_quality, low_quality]
                                    })
                                    
                                    # Tạo biểu đồ
                                    quality_chart = alt.Chart(quality_df).mark_bar().encode(
                                        x=alt.X('Chất lượng:N', title=""),
                                        y=alt.Y('Số lượng:Q'),
                                        color=alt.Color('Chất lượng:N', scale=alt.Scale(
                                            domain=['Cao (80-100%)', 'Trung bình (50-79%)', 'Thấp (0-49%)'],
                                            range=['#4CAF50', '#FFC107', '#F44336']
                                        )),
                                        tooltip=['Chất lượng', 'Số lượng']
                                    ).properties(
                                        width=600,
                                        height=400,
                                        title="Phân bố chất lượng nội dung theo đánh giá AI"
                                    )
                                    
                                    st.altair_chart(quality_chart, use_container_width=True)
                                    
                                    # Hiển thị thông tin tỷ lệ
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        total_verified = len(verified_steps)
                                        high_percent = (high_quality / total_verified) * 100 if total_verified > 0 else 0
                                        st.metric("Chất lượng cao", f"{high_quality} steps ({high_percent:.1f}%)")
                                    
                                    with col2:
                                        medium_percent = (medium_quality / total_verified) * 100 if total_verified > 0 else 0
                                        st.metric("Chất lượng trung bình", f"{medium_quality} steps ({medium_percent:.1f}%)")
                                    
                                    with col3:
                                        low_percent = (low_quality / total_verified) * 100 if total_verified > 0 else 0
                                        st.metric("Chất lượng thấp", f"{low_quality} steps ({low_percent:.1f}%)")
                                else:
                                    st.warning("Chưa có steps nào được xác minh bằng AI.")
            
            # Tab cho các phần khác nhau
            tab1, tab2 = st.tabs(["Phân tích kịch bản", "Thống kê tổng thể"])
            
            with tab1:
                # Hiển thị danh sách scenarios
                scenario_names = [scenario["name"] for scenario in filtered_scenarios]
                
                # Tạo DataFrame cho scenarios
                scenario_df = pd.DataFrame({
                    "Tên Scenario": scenario_names,
                    "Code": [scenario["code"] for scenario in filtered_scenarios],
                    "Vị trí": [scenario.get("position", 0) for scenario in filtered_scenarios]
                })
                
                st.subheader("Danh sách Scenarios")
                st.dataframe(scenario_df, use_container_width=True)
                
                # Cho phép người dùng chọn một scenario để xem chi tiết
                selected_scenario_name = st.selectbox("Chọn Scenario để xem chi tiết", scenario_names)
                
                # Lưu lại scenario_name được chọn
                if "selected_scenario_name" not in st.session_state or st.session_state.selected_scenario_name != selected_scenario_name:
                    st.session_state.selected_scenario_name = selected_scenario_name
                    # Reset step data khi thay đổi scenario
                    if "step_df" in st.session_state:
                        del st.session_state.step_df
                
                # Tìm scenario được chọn
                selected_scenario = next((s for s in filtered_scenarios if s["name"] == selected_scenario_name), None)
                
                if selected_scenario:
                    st.subheader(f"Chi tiết Scenario: {selected_scenario_name}")
                    
                    # Lấy steps có activate=1
                    active_steps = [step for step in selected_scenario.get("steps", []) if step.get("activate", 0) == 1]
                    
                    if not active_steps:
                        st.warning("Không có step nào được kích hoạt trong scenario này.")
                        return
                    
                    # Kiểm tra nếu step_df chưa có trong session_state hoặc kịch bản đã thay đổi
                    if "step_df" not in st.session_state:
                        # Tạo và xử lý danh sách steps
                        step_data = []
                        
                        # Tạo progress bar
                        progress_text = "Đang phân tích các steps..."
                        if chunks_data is not None:
                            my_bar = st.progress(0, text=progress_text)
                        
                        # Duyệt qua từng step và phân tích
                        for i, step in enumerate(active_steps):
                            # Cập nhật progress bar
                            if chunks_data is not None:
                                progress = (i + 1) / len(active_steps)
                                my_bar.progress(progress, text=f"{progress_text} ({i+1}/{len(active_steps)})")
                            
                            # Tạo nội dung từ text và quick_reply cards
                            content = get_step_content(step)
                            
                            # Xác định kiểu step
                            step_type = determine_step_type(content)
                            
                            # Tìm kiếm trong chunks (nếu có)
                            matching_chunk = None
                            if chunks_data is not None and content.strip() != "":
                                matching_chunk = find_matching_chunk(
                                    content, chunks_data, bot_selection, 
                                    similarity_method=similarity_method,
                                    similarity_threshold=similarity_threshold
                                )
                            
                            step_data.append({
                                "name": step.get("name", "Không có tên"),
                                "code": step["code"],
                                "position": step.get("position", 0),
                                "content": content,
                                "type": step_type,
                                "has_match": matching_chunk is not None,
                                "document_name": matching_chunk["Document name(*)"] if matching_chunk is not None else "",
                                "similarity_score": matching_chunk["similarity_score"] if matching_chunk is not None else 0,
                                "similarity_details": matching_chunk.get("similarity_details", {}) if matching_chunk is not None else {},
                                "scenario_name": selected_scenario_name,
                                "ai_verified": False,  # Thêm trường để theo dõi việc xác minh AI
                                "ai_score": 0,         # Thêm trường để lưu điểm AI
                                "ai_explanation": ""   # Thêm trường để lưu giải thích từ AI
                            })
                        
                        # Hoàn thành progress bar
                        if chunks_data is not None:
                            my_bar.progress(100, text="Phân tích hoàn tất!")
                            st.success("Đã phân tích xong tất cả các steps!")
                        
                        # Tạo DataFrame cho steps và lưu vào session_state
                        st.session_state.step_df = pd.DataFrame(step_data)
                    
                    # Sử dụng step_df từ session_state
                    step_df = st.session_state.step_df
                    
                    # Nút export và verify AI
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Export Steps Data", type="primary"):
                            # Tạo DataFrame cho export
                            export_df = pd.DataFrame({
                                "Scenario": step_df["scenario_name"],
                                "Step Name": step_df["name"],
                                "Step Code": step_df["code"],
                                "Step Type": step_df["type"],
                                "Position": step_df["position"],
                                "Has Match in Chunks": ["Có" if has_match else "Không" for has_match in step_df["has_match"]],
                                "Document Name": step_df["document_name"],
                                "Config Similarity": [f"{score:.2f}%" if score > 0 else "" for score in step_df["similarity_score"]],
                                "AI Verified": ["Có" if verified else "Không" for verified in step_df["ai_verified"]],
                                "AI Score": [f"{score}%" if score > 0 else "" for score in step_df["ai_score"]]
                            })
                            
                            # Tạo CSV để download
                            csv = export_df.to_csv(index=False).encode('utf-8')
                            filename = f"{bot_selection}_{selected_scenario_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            
                            # Hiển thị tải xuống
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=filename,
                                mime="text/csv",
                            )
                    
                    with col2:
                        # Thêm nút để xác minh AI cho tất cả các steps
                        if st.button("Verify All Steps with AI", type="primary"):
                            # Kiểm tra cấu hình Azure OpenAI
                            if not st.session_state.get("use_azure_openai", False):
                                st.error("Vui lòng kích hoạt và cấu hình Azure OpenAI trong tab 'Cấu hình' trước!")
                            else:
                                # Tạo đối tượng verifier
                                verifier = AzureOpenAIVerifier(
                                    api_key=st.session_state.get("azure_openai_api_key"),
                                    endpoint=st.session_state.get("azure_openai_endpoint"),
                                    deployment_name=st.session_state.get("azure_openai_deployment")
                                )
                                
                                if not verifier.is_configured():
                                    st.error("Cấu hình Azure OpenAI chưa đầy đủ. Vui lòng cấu hình trong tab 'Cấu hình'.")
                                else:
                                    # Tạo progress bar cho việc xác minh AI
                                    progress_bar = st.progress(0, text="Đang xác minh steps với AI...")
                                    
                                    # Tạo bảng thông báo để hiển thị trạng thái
                                    status_container = st.empty()
                                    status_container.info("Bắt đầu xác minh...")
                                    
                                    # Đếm số lượng steps đã xác minh thành công
                                    verified_count = 0
                                    
                                    # Duyệt qua từng step trong DataFrame
                                    updated_step_data = step_df.to_dict('records')
                                    for i, step_row in enumerate(updated_step_data):
                                        # Cập nhật progress bar
                                        progress = (i + 1) / len(updated_step_data)
                                        progress_bar.progress(progress, text=f"Đang xác minh steps với AI... ({i+1}/{len(updated_step_data)})")
                                        
                                        # Cập nhật trạng thái
                                        status_container.info(f"Đang xác minh step: {step_row['name']} ({i+1}/{len(updated_step_data)})")
                                        
                                        # Chỉ xác minh steps có match trong chunks
                                        if step_row["has_match"] and chunks_data is not None:
                                            # Tìm nội dung chunk tương ứng
                                            matching_row = chunks_data[chunks_data["Document name(*)"] == step_row["document_name"]]
                                            
                                            if not matching_row.empty:
                                                chunk_content = matching_row.iloc[0]["Content(*)"]
                                                
                                                # Tiến hành xác minh
                                                try:
                                                    result = verifier.verify_similarity(step_row["content"], chunk_content)
                                                    
                                                    if result.get("success", False):
                                                        # Cập nhật thông tin
                                                        step_row["ai_verified"] = True
                                                        step_row["ai_score"] = result["similarity_score"]
                                                        step_row["ai_explanation"] = result["explanation"]
                                                        verified_count += 1
                                                    else:
                                                        step_row["ai_verified"] = False
                                                        step_row["ai_score"] = 0
                                                        step_row["ai_explanation"] = f"Lỗi: {result.get('error', 'Unknown error')}"
                                                except Exception as e:
                                                    step_row["ai_verified"] = False
                                                    step_row["ai_score"] = 0
                                                    step_row["ai_explanation"] = f"Lỗi ngoại lệ: {str(e)}"
                                                
                                                # Thêm độ trễ nhỏ để tránh vượt quá rate limit
                                                time.sleep(0.5)
                                    
                                    # Cập nhật DataFrame
                                    st.session_state.step_df = pd.DataFrame(updated_step_data)
                                    step_df = st.session_state.step_df
                                    
                                    # Hoàn thành progress bar
                                    progress_bar.progress(100, text="Xác minh hoàn tất!")
                                    status_container.success(f"Đã xác minh xong {verified_count}/{len(updated_step_data)} steps có match!")
                                    
                                    # Thêm nút để tải xuống dữ liệu đã verify
                                    st.subheader("Xuất dữ liệu đã verify AI")
                                    verified_steps = step_df[step_df["ai_verified"] == True]
                                    if not verified_steps.empty:
                                        export_verified_df = pd.DataFrame({
                                            "Scenario": verified_steps["scenario_name"],
                                            "Step Name": verified_steps["name"],
                                            "Step Code": verified_steps["code"],
                                            "Step Type": verified_steps["type"],
                                            "Has Match in Chunks": ["Có" if has_match else "Không" for has_match in verified_steps["has_match"]],
                                            "Document Name": verified_steps["document_name"],
                                            "Config Similarity": [f"{score:.2f}%" if score > 0 else "" for score in verified_steps["similarity_score"]],
                                            "AI Score": [f"{score}%" if score > 0 else "" for score in verified_steps["ai_score"]],
                                            "AI Explanation": verified_steps["ai_explanation"]
                                        })
                                        
                                        csv = export_verified_df.to_csv(index=False).encode('utf-8')
                                        filename = f"{bot_selection}_{selected_scenario_name.replace(' ', '_')}_verified_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                        
                                        st.download_button(
                                            label="Download Verified AI Data",
                                            data=csv,
                                            file_name=filename,
                                            mime="text/csv",
                                        )
                                    else:
                                        st.info("Không có step nào được xác minh bởi AI.")
                    
                    # Hiển thị dataframe với cột tùy chỉnh
                    step_display_df = pd.DataFrame({
                        "Tên Step": step_df["name"],
                        "Code": step_df["code"],
                        "Vị trí": step_df["position"],
                        "Loại": step_df["type"],
                        "Có trong Chunks": ["Có" if has_match else "Không" for has_match in step_df["has_match"]],
                        "Document Name": step_df["document_name"],
                        "Độ tương đồng (Config)": [f"{score:.2f}%" if score > 0 else "" for score in step_df["similarity_score"]],
                        "Đã xác minh AI": ["Có" if verified else "" for verified in step_df["ai_verified"]],
                        "Điểm AI": [f"{score}%" if score > 0 else "" for score in step_df["ai_score"]]
                    })
                    
                    st.subheader("Danh sách Steps")
                    st.dataframe(step_display_df, use_container_width=True)
                    
                    # Cho phép người dùng chọn một step để xem nội dung
                    selected_step_name = st.selectbox("Chọn Step để xem nội dung", step_df["name"].tolist())
                    
                    # Tìm step được chọn
                    selected_step_data = step_df[step_df["name"] == selected_step_name].iloc[0] if not step_df[step_df["name"] == selected_step_name].empty else None
                    selected_step = next((s for s in active_steps if s.get("name", "") == selected_step_name), None)
                    
                    if selected_step and selected_step_data is not None:
                        st.subheader(f"Nội dung Step: {selected_step_name}")
                        
                        # Hiển thị thông tin về loại step
                        st.info(f"Loại Step: {selected_step_data['type']}")
                        
                        # Hiển thị nội dung
                        st.text_area("Content", selected_step_data["content"], height=300)
                        
                        # Tìm kiếm trong chunks và hiển thị kết quả
                        if chunks_data is not None and selected_step_data["content"].strip() != "":
                            if selected_step_data["has_match"]:
                                st.success(f"Nội dung này có trong data chunks! (Độ tương đồng theo config: {selected_step_data['similarity_score']:.2f}%)")
                                
                                # Hiển thị kết quả xác minh AI nếu có
                                if selected_step_data["ai_verified"]:
                                    ai_score = selected_step_data["ai_score"]
                                    ai_explanation = selected_step_data["ai_explanation"]
                                    
                                    # Xác định màu dựa trên điểm AI
                                    if ai_score >= 80:
                                        st.success(f"Xác minh AI: {ai_score}% tương đồng ✓")
                                    elif ai_score >= 50:
                                        st.warning(f"Xác minh AI: {ai_score}% tương đồng ⚠")
                                    else:
                                        st.error(f"Xác minh AI: {ai_score}% tương đồng ✗")
                                    
                                    # Hiển thị giải thích
                                    with st.expander("Xem giải thích từ AI"):
                                        st.write(ai_explanation)
                                
                                # Hiển thị thông tin document khớp
                                st.subheader("Thông tin Document khớp")
                                
                                # Tìm thông tin chunk dựa trên document_name
                                matching_chunk = chunks_data[chunks_data["Document name(*)"] == selected_step_data["document_name"]].iloc[0] if not chunks_data[chunks_data["Document name(*)"] == selected_step_data["document_name"]].empty else None
                                
                                if matching_chunk is not None:
                                    # Lấy thông tin từ các cột chính
                                    chunk_info = {
                                        "ID": matching_chunk.name,
                                        "Document name": matching_chunk["Document name(*)"],
                                        "Category": matching_chunk["Category(*)"],
                                        "Subcategory": matching_chunk["Subcategory(*)"],
                                        "Language": matching_chunk["Language(*)"],
                                        "Source": matching_chunk["Source(*)"],
                                        "Region": matching_chunk["Region(*)"],
                                        "Published date": matching_chunk["Published date(*)"]
                                    }
                                    
                                    # Hiển thị thông tin chunk
                                    for key, value in chunk_info.items():
                                        st.write(f"**{key}:** {value}")
                                    
                                    # Hiển thị nội dung chunk
                                    st.text_area("Nội dung Chunk", matching_chunk["Content(*)"], height=300)
                                    
                                    # Hiển thị phân tích nâng cao
                                    render_advanced_comparison(
                                        st, 
                                        selected_step_data, 
                                        matching_chunk, 
                                        bot_selection
                                    )
                                    
                                    # Thêm nút xác minh AI cho step hiện tại nếu chưa được xác minh
                                    # (Chỉ hiển thị nếu step chưa được xác minh)
                                    if not selected_step_data["ai_verified"] and st.session_state.get("use_azure_openai", False):
                                        if st.button("Xác minh step này với AI"):
                                            if not st.session_state.get("use_azure_openai", False):
                                                st.error("Vui lòng kích hoạt và cấu hình Azure OpenAI trong tab 'Cấu hình' trước!")
                                            else:
                                                with st.spinner("Đang phân tích bằng AI..."):
                                                    verifier = AzureOpenAIVerifier(
                                                        api_key=st.session_state.get("azure_openai_api_key"),
                                                        endpoint=st.session_state.get("azure_openai_endpoint"),
                                                        deployment_name=st.session_state.get("azure_openai_deployment")
                                                    )
                                                    
                                                    if verifier.is_configured():
                                                        result = verifier.verify_similarity(selected_step_data["content"], matching_chunk["Content(*)"])
                                                        
                                                        if result.get("success", False):
                                                            # Cập nhật thông tin trong DataFrame
                                                            idx = step_df[step_df["name"] == selected_step_name].index[0]
                                                            st.session_state.step_df.at[idx, "ai_verified"] = True
                                                            st.session_state.step_df.at[idx, "ai_score"] = result["similarity_score"]
                                                            st.session_state.step_df.at[idx, "ai_explanation"] = result["explanation"]
                                                            
                                                            # Hiển thị kết quả
                                                            ai_score = result["similarity_score"]
                                                            if ai_score >= 80:
                                                                st.success(f"Điểm tương đồng theo AI: {ai_score}/100 ✓")
                                                            elif ai_score >= 50:
                                                                st.warning(f"Điểm tương đồng theo AI: {ai_score}/100 ⚠")
                                                            else:
                                                                st.error(f"Điểm tương đồng theo AI: {ai_score}/100 ✗")
                                                            
                                                            st.write("**Giải thích:**")
                                                            st.write(result["explanation"])
                                                            
                                                            # Yêu cầu người dùng rerun để cập nhật UI
                                                            st.rerun()
                                                        else:
                                                            st.error(f"Lỗi khi phân tích: {result.get('error', 'Unknown error')}")
                                                    else:
                                                        st.warning("Azure OpenAI chưa được cấu hình đầy đủ.")
                            else:
                                st.warning("Không tìm thấy nội dung này trong data chunks")
                                
                                # Hiển thị điểm tương đồng cao nhất nếu có
                                if chunks_data is not None:
                                    highest_score, highest_doc, details = get_highest_similarity(
                                        selected_step_data["content"], 
                                        chunks_data, 
                                        bot_selection,
                                        similarity_method
                                    )
                                    st.info(f"Điểm tương đồng cao nhất: {highest_score:.2f}% với document: {highest_doc}")
                                    st.write("(Ngưỡng hiện tại: " + str(similarity_threshold) + "%)")
                                    
                                    # Hiển thị thêm chi tiết về chuỗi con nếu có
                                    if "substring_details" in details and similarity_method in ["substring", "hybrid"]:
                                        substring_details = details["substring_details"]
                                        if "common_substrings" in substring_details and substring_details["common_substrings"]:
                                            st.write("**Các cụm từ chung tiềm năng:**")
                                            for i, phrase in enumerate(substring_details["common_substrings"][:3]):
                                                st.write(f"{i+1}. {phrase}")
                        
                        # Hiển thị thông tin chi tiết về cards
                        st.subheader("Chi tiết Cards")
                        
                        active_cards = []
                        for card in selected_step.get("cards", []):
                            try:
                                if card.get("activate", 0) == 1:
                                    card_config = json.loads(card.get("config", "{}"))
                                    active_cards.append({
                                        "card": card,
                                        "config": card_config
                                    })
                            except json.JSONDecodeError:
                                st.warning(f"Không thể parse JSON config cho card {card.get('name', '')}")
                        
                        # Sắp xếp cards theo position
                        active_cards = sorted(active_cards, key=lambda c: c["card"].get("position", 0))
                        
                        for i, card_data in enumerate(active_cards):
                            card = card_data["card"]
                            config = card_data["config"]
                            
                            card_type = card.get("card_type_id", "unknown")
                            
                            if card_type in ["text", "quick_reply"]:
                                with st.expander(f"{i+1}. {card.get('name', '')} ({card_type})"):
                                    st.write(f"**Code:** {card.get('code', '')}")
                                    st.write(f"**Position:** {card.get('position', 0)}")
                                    
                                    if "text" in config:
                                        st.text_area(f"Text content", config["text"], height=150)
                                    
                                    if card_type == "quick_reply" and "buttons" in config:
                                        st.subheader("Buttons")
                                        for btn in config["buttons"]:
                                            st.write(f"- **{btn.get('title', '')}**: {btn.get('payload', '')}")
                    else:
                        st.error("Không thể tìm thấy step đã chọn.")
                else:
                    st.error("Không thể tìm thấy scenario đã chọn.")
                    
            with tab2:
                st.subheader("Thống kê tổng thể")
                
                # Nếu đã có phân tích từ Verify All Scenarios, hiển thị nút để sử dụng lại
                if "all_verified_steps" in st.session_state:
                    if st.button("Hiển thị kết quả phân tích từ Verify All Scenarios"):
                        st.info("Đang hiển thị kết quả phân tích từ Verify All Scenarios đã thực hiện trước đó.")
                        
                        # Lấy dữ liệu từ session state
                        all_steps_df = st.session_state.all_verified_steps
                        
                        # Phân tích theo bộ phận
                        department_scenarios = {}
                        for dept in all_steps_df["department"].unique():
                            # Đếm số lượng scenarios duy nhất cho mỗi department
                            unique_scenarios = all_steps_df[all_steps_df["department"] == dept]["scenario_name"].nunique()
                            department_scenarios[dept] = unique_scenarios
                        
                        # Đếm số lượng steps cho mỗi scenario
                        scenario_steps_count = all_steps_df.groupby("scenario_name").size().reset_index(name="Số lượng steps")
                        scenario_steps_count = scenario_steps_count.sort_values(by="Số lượng steps", ascending=False)
                        
                        # Đếm theo loại step
                        step_types_count = all_steps_df["type"].value_counts().reset_index()
                        step_types_count.columns = ["Loại step", "Số lượng"]
                        
                        # Thống kê về FAQs
                        faqs_steps = all_steps_df[all_steps_df["type"] == "Answer"]
                        faqs_in_chunks = faqs_steps["has_match"].sum()
                        faqs_not_in_chunks = len(faqs_steps) - faqs_in_chunks
                        
                        # FAQs chưa có trong chunks
                        faqs_missing_chunks = faqs_steps[~faqs_steps["has_match"]].copy()
                        
                        # Tiếp tục hiển thị thống kê như phần "Phân tích dữ liệu toàn bộ"
                        # Hiển thị các biểu đồ và thông tin thống kê tương tự
                        
                        # 1. Thống kê số lượng kịch bản cho từng bộ phận
                        st.subheader("Số lượng kịch bản theo bộ phận")
                        
                        # Tạo DataFrame cho biểu đồ
                        dept_scenarios_df = pd.DataFrame({
                            "Bộ phận": department_scenarios.keys(),
                            "Số lượng kịch bản": department_scenarios.values()
                        }).sort_values(by="Số lượng kịch bản", ascending=False)
                        
                        # Hiển thị biểu đồ
                        # Hiển thị biểu đồ
                        dept_chart = alt.Chart(dept_scenarios_df).mark_bar().encode(
                            x=alt.X('Bộ phận:N', sort='-y'),
                            y='Số lượng kịch bản:Q',
                            color=alt.Color('Bộ phận:N', legend=None),
                            tooltip=['Bộ phận', 'Số lượng kịch bản']
                        ).properties(
                            width=600,
                            height=400,
                            title="Số lượng kịch bản theo bộ phận"
                        )
                        
                        st.altair_chart(dept_chart, use_container_width=True)
                        
                        # Hiển thị bảng dữ liệu
                        st.dataframe(dept_scenarios_df, use_container_width=True)
                        
                        # 2. Thống kê số lượng steps cho mỗi kịch bản
                        st.subheader("Số lượng steps cho mỗi kịch bản")
                        
                        # Hiển thị biểu đồ
                        if len(scenario_steps_count) > 20:
                            # Nếu có quá nhiều kịch bản, chỉ hiển thị top 20
                            top_scenarios = scenario_steps_count.head(20)
                            st.warning(f"Chỉ hiển thị top 20 kịch bản có nhiều steps nhất (tổng số: {len(scenario_steps_count)})")
                            scenarios_chart_data = top_scenarios
                        else:
                            scenarios_chart_data = scenario_steps_count
                        
                        # Tạo biểu đồ
                        scenarios_chart = alt.Chart(scenarios_chart_data).mark_bar().encode(
                            x=alt.X('Số lượng steps:Q'),
                            y=alt.Y('scenario_name:N', sort='-x', title="Tên kịch bản"),
                            tooltip=['scenario_name', 'Số lượng steps']
                        ).properties(
                            width=600,
                            height=min(400, len(scenarios_chart_data) * 20),
                            title="Số lượng steps cho mỗi kịch bản"
                        )
                        
                        st.altair_chart(scenarios_chart, use_container_width=True)
                        
                        # Hiển thị bảng dữ liệu đầy đủ
                        st.dataframe(scenario_steps_count, use_container_width=True)
                        
                        # 3. Thống kê số lượng steps theo loại
                        st.subheader("Thống kê số lượng steps theo loại")
                        
                        # Tạo biểu đồ pie chart
                        step_types_chart = alt.Chart(step_types_count).mark_arc().encode(
                            theta=alt.Theta(field="Số lượng", type="quantitative"),
                            color=alt.Color(field="Loại step", type="nominal"),
                            tooltip=['Loại step', 'Số lượng']
                        ).properties(
                            width=400,
                            height=400,
                            title="Phân bố loại steps"
                        )
                        
                        st.altair_chart(step_types_chart, use_container_width=True)
                        
                        # Hiển thị bảng dữ liệu
                        st.dataframe(step_types_count, use_container_width=True)
                        
                        # 4. Thống kê số lượng FAQs đã có chunks và chưa có chunks
                        st.subheader("Thống kê FAQs đã có và chưa có trong chunks")
                        
                        # Tạo DataFrame cho biểu đồ
                        faqs_chunks_df = pd.DataFrame({
                            "Trạng thái": ["Đã có trong chunks", "Chưa có trong chunks"],
                            "Số lượng": [faqs_in_chunks, faqs_not_in_chunks]
                        })
                        
                        # Tạo biểu đồ
                        faqs_chunks_chart = alt.Chart(faqs_chunks_df).mark_bar().encode(
                            x=alt.X('Trạng thái:N', title=""),
                            y=alt.Y('Số lượng:Q'),
                            color=alt.Color('Trạng thái:N', scale=alt.Scale(
                                domain=['Đã có trong chunks', 'Chưa có trong chunks'],
                                range=['#4CAF50', '#F44336']
                            )),
                            tooltip=['Trạng thái', 'Số lượng']
                        ).properties(
                            width=400,
                            height=300,
                            title="FAQs đã có và chưa có trong chunks"
                        )
                        
                        st.altair_chart(faqs_chunks_chart, use_container_width=True)
                        
                        # Hiển thị bảng dữ liệu
                        st.dataframe(faqs_chunks_df, use_container_width=True)
                        
                        # Hiển thị tỷ lệ phần trăm
                        col1, col2 = st.columns(2)
                        with col1:
                            total_faqs = len(faqs_steps)
                            if total_faqs > 0:
                                in_chunks_percent = (faqs_in_chunks / total_faqs) * 100
                                not_in_chunks_percent = (faqs_not_in_chunks / total_faqs) * 100
                                st.metric("Tỷ lệ đã có trong chunks", f"{in_chunks_percent:.2f}%")
                            
                        with col2:
                            if total_faqs > 0:
                                st.metric("Tỷ lệ chưa có trong chunks", f"{not_in_chunks_percent:.2f}%")
                        
                        # 5. Danh sách các FAQs chưa có trong chunks
                        st.subheader("Danh sách FAQs chưa có trong chunks")
                        
                        if not faqs_missing_chunks.empty:
                            # Sắp xếp theo bộ phận
                            faqs_missing_chunks = faqs_missing_chunks.sort_values(by=["department", "scenario_name"])
                            
                            # Tạo DataFrame hiển thị
                            faqs_missing_display = pd.DataFrame({
                                "Bộ phận": faqs_missing_chunks["department"],
                                "Kịch bản": faqs_missing_chunks["scenario_name"],
                                "Tên Step": faqs_missing_chunks["name"],
                                "Code": faqs_missing_chunks["code"]
                            })
                            
                            # Hiển thị danh sách
                            st.dataframe(faqs_missing_display, use_container_width=True)
                            
                            # Nút export
                            missing_csv = faqs_missing_chunks[[
                                "department", "scenario_name", "name", "code", "content"
                            ]].to_csv(index=False).encode('utf-8')
                            
                            filename = f"missing_faqs_{bot_selection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            
                            st.download_button(
                                label="Export FAQs chưa có chunks",
                                data=missing_csv,
                                file_name=filename,
                                mime="text/csv",
                            )
                        else:
                            st.info("Không có FAQs nào chưa có trong chunks.")
                        
                        # 6. Thống kê về điểm AI
                        st.subheader("Thống kê về đánh giá AI")
                        verified_steps = all_steps_df[all_steps_df["ai_verified"] == True]
                        
                        if not verified_steps.empty:
                            # Phân loại theo điểm số
                            high_quality = len(verified_steps[verified_steps["ai_score"] >= 80])
                            medium_quality = len(verified_steps[(verified_steps["ai_score"] >= 50) & (verified_steps["ai_score"] < 80)])
                            low_quality = len(verified_steps[verified_steps["ai_score"] < 50])
                            
                            # Tạo DataFrame cho biểu đồ
                            quality_df = pd.DataFrame({
                                "Chất lượng": ["Cao (80-100%)", "Trung bình (50-79%)", "Thấp (0-49%)"],
                                "Số lượng": [high_quality, medium_quality, low_quality]
                            })
                            
                            # Tạo biểu đồ
                            quality_chart = alt.Chart(quality_df).mark_bar().encode(
                                x=alt.X('Chất lượng:N', title=""),
                                y=alt.Y('Số lượng:Q'),
                                color=alt.Color('Chất lượng:N', scale=alt.Scale(
                                    domain=['Cao (80-100%)', 'Trung bình (50-79%)', 'Thấp (0-49%)'],
                                    range=['#4CAF50', '#FFC107', '#F44336']
                                )),
                                tooltip=['Chất lượng', 'Số lượng']
                            ).properties(
                                width=600,
                                height=400,
                                title="Phân bố chất lượng nội dung theo đánh giá AI"
                            )
                            
                            st.altair_chart(quality_chart, use_container_width=True)
                            
                            # Hiển thị thông tin tỷ lệ
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total_verified = len(verified_steps)
                                high_percent = (high_quality / total_verified) * 100 if total_verified > 0 else 0
                                st.metric("Chất lượng cao", f"{high_quality} steps ({high_percent:.1f}%)")
                            
                            with col2:
                                medium_percent = (medium_quality / total_verified) * 100 if total_verified > 0 else 0
                                st.metric("Chất lượng trung bình", f"{medium_quality} steps ({medium_percent:.1f}%)")
                            
                            with col3:
                                low_percent = (low_quality / total_verified) * 100 if total_verified > 0 else 0
                                st.metric("Chất lượng thấp", f"{low_quality} steps ({low_percent:.1f}%)")
                                
                            # 7. So sánh đánh giá Config vs AI
                            st.subheader("So sánh đánh giá Config vs AI")
                            
                            # Tạo dữ liệu cho scatter plot
                            compare_df = verified_steps[["name", "similarity_score", "ai_score"]].copy()
                            compare_df.columns = ["Step", "Config Score", "AI Score"]
                            
                            # Tạo scatter plot
                            scatter_chart = alt.Chart(compare_df).mark_circle(size=60, opacity=0.6).encode(
                                x=alt.X('Config Score:Q', scale=alt.Scale(domain=[0, 100]), title="Điểm tương đồng theo Config (%)"),
                                y=alt.Y('AI Score:Q', scale=alt.Scale(domain=[0, 100]), title="Điểm tương đồng theo AI (%)"),
                                tooltip=['Step', 'Config Score', 'AI Score']
                            ).properties(
                                width=600,
                                height=400,
                                title="So sánh điểm tương đồng giữa Config và AI"
                            )
                            
                            # Thêm đường tham chiếu
                            ref_line = alt.Chart(pd.DataFrame({'x': [0, 100], 'y': [0, 100]})).mark_line(color='red', strokeDash=[3, 3]).encode(
                                x='x',
                                y='y'
                            )
                            
                            st.altair_chart(scatter_chart + ref_line, use_container_width=True)
                            
                            # Thống kê về sự khác biệt giữa Config và AI
                            compare_df["Difference"] = abs(compare_df["Config Score"] - compare_df["AI Score"])
                            avg_diff = compare_df["Difference"].mean()
                            max_diff = compare_df["Difference"].max()
                            steps_with_large_diff = len(compare_df[compare_df["Difference"] > 20])
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Sai lệch trung bình", f"{avg_diff:.2f}%")
                            
                            with col2:
                                st.metric("Sai lệch lớn nhất", f"{max_diff:.2f}%")
                            
                            with col3:
                                st.metric("Steps có sai lệch >20%", f"{steps_with_large_diff}")
                            
                            # 8. Danh sách steps có sai lệch lớn giữa Config và AI
                            if steps_with_large_diff > 0:
                                st.subheader("Steps có sai lệch lớn giữa Config và AI (>20%)")
                                
                                large_diff_df = compare_df[compare_df["Difference"] > 20].sort_values(by="Difference", ascending=False)
                                
                                # Hiển thị danh sách
                                st.dataframe(large_diff_df, use_container_width=True)
                                
                                # Nút export
                                large_diff_export = large_diff_df.to_csv(index=False).encode('utf-8')
                                filename = f"large_diff_steps_{bot_selection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                
                                st.download_button(
                                    label="Export Steps có sai lệch lớn",
                                    data=large_diff_export,
                                    file_name=filename,
                                    mime="text/csv",
                                )
                        else:
                            st.warning("Chưa có steps nào được xác minh bằng AI.")
                
                elif st.button("Phân tích dữ liệu toàn bộ"):
                    # Hiển thị thanh tiến trình phân tích
                    progress_text = "Đang phân tích toàn bộ dữ liệu..."
                    progress_bar = st.progress(0, text=progress_text)
                    
                    # Thu thập dữ liệu thống kê từ tất cả scenarios và steps
                    all_steps_data = []
                    department_scenarios = {}
                    
                    # Duyệt qua từng scenario
                    for i, scenario in enumerate(filtered_scenarios):
                        # Cập nhật thanh tiến trình
                        progress = (i + 1) / len(filtered_scenarios)
                        progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{len(filtered_scenarios)})")
                        
                        # Trích xuất department từ tên scenario
                        scenario_name = scenario.get("name", "")
                        department_match = re.search(r'^\[(\d+)_([a-zA-Z]+)\]', scenario_name)
                        department = department_match.group(2) if department_match else "Unknown"
                        
                        # Đếm số lượng scenarios cho mỗi department
                        if department in department_scenarios:
                            department_scenarios[department] += 1
                        else:
                            department_scenarios[department] = 1
                        
                        # Lấy steps có activate=1
                        active_steps = [step for step in scenario.get("steps", []) if step.get("activate", 0) == 1]
                        
                        # Duyệt qua từng step
                        for step in active_steps:
                            # Tạo nội dung từ text và quick_reply cards
                            content = get_step_content(step)
                            
                            # Xác định kiểu step
                            step_type = determine_step_type(content)
                            
                            # Tìm kiếm trong chunks (nếu có)
                            matching_chunk = None
                            if chunks_data is not None and content.strip() != "":
                                matching_chunk = find_matching_chunk(
                                    content, chunks_data, bot_selection, 
                                    similarity_method=similarity_method,
                                    similarity_threshold=similarity_threshold
                                )
                            
                            all_steps_data.append({
                                "name": step.get("name", "Không có tên"),
                                "code": step["code"],
                                "position": step.get("position", 0),
                                "content": content,
                                "type": step_type,
                                "has_match": matching_chunk is not None,
                                "document_name": matching_chunk["Document name(*)"] if matching_chunk is not None else "",
                                "similarity_score": matching_chunk["similarity_score"] if matching_chunk is not None else 0,
                                "scenario_name": scenario_name,
                                "department": department,
                                "ai_verified": False,
                                "ai_score": 0
                            })
                        
                        # Thêm độ trễ nhỏ để cho thanh tiến trình hiển thị mượt hơn
                        time.sleep(0.01)
                    
                    # Hoàn tất phân tích
                    progress_bar.progress(100, text="Phân tích hoàn tất!")
                    st.success("Đã phân tích xong toàn bộ dữ liệu!")
                    
                    # Tạo DataFrame từ dữ liệu đã thu thập
                    all_steps_df = pd.DataFrame(all_steps_data)
                    
                    # 1. Thống kê số lượng kịch bản cho từng bộ phận
                    st.subheader("Số lượng kịch bản theo bộ phận")
                    
                    # Tạo DataFrame cho biểu đồ
                    dept_scenarios_df = pd.DataFrame({
                        "Bộ phận": department_scenarios.keys(),
                        "Số lượng kịch bản": department_scenarios.values()
                    }).sort_values(by="Số lượng kịch bản", ascending=False)
                    
                    # Hiển thị biểu đồ
                    dept_chart = alt.Chart(dept_scenarios_df).mark_bar().encode(
                        x=alt.X('Bộ phận:N', sort='-y'),
                        y='Số lượng kịch bản:Q',
                        color=alt.Color('Bộ phận:N', legend=None),
                        tooltip=['Bộ phận', 'Số lượng kịch bản']
                    ).properties(
                        width=600,
                        height=400,
                        title="Số lượng kịch bản theo bộ phận"
                    )
                    
                    st.altair_chart(dept_chart, use_container_width=True)
                    
                    # Hiển thị bảng dữ liệu
                    st.dataframe(dept_scenarios_df, use_container_width=True)
                    
                    # 2. Thống kê số lượng steps cho mỗi kịch bản
                    st.subheader("Số lượng steps cho mỗi kịch bản")
                    
                    # Đếm số lượng steps cho mỗi scenario
                    scenario_steps_count = all_steps_df.groupby("scenario_name").size().reset_index(name="Số lượng steps")
                    scenario_steps_count = scenario_steps_count.sort_values(by="Số lượng steps", ascending=False)
                    
                    # Hiển thị biểu đồ
                    if len(scenario_steps_count) > 20:
                        # Nếu có quá nhiều kịch bản, chỉ hiển thị top 20
                        top_scenarios = scenario_steps_count.head(20)
                        st.warning(f"Chỉ hiển thị top 20 kịch bản có nhiều steps nhất (tổng số: {len(scenario_steps_count)})")
                        scenarios_chart_data = top_scenarios
                    else:
                        scenarios_chart_data = scenario_steps_count
                    
                    # Tạo biểu đồ
                    scenarios_chart = alt.Chart(scenarios_chart_data).mark_bar().encode(
                        x=alt.X('Số lượng steps:Q'),
                        y=alt.Y('scenario_name:N', sort='-x', title="Tên kịch bản"),
                        tooltip=['scenario_name', 'Số lượng steps']
                    ).properties(
                        width=600,
                        height=min(400, len(scenarios_chart_data) * 20),
                        title="Số lượng steps cho mỗi kịch bản"
                    )
                    
                    st.altair_chart(scenarios_chart, use_container_width=True)
                    
                    # Hiển thị bảng dữ liệu đầy đủ
                    st.dataframe(scenario_steps_count, use_container_width=True)
                    
                    # 3. Thống kê số lượng steps theo loại
                    st.subheader("Thống kê số lượng steps theo loại")
                    
                    # Đếm số lượng steps cho mỗi loại
                    step_types_count = all_steps_df["type"].value_counts().reset_index()
                    step_types_count.columns = ["Loại step", "Số lượng"]
                    
                    # Tạo biểu đồ pie chart
                    step_types_chart = alt.Chart(step_types_count).mark_arc().encode(
                        theta=alt.Theta(field="Số lượng", type="quantitative"),
                        color=alt.Color(field="Loại step", type="nominal"),
                        tooltip=['Loại step', 'Số lượng']
                    ).properties(
                        width=400,
                        height=400,
                        title="Phân bố loại steps"
                    )
                    
                    st.altair_chart(step_types_chart, use_container_width=True)
                    
                    # Hiển thị bảng dữ liệu
                    st.dataframe(step_types_count, use_container_width=True)
                    
                    # 4. Thống kê số lượng FAQs đã có chunks và chưa có chunks
                    st.subheader("Thống kê FAQs đã có và chưa có trong chunks")
                    
                    # Lọc ra các steps dạng Answer (FAQs)
                    faqs_steps = all_steps_df[all_steps_df["type"] == "Answer"]
                    
                    # Đếm số lượng FAQs đã có và chưa có trong chunks
                    faqs_in_chunks = faqs_steps["has_match"].sum()
                    faqs_not_in_chunks = len(faqs_steps) - faqs_in_chunks
                    
                    # Tạo DataFrame cho biểu đồ
                    faqs_chunks_df = pd.DataFrame({
                        "Trạng thái": ["Đã có trong chunks", "Chưa có trong chunks"],
                        "Số lượng": [faqs_in_chunks, faqs_not_in_chunks]
                    })
                    
                    # Tạo biểu đồ
                    faqs_chunks_chart = alt.Chart(faqs_chunks_df).mark_bar().encode(
                        x=alt.X('Trạng thái:N', title=""),
                        y=alt.Y('Số lượng:Q'),
                        color=alt.Color('Trạng thái:N', scale=alt.Scale(
                            domain=['Đã có trong chunks', 'Chưa có trong chunks'],
                            range=['#4CAF50', '#F44336']
                        )),
                        tooltip=['Trạng thái', 'Số lượng']
                    ).properties(
                        width=400,
                        height=300,
                        title="FAQs đã có và chưa có trong chunks"
                    )
                    
                    st.altair_chart(faqs_chunks_chart, use_container_width=True)
                    
                    # Hiển thị bảng dữ liệu
                    st.dataframe(faqs_chunks_df, use_container_width=True)
                    
                    # Hiển thị tỷ lệ phần trăm
                    col1, col2 = st.columns(2)
                    with col1:
                        total_faqs = len(faqs_steps)
                        if total_faqs > 0:
                            in_chunks_percent = (faqs_in_chunks / total_faqs) * 100
                            not_in_chunks_percent = (faqs_not_in_chunks / total_faqs) * 100
                            st.metric("Tỷ lệ đã có trong chunks", f"{in_chunks_percent:.2f}%")
                        
                    with col2:
                        if total_faqs > 0:
                            st.metric("Tỷ lệ chưa có trong chunks", f"{not_in_chunks_percent:.2f}%")
                    
                    # 5. Danh sách các FAQs chưa có trong chunks
                    st.subheader("Danh sách FAQs chưa có trong chunks")
                    
                    # Lọc ra các FAQs chưa có trong chunks
                    faqs_missing_chunks = faqs_steps[~faqs_steps["has_match"]].copy()
                    
                    if not faqs_missing_chunks.empty:
                        # Sắp xếp theo bộ phận
                        faqs_missing_chunks = faqs_missing_chunks.sort_values(by=["department", "scenario_name"])
                        
                        # Tạo DataFrame hiển thị
                        faqs_missing_display = pd.DataFrame({
                            "Bộ phận": faqs_missing_chunks["department"],
                            "Kịch bản": faqs_missing_chunks["scenario_name"],
                            "Tên Step": faqs_missing_chunks["name"],
                            "Code": faqs_missing_chunks["code"]
                        })
                        
                        # Hiển thị danh sách
                        st.dataframe(faqs_missing_display, use_container_width=True)
                        
                        # Nút export với đánh giá config thêm vào
                        export_missing_df = pd.DataFrame({
                            "Department": faqs_missing_chunks["department"],
                            "Scenario": faqs_missing_chunks["scenario_name"],
                            "Step Name": faqs_missing_chunks["name"],
                            "Step Code": faqs_missing_chunks["code"],
                            "Content": faqs_missing_chunks["content"],
                            "Config Highest Similarity": [get_highest_similarity(content, chunks_data, bot_selection, similarity_method)[0] if chunks_data is not None else 0 for content in faqs_missing_chunks["content"]],
                            "Best Matching Document": [get_highest_similarity(content, chunks_data, bot_selection, similarity_method)[1] if chunks_data is not None else "" for content in faqs_missing_chunks["content"]]
                        })
                        
                        csv = export_missing_df.to_csv(index=False).encode('utf-8')
                        filename = f"missing_faqs_{bot_selection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="Export FAQs chưa có chunks",
                            data=csv,
                            file_name=filename,
                            mime="text/csv",
                        )
                    else:
                        st.info("Không có FAQs nào chưa có trong chunks.")
                    
                    # Lưu kết quả phân tích vào session state để tránh phân tích lại
                    st.session_state.analysis_results = {
                        "all_steps_df": all_steps_df,
                        "department_scenarios": department_scenarios,
                        "scenario_steps_count": scenario_steps_count,
                        "step_types_count": step_types_count,
                        "faqs_in_chunks": faqs_in_chunks,
                        "faqs_not_in_chunks": faqs_not_in_chunks,
                        "faqs_missing_chunks": faqs_missing_chunks
                    }
                
                # Hiển thị kết quả đã lưu nếu có
                elif "analysis_results" in st.session_state:
                    st.info("Hiển thị kết quả phân tích đã lưu. Nhấn 'Phân tích dữ liệu toàn bộ' để phân tích lại.")
                    
                    # Lấy kết quả đã lưu
                    results = st.session_state.analysis_results
                    
                    # 1. Thống kê số lượng kịch bản cho từng bộ phận
                    st.subheader("Số lượng kịch bản theo bộ phận")
                    
                    dept_scenarios_df = pd.DataFrame({
                        "Bộ phận": results["department_scenarios"].keys(),
                        "Số lượng kịch bản": results["department_scenarios"].values()
                    }).sort_values(by="Số lượng kịch bản", ascending=False)
                    
                    dept_chart = alt.Chart(dept_scenarios_df).mark_bar().encode(
                        x=alt.X('Bộ phận:N', sort='-y'),
                        y='Số lượng kịch bản:Q',
                        color=alt.Color('Bộ phận:N', legend=None),
                        tooltip=['Bộ phận', 'Số lượng kịch bản']
                    ).properties(
                        width=600,
                        height=400,
                        title="Số lượng kịch bản theo bộ phận"
                    )
                    
                    st.altair_chart(dept_chart, use_container_width=True)
                    st.dataframe(dept_scenarios_df, use_container_width=True)
                    
                    # 2. Số lượng steps cho mỗi kịch bản
                    st.subheader("Số lượng steps cho mỗi kịch bản")
                    
                    scenario_steps_count = results["scenario_steps_count"]
                    
                    if len(scenario_steps_count) > 20:
                        top_scenarios = scenario_steps_count.head(20)
                        st.warning(f"Chỉ hiển thị top 20 kịch bản có nhiều steps nhất (tổng số: {len(scenario_steps_count)})")
                        scenarios_chart_data = top_scenarios
                    else:
                        scenarios_chart_data = scenario_steps_count
                    
                    scenarios_chart = alt.Chart(scenarios_chart_data).mark_bar().encode(
                        x=alt.X('Số lượng steps:Q'),
                        y=alt.Y('scenario_name:N', sort='-x', title="Tên kịch bản"),
                        tooltip=['scenario_name', 'Số lượng steps']
                    ).properties(
                        width=600,
                        height=min(400, len(scenarios_chart_data) * 20),
                        title="Số lượng steps cho mỗi kịch bản"
                    )
                    
                    st.altair_chart(scenarios_chart, use_container_width=True)
                    st.dataframe(scenario_steps_count, use_container_width=True)
                    
                    # 3. Thống kê số lượng steps theo loại
                    st.subheader("Thống kê số lượng steps theo loại")
                    
                    step_types_count = results["step_types_count"]
                    
                    step_types_chart = alt.Chart(step_types_count).mark_arc().encode(
                        theta=alt.Theta(field="Số lượng", type="quantitative"),
                        color=alt.Color(field="Loại step", type="nominal"),
                        tooltip=['Loại step', 'Số lượng']
                    ).properties(
                        width=400,
                        height=400,
                        title="Phân bố loại steps"
                    )
                    
                    st.altair_chart(step_types_chart, use_container_width=True)
                    st.dataframe(step_types_count, use_container_width=True)
                    
                    # 4. Thống kê số lượng FAQs đã có chunks và chưa có chunks
                    st.subheader("Thống kê FAQs đã có và chưa có trong chunks")
                    
                    faqs_in_chunks = results["faqs_in_chunks"]
                    faqs_not_in_chunks = results["faqs_not_in_chunks"]
                    
                    faqs_chunks_df = pd.DataFrame({
                        "Trạng thái": ["Đã có trong chunks", "Chưa có trong chunks"],
                        "Số lượng": [faqs_in_chunks, faqs_not_in_chunks]
                    })
                    
                    faqs_chunks_chart = alt.Chart(faqs_chunks_df).mark_bar().encode(
                        x=alt.X('Trạng thái:N', title=""),
                        y=alt.Y('Số lượng:Q'),
                        color=alt.Color('Trạng thái:N', scale=alt.Scale(
                            domain=['Đã có trong chunks', 'Chưa có trong chunks'],
                            range=['#4CAF50', '#F44336']
                        )),
                        tooltip=['Trạng thái', 'Số lượng']
                    ).properties(
                        width=400,
                        height=300,
                        title="FAQs đã có và chưa có trong chunks"
                    )
                    
                    st.altair_chart(faqs_chunks_chart, use_container_width=True)
                    st.dataframe(faqs_chunks_df, use_container_width=True)
                    
                    # Hiển thị tỷ lệ phần trăm
                    col1, col2 = st.columns(2)
                    with col1:
                        total_faqs = faqs_in_chunks + faqs_not_in_chunks
                        if total_faqs > 0:
                            in_chunks_percent = (faqs_in_chunks / total_faqs) * 100
                            not_in_chunks_percent = (faqs_not_in_chunks / total_faqs) * 100
                            st.metric("Tỷ lệ đã có trong chunks", f"{in_chunks_percent:.2f}%")
                        
                    with col2:
                        if total_faqs > 0:
                            st.metric("Tỷ lệ chưa có trong chunks", f"{not_in_chunks_percent:.2f}%")
                    
                    # 5. Danh sách các FAQs chưa có trong chunks
                    st.subheader("Danh sách FAQs chưa có trong chunks")
                    
                    faqs_missing_chunks = results["faqs_missing_chunks"]
                    
                    if not faqs_missing_chunks.empty:
                        faqs_missing_display = pd.DataFrame({
                            "Bộ phận": faqs_missing_chunks["department"],
                            "Kịch bản": faqs_missing_chunks["scenario_name"],
                            "Tên Step": faqs_missing_chunks["name"],
                            "Code": faqs_missing_chunks["code"]
                        })
                        
                        st.dataframe(faqs_missing_display, use_container_width=True)
                        
                        # Nút export
                        export_missing_df = pd.DataFrame({
                            "Department": faqs_missing_chunks["department"],
                            "Scenario": faqs_missing_chunks["scenario_name"],
                            "Step Name": faqs_missing_chunks["name"],
                            "Step Code": faqs_missing_chunks["code"],
                            "Content": faqs_missing_chunks["content"],
                            "Config Highest Similarity": [get_highest_similarity(content, chunks_data, bot_selection, similarity_method)[0] if chunks_data is not None else 0 for content in faqs_missing_chunks["content"]],
                            "Best Matching Document": [get_highest_similarity(content, chunks_data, bot_selection, similarity_method)[1] if chunks_data is not None else "" for content in faqs_missing_chunks["content"]]
                        })
                        
                        csv = export_missing_df.to_csv(index=False).encode('utf-8')
                        filename = f"missing_faqs_{bot_selection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="Export FAQs chưa có chunks",
                            data=csv,
                            file_name=filename,
                            mime="text/csv",
                        )
                    else:
                        st.info("Không có FAQs nào chưa có trong chunks.")
                
                else:
                    st.info("Nhấn nút 'Phân tích dữ liệu toàn bộ' để xem thống kê tổng thể.")
        
        except json.JSONDecodeError as e:
            st.error(f"Lỗi khi đọc file JSON: {str(e)}")
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {str(e)}")
            st.exception(e)

def get_latest_chunks_file():
    """Lấy file Excel chunks data mới nhất trong thư mục data/chunks/"""
    chunks_folder = "data/chunks"
    
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(chunks_folder):
        return None
    
    # Tìm tất cả các file Excel
    excel_files = glob.glob(os.path.join(chunks_folder, "chunks_list_*.xlsx"))
    
    if not excel_files:
        return None
    
    # Sắp xếp theo thời gian tạo file (mới nhất đầu tiên)
    excel_files.sort(key=os.path.getmtime, reverse=True)
    
    return excel_files[0]

def get_step_content(step):
    """Trích xuất nội dung từ cards của một step"""
    content = ""
    
    for card in step.get("cards", []):
        if card.get("activate", 0) != 1:
            continue
            
        try:
            config = json.loads(card.get("config", "{}"))
            
            if card.get("card_type_id") == "text" and "text" in config:
                content += config["text"] + "\n\n"
            
            elif card.get("card_type_id") == "quick_reply" and "text" in config:
                content += config["text"] + "\n\n"
                
        except json.JSONDecodeError:
            pass
    
    return content.strip()

def determine_step_type(content):
    """Xác định loại step dựa vào nội dung"""
    if not content.strip():
        return "Other"
    
    # Kiểm tra xem có dạng menu hay không (chứa 【number】)
    if re.search(r'【\d+】', content):
        return "Menu"
    else:
        return "Answer"

def preprocess_text(text, is_vietnamese=False):
    """
    Tiền xử lý văn bản cho việc so sánh
    
    Args:
        text (str): Văn bản cần xử lý
        is_vietnamese (bool): Có phải văn bản tiếng Việt không
        
    Returns:
        str: Văn bản đã được xử lý
    """
    # Chuyển về chuỗi nếu không phải
    text = str(text)
    
    # Chuyển về chữ thường
    text = text.lower()
    
    # Loại bỏ URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Loại bỏ emoji
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Loại bỏ markdown
    text = re.sub(r'[*_~`|>#\[\]\(\)]+', '', text)
    
    # Loại bỏ các ký tự đặc biệt và giữ lại chữ cái, số, khoảng trắng
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Loại bỏ dấu tiếng Việt nếu cần
    if is_vietnamese:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def sequence_similarity(str1, str2):
    """
    Tính độ tương đồng giữa hai chuỗi bằng thuật toán SequenceMatcher
    
    Args:
        str1 (str): Chuỗi thứ nhất
        str2 (str): Chuỗi thứ hai
        
    Returns:
        float: Độ tương đồng (0-1)
    """
    return SequenceMatcher(None, str1, str2).ratio()

def vector_similarity(str1, str2):
    """
    Tính độ tương đồng giữa hai chuỗi bằng cách biểu diễn vector và tính cosine similarity
    
    Args:
        str1 (str): Chuỗi thứ nhất
        str2 (str): Chuỗi thứ hai
        
    Returns:
        float: Độ tương đồng (0-1)
    """
    if not VECTOR_AVAILABLE:
        return 0
        
    try:
        # Tạo TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        
        # Tạo ma trận TF-IDF
        tfidf_matrix = vectorizer.fit_transform([str1, str2])
        
        # Tính cosine similarity
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Đảm bảo giá trị nằm trong khoảng 0-1
        return max(0, min(sim, 1))
    except Exception as e:
        print(f"Lỗi khi tính vector similarity: {str(e)}")
        # Nếu có lỗi, trả về 0
        return 0

def find_common_substrings(str1, str2, min_length=3):
    """
    Tìm tất cả các chuỗi con chung giữa hai chuỗi, với độ dài tối thiểu
    
    Args:
        str1 (str): Chuỗi thứ nhất
        str2 (str): Chuỗi thứ hai
        min_length (int): Độ dài tối thiểu của chuỗi con
        
    Returns:
        list: Danh sách các chuỗi con chung (sắp xếp theo độ dài giảm dần)
    """
    substrings = []
    
    # Chuyển về lowercase để so sánh không phân biệt hoa thường
    str1 = str1.lower()
    str2 = str2.lower()
    
    # Tách thành từng từ
    words1 = str1.split()
    words2 = str2.split()
    
    # Tìm các chuỗi từ liên tiếp chung
    for i in range(len(words1)):
        for j in range(len(words2)):
            k = 0
            while (i + k < len(words1) and 
                  j + k < len(words2) and 
                  words1[i + k] == words2[j + k]):
                k += 1
                
                # Nếu tìm được chuỗi từ đủ độ dài
                if k >= min_length:
                    common_phrase = " ".join(words1[i:i+k])
                    if common_phrase not in substrings:
                        substrings.append(common_phrase)
    
    # Sắp xếp theo độ dài (dài nhất trước)
    substrings.sort(key=lambda x: len(x), reverse=True)
    
    # Loại bỏ các chuỗi con trùng lặp (nếu chuỗi này là con của chuỗi kia)
    non_redundant_substrings = []
    for i, phrase in enumerate(substrings):
        is_subset = False
        for j, other_phrase in enumerate(substrings):
            if i != j and phrase in other_phrase:
                is_subset = True
                break
        if not is_subset:
            non_redundant_substrings.append(phrase)
    
    return non_redundant_substrings

def substring_similarity(str1, str2, min_length=3):
    """
    Tính độ tương đồng dựa trên các chuỗi con chung
    
    Args:
        str1 (str): Chuỗi thứ nhất
        str2 (str): Chuỗi thứ hai
        min_length (int): Độ dài tối thiểu của chuỗi con (tính bằng số từ)
        
    Returns:
        float: Điểm tương đồng từ 0-1
        dict: Thông tin chi tiết về các chuỗi con
    """
    if not str1 or not str2:
        return 0, {}
    
    # Tìm các chuỗi con chung
    common_substrings = find_common_substrings(str1, str2, min_length)
    
    if not common_substrings:
        return 0, {}
    
    # Tính tổng độ dài của các chuỗi con (số từ)
    total_words_in_substrings = sum(len(s.split()) for s in common_substrings)
    
    # Tính tổng số từ trong cả hai chuỗi
    total_words = len(str1.split()) + len(str2.split())
    
    # Công thức tính điểm:
    # - Tỷ lệ số từ trong chuỗi con so với tổng số từ
    # - Thưởng thêm cho chuỗi con dài nhất
    
    # Chuỗi con dài nhất (nếu có)
    longest_substring = common_substrings[0] if common_substrings else ""
    longest_length = len(longest_substring.split())
    
    # Tính điểm dựa trên tỷ lệ số từ được bao phủ
    coverage_score = (2 * total_words_in_substrings) / total_words if total_words > 0 else 0
    
    # Thưởng thêm cho chuỗi con dài nhất
    longest_bonus = longest_length / (min(len(str1.split()), len(str2.split())))
    
    # Điểm cuối cùng (kết hợp cả hai yếu tố)
    score = 0.7 * coverage_score + 0.3 * longest_bonus
    
    # Giới hạn điểm trong khoảng 0-1
    score = min(1.0, score)
    
    # Thông tin chi tiết
    details = {
        "common_substrings": common_substrings,
        "total_common_words": total_words_in_substrings,
        "longest_substring": longest_substring,
        "longest_length": longest_length,
        "coverage_score": coverage_score,
        "longest_bonus": longest_bonus,
        "final_score": score
    }
    
    return score, details

def common_phrases_similarity(str1, str2):
    """
    Tính độ tương đồng dựa trên cụm từ chung với trọng số
    
    Args:
        str1 (str): Chuỗi thứ nhất
        str2 (str): Chuỗi thứ hai
        
    Returns:
        float: Điểm tương đồng từ 0-1
    """
    # Tìm các chuỗi con của các độ dài khác nhau
    score_3, details_3 = substring_similarity(str1, str2, min_length=3)  # Cụm từ ít nhất 3 từ
    score_2, details_2 = substring_similarity(str1, str2, min_length=2)  # Cụm từ ít nhất 2 từ
    
    # Kết hợp các điểm số với trọng số khác nhau
    # - Trọng số cao hơn cho cụm từ dài (3+ từ)
    combined_score = 0.7 * score_3 + 0.3 * score_2
    
    return combined_score, {
        "details_3words": details_3,
        "details_2words": details_2,
        "combined_score": combined_score
    }

def similarity_ratio(str1, str2, bot_type=None, method="hybrid"):
    """
    Tính tỷ lệ tương đồng giữa hai chuỗi với các phương pháp khác nhau
    
    Args:
        str1 (str): Chuỗi thứ nhất cần so sánh
        str2 (str): Chuỗi thứ hai cần so sánh
        bot_type (str): Loại bot (VN hoặc JP) để xử lý đúng ngôn ngữ
        method (str): Phương pháp so sánh ("sequence", "vector", "hybrid", "substring")
        
    Returns:
        float: Tỷ lệ tương đồng từ 0 đến 1
    """
    # Kiểm tra nếu một trong hai chuỗi trống
    if not str1 or not str2:
        return 0
        
    # Chuyển thành chuỗi nếu không phải
    str1 = str(str1)
    str2 = str(str2)
    
    # Kiểm tra chênh lệch độ dài quá lớn
    len_ratio = min(len(str1), len(str2)) / max(len(str1), len(str2)) if max(len(str1), len(str2)) > 0 else 0
    if len_ratio < 0.3:  # Nếu độ dài chênh lệch quá 70%
        return 0
    
    # Tiền xử lý cả hai chuỗi
    is_vietnamese = bot_type and "VN" in bot_type
    processed_str1 = preprocess_text(str1, is_vietnamese)
    processed_str2 = preprocess_text(str2, is_vietnamese)
    
    # Nếu sau khi xử lý, một trong hai chuỗi trống
    if not processed_str1 or not processed_str2:
        return 0
    
    # Kiểm tra lại chênh lệch độ dài sau khi xử lý
    processed_len_ratio = min(len(processed_str1), len(processed_str2)) / max(len(processed_str1), len(processed_str2)) if max(len(processed_str1), len(processed_str2)) > 0 else 0
    if processed_len_ratio < 0.3:
        return 0
    
    # Tính độ tương đồng dựa trên phương pháp được chọn
    if method == "sequence":
        # Phương pháp so sánh chuỗi truyền thống
        return sequence_similarity(processed_str1, processed_str2)
    elif method == "vector" and VECTOR_AVAILABLE:
        # Phương pháp so sánh vector (nếu có thư viện)
        return vector_similarity(processed_str1, processed_str2)
    elif method == "substring":
        # Phương pháp so sánh chuỗi con
        score, _ = common_phrases_similarity(processed_str1, processed_str2)
        return score
    elif method == "hybrid" and VECTOR_AVAILABLE:
        # Kết hợp cả ba phương pháp
        seq_sim = sequence_similarity(processed_str1, processed_str2)
        vec_sim = vector_similarity(processed_str1, processed_str2)
        sub_sim, _ = common_phrases_similarity(processed_str1, processed_str2)
        return (0.3 * seq_sim + 0.3 * vec_sim + 0.4 * sub_sim)
    elif method == "hybrid":
        # Kết hợp hai phương pháp nếu không có vector
        seq_sim = sequence_similarity(processed_str1, processed_str2)
        sub_sim, _ = common_phrases_similarity(processed_str1, processed_str2)
        return (0.4 * seq_sim + 0.6 * sub_sim)
    else:
        # Mặc định sử dụng so sánh chuỗi
        return sequence_similarity(processed_str1, processed_str2)

def find_matching_chunk(content, chunks_data, bot_type=None, similarity_method="hybrid", similarity_threshold=70):
    """
    Tìm chunk phù hợp nhất với nội dung step
    
    Args:
        content (str): Nội dung cần tìm kiếm
        chunks_data (DataFrame): DataFrame chứa dữ liệu chunks
        bot_type (str): Loại bot (VN hoặc JP) để xử lý đúng ngôn ngữ
        similarity_method (str): Phương pháp tính độ tương đồng
        similarity_threshold (float): Ngưỡng tỷ lệ tương đồng (0-100)
        
    Returns:
        Series: Dòng dữ liệu chunk phù hợp nhất, hoặc None nếu không tìm thấy
    """
    if content.strip() == "" or "Content(*)" not in chunks_data.columns:
        return None
    
    # Thêm cột điểm tương đồng
    similarity_scores = chunks_data["Content(*)"].apply(
        lambda x: similarity_ratio(content, str(x), bot_type, similarity_method) * 100 if pd.notna(x) else 0
    )
    
    # Lọc những chunk có độ tương đồng vượt ngưỡng
    max_score_idx = similarity_scores.idxmax() if not similarity_scores.empty else None
    max_score = similarity_scores.max() if not similarity_scores.empty else 0
    
    if max_score_idx is not None and max_score >= similarity_threshold:
        chunk = chunks_data.iloc[max_score_idx].copy()
        chunk["similarity_score"] = max_score
        
        # Thêm chi tiết về chuỗi con nếu sử dụng phương pháp substring hoặc hybrid
        if similarity_method in ["substring", "hybrid"]:
            is_vietnamese = bot_type and "VN" in bot_type
            processed_content = preprocess_text(content, is_vietnamese)
            processed_chunk = preprocess_text(str(chunk["Content(*)"]), is_vietnamese)
            
            _, substring_details = common_phrases_similarity(processed_content, processed_chunk)
            chunk["similarity_details"] = {
                "substring_details": substring_details
            }
        
        return chunk
    
    return None

def get_highest_similarity(content, chunks_data, bot_type=None, similarity_method="hybrid"):
    """
    Lấy điểm tương đồng cao nhất và document tương ứng
    
    Args:
        content (str): Nội dung cần tìm kiếm
        chunks_data (DataFrame): DataFrame chứa dữ liệu chunks
        bot_type (str): Loại bot (VN hoặc JP) để xử lý đúng ngôn ngữ
        similarity_method (str): Phương pháp tính độ tương đồng
        
    Returns:
        tuple: (highest_score, document_name, details)
    """
    if content.strip() == "" or "Content(*)" not in chunks_data.columns or chunks_data.empty:
        return 0, "N/A", {}
    
    # Tiền xử lý nội dung
    is_vietnamese = bot_type and "VN" in bot_type
    processed_content = preprocess_text(content, is_vietnamese)
    
    # Khởi tạo biến lưu kết quả
    max_score = 0
    max_idx = -1
    details = {}
    
    # Tính điểm tương đồng cho từng chunk
    for idx, chunk_content in enumerate(chunks_data["Content(*)"]):
        if pd.isna(chunk_content):
            continue
            
        # Tiền xử lý chunk
        processed_chunk = preprocess_text(str(chunk_content), is_vietnamese)
        
        # Tính điểm tương đồng
        score = 0
        
        if similarity_method == "sequence":
            score = sequence_similarity(processed_content, processed_chunk)
        elif similarity_method == "vector" and VECTOR_AVAILABLE:
            score = vector_similarity(processed_content, processed_chunk)
        elif similarity_method == "substring":
            score, substring_details = common_phrases_similarity(processed_content, processed_chunk)
            if score > max_score:
                details = {"substring_details": substring_details}
        elif similarity_method == "hybrid":
            seq_sim = sequence_similarity(processed_content, processed_chunk)
            sub_sim, substring_details = common_phrases_similarity(processed_content, processed_chunk)
            vec_sim = 0
            if VECTOR_AVAILABLE:
                vec_sim = vector_similarity(processed_content, processed_chunk)
                score = (0.3 * seq_sim + 0.3 * vec_sim + 0.4 * sub_sim)
            else:
                score = (0.4 * seq_sim + 0.6 * sub_sim)
                
            if score > max_score:
                details = {"substring_details": substring_details}
        else:
            score = sequence_similarity(processed_content, processed_chunk)
        
        # Cập nhật điểm cao nhất
        if score > max_score:
            max_score = score
            max_idx = idx
    
    # Nếu tìm thấy điểm cao nhất
    if max_idx >= 0:
        document_name = chunks_data.iloc[max_idx]["Document name(*)"]
        return max_score * 100, document_name, details
    
    return 0, "N/A", {}

def render_advanced_comparison(st, selected_step_data, matching_chunk, bot_selection):
    """
    Hiển thị so sánh nâng cao giữa step và chunk
    
    Args:
        st: Đối tượng Streamlit
        selected_step_data: Dữ liệu của step được chọn
        matching_chunk: Dữ liệu của chunk khớp
        bot_selection: Loại bot được chọn
    """
    st.subheader("So sánh chi tiết")
    
    # Lấy nội dung
    step_content = selected_step_data["content"]
    chunk_content = matching_chunk["Content(*)"]
    
    # Xử lý nội dung để so sánh
    is_vietnamese = "VN" in bot_selection
    processed_step = preprocess_text(step_content, is_vietnamese)
    processed_chunk = preprocess_text(str(chunk_content), is_vietnamese)
    
    # Hiển thị nội dung đã xử lý
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Nội dung Step (đã xử lý)**")
        st.text_area("Nội dung step", processed_step, height=150, label_visibility="hidden")
    with col2:
        st.write("**Nội dung Chunk (đã xử lý)**")
        st.text_area("Nội dung chunk", processed_chunk, height=150, label_visibility="hidden")
    
    # Phân tích chuỗi con chung
    score, details = common_phrases_similarity(processed_step, processed_chunk)
    
    # Hiển thị kết quả phân tích
    st.write(f"**Điểm tương đồng dựa trên chuỗi con: {score*100:.2f}%**")
    
    # Hiển thị các chuỗi con dài (3+ từ) - chỉ hiển thị các chuỗi không trùng lặp
    common_substrings = details["details_3words"]["common_substrings"]
    if common_substrings:
        st.write("**Các cụm từ chung không trùng lặp (3+ từ):**")
        for i, phrase in enumerate(common_substrings[:5]):  # Hiển thị tối đa 5 cụm từ
            st.info(f"{i+1}. {phrase}")