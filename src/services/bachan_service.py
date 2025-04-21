import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import time

class BAchanService:
    """
    Service để tương tác với BAchan Workplace API
    """
    
    def __init__(self, token):
        """
        Khởi tạo service với token API
        
        Args:
            token (str): Token API từ BAchan Workplace
        """
        self.token = token
        self.base_url = "https://bachan.fpt.ai/api"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def check_token(self):
        """
        Kiểm tra token có hợp lệ không
        
        Returns:
            bool: True nếu token hợp lệ, False nếu không
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ BAchan
        # url = f"{self.base_url}/auth/validate"
        # 
        # try:
        #     response = self.session.get(url)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     return data.get("valid", False)
        # except Exception as e:
        #     print(f"Error validating token: {str(e)}")
        #     return False
        
        # Giả lập token hợp lệ
        return True
    
    def send_message(self, message):
        """
        Gửi tin nhắn đến BAchan và nhận phản hồi
        
        Args:
            message (str): Tin nhắn cần gửi
            
        Returns:
            str: Phản hồi từ BAchan
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ BAchan
        # url = f"{self.base_url}/chat"
        # 
        # payload = {
        #     "message": message
        # }
        # 
        # try:
        #     response = self.session.post(url, json=payload)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     return data.get("response", "")
        # except Exception as e:
        #     print(f"Error sending message: {str(e)}")
        #     return f"Error: {str(e)}"
        
        # Giả lập phản hồi từ BAchan
        return self._generate_sample_response(message)
    
    def create_meeting(self, title, date, time, participants):
        """
        Tạo lịch họp mới trong BAchan Workplace
        
        Args:
            title (str): Tiêu đề cuộc họp
            date (date): Ngày họp
            time (time): Giờ họp
            participants (list): Danh sách người tham gia
            
        Returns:
            dict: Thông tin về cuộc họp đã tạo
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ BAchan
        # url = f"{self.base_url}/meeting"
        # 
        # # Chuyển đổi danh sách người tham gia
        # if isinstance(participants, str):
        #     participants = [p.strip() for p in participants.split("\n") if p.strip()]
        # 
        # payload = {
        #     "title": title,
        #     "date": date.strftime("%Y-%m-%d"),
        #     "time": time.strftime("%H:%M:%S"),
        #     "participants": participants
        # }
        # 
        # try:
        #     response = self.session.post(url, json=payload)
        #     response.raise_for_status()
        #     
        #     return response.json()
        # except Exception as e:
        #     print(f"Error creating meeting: {str(e)}")
        #     return {"error": str(e)}
        
        # Giả lập tạo lịch họp thành công
        if isinstance(participants, str):
            participants = [p.strip() for p in participants.split("\n") if p.strip()]
            
        meeting_id = f"M{int(time.time())}"
        
        return {
            "id": meeting_id,
            "title": title,
            "date": date.strftime("%Y-%m-%d"),
            "time": time.strftime("%H:%M:%S"),
            "participants": participants,
            "status": "scheduled",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def create_report(self, title, report_type, department, include_charts=False):
        """
        Tạo báo cáo mới trong BAchan Workplace
        
        Args:
            title (str): Tiêu đề báo cáo
            report_type (str): Loại báo cáo (daily, weekly, monthly)
            department (str): Phòng ban
            include_charts (bool): Có bao gồm biểu đồ hay không
            
        Returns:
            dict: Thông tin về báo cáo đã tạo
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ BAchan
        # url = f"{self.base_url}/report"
        # 
        # payload = {
        #     "title": title,
        #     "type": report_type,
        #     "department": department,
        #     "include_charts": include_charts
        # }
        # 
        # try:
        #     response = self.session.post(url, json=payload)
        #     response.raise_for_status()
        #     
        #     return response.json()
        # except Exception as e:
        #     print(f"Error creating report: {str(e)}")
        #     return {"error": str(e)}
        
        # Giả lập tạo báo cáo thành công
        report_id = f"R{int(time.time())}"
        
        return {
            "id": report_id,
            "title": title,
            "type": report_type,
            "department": department,
            "include_charts": include_charts,
            "status": "created",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def analyze_data(self, data_source, analysis_type, params=None):
        """
        Yêu cầu BAchan phân tích dữ liệu
        
        Args:
            data_source (str): Nguồn dữ liệu (fptai, maya, file)
            analysis_type (str): Loại phân tích
            params (dict, optional): Các tham số bổ sung
            
        Returns:
            dict: Kết quả phân tích
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ BAchan
        # url = f"{self.base_url}/analyze"
        # 
        # payload = {
        #     "data_source": data_source,
        #     "analysis_type": analysis_type
        # }
        # 
        # if params:
        #     payload.update(params)
        # 
        # try:
        #     response = self.session.post(url, json=payload)
        #     response.raise_for_status()
        #     
        #     return response.json()
        # except Exception as e:
        #     print(f"Error analyzing data: {str(e)}")
        #     return {"error": str(e)}
        
        # Giả lập phân tích dữ liệu thành công
        analysis_id = f"A{int(time.time())}"
        
        return {
            "id": analysis_id,
            "data_source": data_source,
            "analysis_type": analysis_type,
            "status": "completed",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": {
                "summary": "Phân tích dữ liệu hoàn tất",
                "metrics": {
                    "total_records": 1253,
                    "growth_rate": "15%",
                    "anomalies": 3
                },
                "insights": [
                    "Tăng trưởng ổn định 15% so với tháng trước",
                    "Phát hiện 3 điểm dữ liệu ngoại lệ cần kiểm tra",
                    "Xu hướng tăng mạnh trong nhóm khách hàng doanh nghiệp"
                ],
                "recommendations": [
                    "Tập trung vào phân khúc khách hàng doanh nghiệp",
                    "Kiểm tra lại các điểm dữ liệu ngoại lệ",
                    "Cần thu thập thêm dữ liệu về chi tiết A để phân tích sâu hơn"
                ]
            }
        }
    
    def _generate_sample_response(self, message):
        """
        Tạo phản hồi mẫu dựa trên tin nhắn đầu vào
        
        Args:
            message (str): Tin nhắn từ người dùng
            
        Returns:
            str: Phản hồi mẫu
        """
        # Chuyển đổi tin nhắn sang chữ thường để dễ dàng so sánh
        message_lower = message.lower()
        
        # Các mẫu phản hồi dựa trên từ khóa
        if "xin chào" in message_lower or "hello" in message_lower:
            return "Xin chào! Tôi là BAchan, trợ lý ảo của FPT AI. Tôi có thể giúp gì cho bạn hôm nay?"
        
        elif "giúp" in message_lower or "help" in message_lower:
            return """Tôi có thể giúp bạn với nhiều tác vụ khác nhau như:
- Thu thập và phân tích dữ liệu
- Lên lịch họp và quản lý nhiệm vụ
- Tạo báo cáo tự động
- Trả lời các câu hỏi về FPT AI

Bạn cần hỗ trợ về vấn đề cụ thể nào?"""
        
        elif "dữ liệu" in message_lower or "data" in message_lower:
            return "Tôi có thể giúp bạn truy cập và phân tích dữ liệu từ nhiều nguồn khác nhau bao gồm FPT.AI, Maya Portal hoặc từ các file bạn tải lên. Bạn muốn làm việc với dữ liệu nào?"
        
        elif "báo cáo" in message_lower or "report" in message_lower:
            return "Tôi có thể giúp bạn tạo các loại báo cáo khác nhau như báo cáo hiệu suất, báo cáo dự án, báo cáo tài chính... Bạn cần loại báo cáo nào?"
        
        elif "lịch họp" in message_lower or "meeting" in message_lower:
            return "Tôi có thể giúp bạn lên lịch họp mới. Vui lòng cung cấp thông tin về tiêu đề, thời gian và người tham gia."
        
        elif "fpt ai" in message_lower:
            return "FPT AI là bộ phận nghiên cứu và phát triển trí tuệ nhân tạo của Tập đoàn FPT. Chúng tôi cung cấp nhiều sản phẩm và dịch vụ AI như xử lý ngôn ngữ tự nhiên, computer vision, chatbot và nhiều giải pháp AI tùy chỉnh khác."
        
        # Phản hồi mặc định
        return "Cảm ơn bạn đã liên hệ. Tôi đang xử lý yêu cầu của bạn. Tôi có thể giúp bạn với các tác vụ liên quan đến dữ liệu FPT AI, quản lý lịch họp, tạo báo cáo và nhiều việc khác. Vui lòng cho tôi biết chi tiết hơn về nhu cầu của bạn."