import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta

class FPTAIService:
    """
    Service để tương tác với FPT.AI API
    """
    
    def __init__(self, api_key):
        """
        Khởi tạo service với API key
        
        Args:
            api_key (str): API key từ FPT.AI
        """
        self.api_key = api_key
        self.base_url = "https://api.fpt.ai"
        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    def get_usage_data(self, service_type, start_date, end_date):
        """
        Lấy dữ liệu sử dụng cho một dịch vụ cụ thể
        
        Args:
            service_type (str): Loại dịch vụ (tts, stt, ocr, etc.)
            start_date (datetime): Ngày bắt đầu
            end_date (datetime): Ngày kết thúc
            
        Returns:
            DataFrame: Dữ liệu sử dụng dạng pandas DataFrame
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ FPT.AI
        # Đây là mô phỏng
        
        # Chuyển đổi ngày sang định dạng phù hợp
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # URL thực tế sẽ phụ thuộc vào API của FPT.AI
        # url = f"{self.base_url}/usage/{service_type}?start={start_str}&end={end_str}"
        
        # try:
        #     response = requests.get(url, headers=self.headers)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     # Xử lý dữ liệu và chuyển đổi sang DataFrame
        #     return pd.DataFrame(data["usage"])
        # except Exception as e:
        #     print(f"Error fetching usage data: {str(e)}")
        #     return pd.DataFrame()
        
        # Tạo dữ liệu mẫu thay thế
        return self._generate_sample_data(service_type, start_date, end_date)
    
    def text_to_speech(self, text, voice="female", format="mp3"):
        """
        Chuyển đổi văn bản thành giọng nói sử dụng FPT.AI TTS
        
        Args:
            text (str): Văn bản cần chuyển đổi
            voice (str): Giọng nói (female, male, etc.)
            format (str): Định dạng âm thanh (mp3, wav)
            
        Returns:
            str: URL để tải file âm thanh
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ FPT.AI
        # url = f"{self.base_url}/tts/v1"
        # 
        # payload = {
        #     "text": text,
        #     "voice": voice,
        #     "format": format
        # }
        # 
        # try:
        #     response = requests.post(url, headers=self.headers, json=payload)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data.get("error") == 0:
        #         return data.get("async")  # URL để tải file âm thanh
        #     else:
        #         print(f"TTS API error: {data.get('message')}")
        #         return None
        # except Exception as e:
        #     print(f"Error calling TTS API: {str(e)}")
        #     return None
        
        # Trả về URL mẫu
        return f"https://api.fpt.ai/tts/v1/download/{datetime.now().strftime('%Y%m%d%H%M%S')}.{format}"
    
    def speech_to_text(self, audio_file, language="vi"):
        """
        Chuyển đổi giọng nói thành văn bản sử dụng FPT.AI STT
        
        Args:
            audio_file (str): Đường dẫn đến file âm thanh
            language (str): Ngôn ngữ (vi, en, etc.)
            
        Returns:
            str: Kết quả chuyển đổi
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ FPT.AI
        # url = f"{self.base_url}/stt/v1"
        # 
        # # Đọc file âm thanh
        # with open(audio_file, "rb") as f:
        #     files = {"file": f}
        #     headers = {"api-key": self.api_key}
        #     
        #     try:
        #         response = requests.post(url, headers=headers, files=files)
        #         response.raise_for_status()
        #         
        #         data = response.json()
        #         if "hypotheses" in data and len(data["hypotheses"]) > 0:
        #             return data["hypotheses"][0]["utterance"]
        #         else:
        #             print("No transcription result found")
        #             return None
        #     except Exception as e:
        #         print(f"Error calling STT API: {str(e)}")
        #         return None
        
        # Trả về kết quả mẫu
        return "Đây là văn bản mẫu được chuyển đổi từ âm thanh."
    
    def ocr(self, image_file, language="vi"):
        """
        Nhận dạng ký tự trong ảnh sử dụng FPT.AI OCR
        
        Args:
            image_file (str): Đường dẫn đến file ảnh
            language (str): Ngôn ngữ (vi, en, etc.)
            
        Returns:
            dict: Kết quả nhận dạng
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ FPT.AI
        # url = f"{self.base_url}/ocr/v1"
        # 
        # # Đọc file ảnh
        # with open(image_file, "rb") as f:
        #     files = {"image": f}
        #     headers = {"api-key": self.api_key}
        #     
        #     try:
        #         response = requests.post(url, headers=headers, files=files)
        #         response.raise_for_status()
        #         
        #         return response.json()
        #     except Exception as e:
        #         print(f"Error calling OCR API: {str(e)}")
        #         return None
        
        # Trả về kết quả mẫu
        return {
            "errorCode": 0,
            "data": [
                {
                    "text": "Đây là văn bản mẫu trong ảnh",
                    "confidence": 0.95,
                    "box": [10, 10, 100, 50]
                }
            ]
        }
    
    def face_recognition(self, image_file):
        """
        Nhận dạng khuôn mặt trong ảnh sử dụng FPT.AI Face Recognition
        
        Args:
            image_file (str): Đường dẫn đến file ảnh
            
        Returns:
            dict: Kết quả nhận dạng
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ FPT.AI
        # url = f"{self.base_url}/vision/v1/face"
        # 
        # # Đọc file ảnh
        # with open(image_file, "rb") as f:
        #     files = {"image": f}
        #     headers = {"api-key": self.api_key}
        #     
        #     try:
        #         response = requests.post(url, headers=headers, files=files)
        #         response.raise_for_status()
        #         
        #         return response.json()
        #     except Exception as e:
        #         print(f"Error calling Face Recognition API: {str(e)}")
        #         return None
        
        # Trả về kết quả mẫu
        return {
            "errorCode": 0,
            "data": [
                {
                    "face_id": "face123",
                    "confidence": 0.95,
                    "box": [10, 10, 100, 100],
                    "landmarks": {
                        "left_eye": [30, 40],
                        "right_eye": [70, 40],
                        "nose": [50, 60],
                        "left_mouth": [30, 80],
                        "right_mouth": [70, 80]
                    }
                }
            ]
        }
    
    def _generate_sample_data(self, service_type, start_date, end_date):
        """
        Tạo dữ liệu mẫu cho mục đích demo
        
        Args:
            service_type (str): Loại dịch vụ
            start_date (datetime): Ngày bắt đầu
            end_date (datetime): Ngày kết thúc
            
        Returns:
            DataFrame: Dữ liệu mẫu
        """
        import numpy as np
        
        # Tạo dãy ngày
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Số lượng mẫu
        n_samples = len(date_range) * 20  # 20 mẫu cho mỗi ngày
        
        # Tạo dữ liệu mẫu
        data = {
            'date': np.random.choice(date_range, n_samples),
            'api_type': service_type,
            'status': np.random.choice(['success', 'error'], n_samples, p=[0.95, 0.05]),
            'response_time': np.random.normal(200, 50, n_samples),
            'cost': np.random.uniform(1000, 5000, n_samples),
            'request_id': [f"req_{i}" for i in range(n_samples)]
        }
        
        # Tạo DataFrame
        df = pd.DataFrame(data)
        
        # Sắp xếp theo ngày
        df = df.sort_values(by='date')
        
        # Thêm cột count cho biểu đồ theo thời gian
        df['count'] = 1
        
        return df