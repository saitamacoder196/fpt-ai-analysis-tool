import requests
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MayaPortalService:
    """
    Service để tương tác với Maya Portal API
    """
    
    def __init__(self, username, password):
        """
        Khởi tạo service với thông tin đăng nhập
        
        Args:
            username (str): Tên đăng nhập Maya Portal
            password (str): Mật khẩu Maya Portal
        """
        self.username = username
        self.password = password
        self.base_url = "https://maya.fpt.ai/api"
        self.token = None
        self.session = requests.Session()
    
    def login(self):
        """
        Đăng nhập vào Maya Portal và lấy token
        
        Returns:
            bool: True nếu đăng nhập thành công, False nếu thất bại
        """
        # Trong thực tế, bạn sẽ gọi API thực sự từ Maya Portal
        # url = f"{self.base_url}/auth/login"
        # 
        # payload = {
        #     "username": self.username,
        #     "password": self.password
        # }
        # 
        # try:
        #     response = self.session.post(url, json=payload)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data.get("success"):
        #         self.token = data.get("token")
        #         self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        #         return True
        #     else:
        #         print(f"Login failed: {data.get('message')}")
        #         return False
        # except Exception as e:
        #     print(f"Error during login: {str(e)}")
        #     return False
        
        # Giả lập đăng nhập thành công
        self.token = "fake_token_for_demonstration"
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        return True
    
    def get_projects(self, status=None, start_date=None, end_date=None):
        """
        Lấy danh sách dự án từ Maya Portal
        
        Args:
            status (str, optional): Lọc theo trạng thái dự án
            start_date (datetime, optional): Ngày bắt đầu
            end_date (datetime, optional): Ngày kết thúc
            
        Returns:
            DataFrame: Danh sách dự án dạng pandas DataFrame
        """
        # Kiểm tra đăng nhập
        if not self.token:
            if not self.login():
                return pd.DataFrame()
        
        # Trong thực tế, bạn sẽ gọi API thực sự từ Maya Portal
        # url = f"{self.base_url}/projects"
        # 
        # params = {}
        # if status:
        #     params["status"] = status
        # if start_date:
        #     params["start_date"] = start_date.strftime("%Y-%m-%d")
        # if end_date:
        #     params["end_date"] = end_date.strftime("%Y-%m-%d")
        # 
        # try:
        #     response = self.session.get(url, params=params)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data.get("success"):
        #         return pd.DataFrame(data.get("projects"))
        #     else:
        #         print(f"Failed to get projects: {data.get('message')}")
        #         return pd.DataFrame()
        # except Exception as e:
        #     print(f"Error getting projects: {str(e)}")
        #     return pd.DataFrame()
        
        # Tạo dữ liệu mẫu thay thế
        return self._generate_projects_data(status, start_date, end_date)
    
    def get_personnel(self, department=None, position=None):
        """
        Lấy danh sách nhân sự từ Maya Portal
        
        Args:
            department (str, optional): Lọc theo phòng ban
            position (str, optional): Lọc theo vị trí
            
        Returns:
            DataFrame: Danh sách nhân sự dạng pandas DataFrame
        """
        # Kiểm tra đăng nhập
        if not self.token:
            if not self.login():
                return pd.DataFrame()
        
        # Trong thực tế, bạn sẽ gọi API thực sự từ Maya Portal
        # url = f"{self.base_url}/personnel"
        # 
        # params = {}
        # if department:
        #     params["department"] = department
        # if position:
        #     params["position"] = position
        # 
        # try:
        #     response = self.session.get(url, params=params)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data.get("success"):
        #         return pd.DataFrame(data.get("personnel"))
        #     else:
        #         print(f"Failed to get personnel: {data.get('message')}")
        #         return pd.DataFrame()
        # except Exception as e:
        #     print(f"Error getting personnel: {str(e)}")
        #     return pd.DataFrame()
        
        # Tạo dữ liệu mẫu thay thế
        return self._generate_personnel_data(department, position)
    
    def get_customers(self, status=None, type=None):
        """
        Lấy danh sách khách hàng từ Maya Portal
        
        Args:
            status (str, optional): Lọc theo trạng thái khách hàng
            type (str, optional): Lọc theo loại khách hàng
            
        Returns:
            DataFrame: Danh sách khách hàng dạng pandas DataFrame
        """
        # Kiểm tra đăng nhập
        if not self.token:
            if not self.login():
                return pd.DataFrame()
        
        # Trong thực tế, bạn sẽ gọi API thực sự từ Maya Portal
        # url = f"{self.base_url}/customers"
        # 
        # params = {}
        # if status:
        #     params["status"] = status
        # if type:
        #     params["type"] = type
        # 
        # try:
        #     response = self.session.get(url, params=params)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data.get("success"):
        #         return pd.DataFrame(data.get("customers"))
        #     else:
        #         print(f"Failed to get customers: {data.get('message')}")
        #         return pd.DataFrame()
        # except Exception as e:
        #     print(f"Error getting customers: {str(e)}")
        #     return pd.DataFrame()
        
        # Tạo dữ liệu mẫu thay thế
        return self._generate_customers_data(status, type)
    
    def get_financial_reports(self, start_date=None, end_date=None):
        """
        Lấy báo cáo tài chính từ Maya Portal
        
        Args:
            start_date (datetime, optional): Ngày bắt đầu
            end_date (datetime, optional): Ngày kết thúc
            
        Returns:
            DataFrame: Báo cáo tài chính dạng pandas DataFrame
        """
        # Kiểm tra đăng nhập
        if not self.token:
            if not self.login():
                return pd.DataFrame()
        
        # Trong thực tế, bạn sẽ gọi API thực sự từ Maya Portal
        # url = f"{self.base_url}/financial-reports"
        # 
        # params = {}
        # if start_date:
        #     params["start_date"] = start_date.strftime("%Y-%m-%d")
        # if end_date:
        #     params["end_date"] = end_date.strftime("%Y-%m-%d")
        # 
        # try:
        #     response = self.session.get(url, params=params)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data.get("success"):
        #         return pd.DataFrame(data.get("reports"))
        #     else:
        #         print(f"Failed to get financial reports: {data.get('message')}")
        #         return pd.DataFrame()
        # except Exception as e:
        #     print(f"Error getting financial reports: {str(e)}")
        #     return pd.DataFrame()
        
        # Tạo dữ liệu mẫu thay thế
        return self._generate_financial_reports(start_date, end_date)
    
    def get_operational_reports(self, report_type=None, department=None, start_date=None, end_date=None):
        """
        Lấy báo cáo hoạt động từ Maya Portal
        
        Args:
            report_type (str, optional): Loại báo cáo
            department (str, optional): Phòng ban
            start_date (datetime, optional): Ngày bắt đầu
            end_date (datetime, optional): Ngày kết thúc
            
        Returns:
            DataFrame: Báo cáo hoạt động dạng pandas DataFrame
        """
        # Kiểm tra đăng nhập
        if not self.token:
            if not self.login():
                return pd.DataFrame()
        
        # Trong thực tế, bạn sẽ gọi API thực sự từ Maya Portal
        # url = f"{self.base_url}/operational-reports"
        # 
        # params = {}
        # if report_type:
        #     params["type"] = report_type
        # if department:
        #     params["department"] = department
        # if start_date:
        #     params["start_date"] = start_date.strftime("%Y-%m-%d")
        # if end_date:
        #     params["end_date"] = end_date.strftime("%Y-%m-%d")
        # 
        # try:
        #     response = self.session.get(url, params=params)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data.get("success"):
        #         return pd.DataFrame(data.get("reports"))
        #     else:
        #         print(f"Failed to get operational reports: {data.get('message')}")
        #         return pd.DataFrame()
        # except Exception as e:
        #     print(f"Error getting operational reports: {str(e)}")
        #     return pd.DataFrame()
        
        # Tạo dữ liệu mẫu thay thế
        return self._generate_operational_reports(report_type, department, start_date, end_date)
    
    def _generate_projects_data(self, status=None, start_date=None, end_date=None):
        """
        Tạo dữ liệu dự án mẫu cho mục đích demo
        
        Args:
            status (str, optional): Lọc theo trạng thái dự án
            start_date (datetime, optional): Ngày bắt đầu
            end_date (datetime, optional): Ngày kết thúc
            
        Returns:
            DataFrame: Dữ liệu mẫu
        """
        # Số lượng dự án
        n_samples = 50
        
        # Đặt seed để có kết quả nhất quán
        np.random.seed(42)
        
        # Tạo dữ liệu dự án
        project_names = [f"Project {chr(65 + i)}" for i in range(n_samples)]
        project_types = np.random.choice(["AI", "Blockchain", "Web/App", "IoT", "Khác"], n_samples)
        statuses = np.random.choice(["Đang thực hiện", "Hoàn thành", "Tạm dừng", "Hủy bỏ"], n_samples, p=[0.5, 0.3, 0.1, 0.1])
        
        # Ngày bắt đầu và kết thúc
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        delta = (end_date - start_date).days
        start_dates = [start_date + timedelta(days=np.random.randint(0, delta)) for _ in range(n_samples)]
        end_dates = [d + timedelta(days=np.random.randint(30, 180)) for d in start_dates]
        
        # Ngân sách và tiến độ
        budgets = np.random.uniform(50000000, 500000000, n_samples)
        progress = np.random.uniform(0, 100, n_samples)
        
        # Tạo DataFrame
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
        
        df = pd.DataFrame(data)
        
        # Lọc theo status nếu có
        if status and status != "Tất cả":
            df = df[df['status'] == status]
            
        return df
    
    def _generate_personnel_data(self, department=None, position=None):
        """
        Tạo dữ liệu nhân sự mẫu cho mục đích demo
        
        Args:
            department (str, optional): Lọc theo phòng ban
            position (str, optional): Lọc theo vị trí
            
        Returns:
            DataFrame: Dữ liệu mẫu
        """
        # Số lượng nhân sự
        n_samples = 100
        
        # Đặt seed để có kết quả nhất quán
        np.random.seed(42)
        
        # Tạo dữ liệu nhân sự
        employee_ids = [f"EMP{i:04d}" for i in range(n_samples)]
        names = [f"Employee {i}" for i in range(n_samples)]
        departments = np.random.choice(["AI Lab", "R&D", "Sales", "Marketing", "HR", "Admin"], n_samples)
        positions = np.random.choice(["Developer", "Researcher", "Manager", "Director", "Admin"], n_samples)
        join_dates = [datetime.now() - timedelta(days=np.random.randint(0, 1000)) for _ in range(n_samples)]
        salaries = np.random.normal(25000000, 5000000, n_samples)
        
        # Tạo DataFrame
        data = {
            'employee_id': employee_ids,
            'name': names,
            'department': departments,
            'position': positions,
            'join_date': join_dates,
            'salary': salaries,
            'status': np.random.choice(["Active", "On leave", "Terminated"], n_samples, p=[0.9, 0.05, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Thêm trường type
        df['type'] = "Nhân sự"
        
        # Lọc theo department nếu có
        if department and department != "Tất cả":
            df = df[df['department'] == department]
            
        # Lọc theo position nếu có
        if position and position != "Tất cả":
            df = df[df['position'] == position]
            
        return df
    
    def _generate_customers_data(self, status=None, type=None):
        """
        Tạo dữ liệu khách hàng mẫu cho mục đích demo
        
        Args:
            status (str, optional): Lọc theo trạng thái khách hàng
            type (str, optional): Lọc theo loại khách hàng
            
        Returns:
            DataFrame: Dữ liệu mẫu
        """
        # Số lượng khách hàng
        n_samples = 80
        
        # Đặt seed để có kết quả nhất quán
        np.random.seed(42)
        
        # Tạo dữ liệu khách hàng
        customer_ids = [f"CUS{i:04d}" for i in range(n_samples)]
        names = [f"Customer {i}" for i in range(n_samples)]
        types = np.random.choice(["Enterprise", "SME", "Startup", "Government"], n_samples)
        join_dates = [datetime.now() - timedelta(days=np.random.randint(0, 500)) for _ in range(n_samples)]
        contract_values = np.random.lognormal(mean=20, sigma=1, size=n_samples) * 1000000
        
        # Tạo DataFrame
        data = {
            'customer_id': customer_ids,
            'name': names,
            'type': types,
            'join_date': join_dates,
            'contract_value': contract_values,
            'status': np.random.choice(["Active", "Inactive"], n_samples, p=[0.8, 0.2]),
            'projects_count': np.random.randint(1, 10, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Lọc theo status nếu có
        if status and status != "Tất cả":
            df = df[df['status'] == status]
            
        # Lọc theo type nếu có
        if type and type != "Tất cả":
            df = df[df['type'] == type]
            
        return df
    
    def _generate_financial_reports(self, start_date=None, end_date=None):
        """
        Tạo dữ liệu báo cáo tài chính mẫu cho mục đích demo
        
        Args:
            start_date (datetime, optional): Ngày bắt đầu
            end_date (datetime, optional): Ngày kết thúc
            
        Returns:
            DataFrame: Dữ liệu mẫu
        """
        # Số lượng báo cáo (tháng)
        n_samples = 12
        
        # Đặt seed để có kết quả nhất quán
        np.random.seed(42)
        
        # Ngày báo cáo
        if not start_date:
            start_date = datetime.now().replace(day=1) - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
            
        # Tạo các tháng báo cáo
        months = pd.date_range(start=start_date.replace(day=1), 
                               end=end_date.replace(day=1), 
                               freq='MS')
        
        if len(months) < n_samples:
            n_samples = len(months)
        
        # Doanh thu, chi phí và lợi nhuận
        revenues = np.random.normal(5000000000, 1000000000, n_samples)
        expenses = np.random.normal(4000000000, 800000000, n_samples)
        profits = revenues - expenses
        
        # Tạo DataFrame
        data = {
            'month': months[:n_samples],
            'revenue': revenues,
            'expense': expenses,
            'profit': profits,
            'growth_rate': np.random.normal(0.05, 0.02, n_samples),
            'project_count': np.random.randint(5, 20, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Thêm trường type
        df['type'] = "Báo cáo tài chính"
        
        return df
    
    def _generate_operational_reports(self, report_type=None, department=None, start_date=None, end_date=None):
        """
        Tạo dữ liệu báo cáo hoạt động mẫu cho mục đích demo
        
        Args:
            report_type (str, optional): Loại báo cáo
            department (str, optional): Phòng ban
            start_date (datetime, optional): Ngày bắt đầu
            end_date (datetime, optional): Ngày kết thúc
            
        Returns:
            DataFrame: Dữ liệu mẫu
        """
        # Số lượng báo cáo
        n_samples = 20
        
        # Đặt seed để có kết quả nhất quán
        np.random.seed(42)
        
        # Ngày báo cáo
        if not start_date:
            start_date = datetime.now() - timedelta(days=180)
        if not end_date:
            end_date = datetime.now()
            
        delta = (end_date - start_date).days
        dates = [start_date + timedelta(days=np.random.randint(0, delta)) for _ in range(n_samples)]
        
        # Tạo dữ liệu báo cáo
        report_ids = [f"REP{i:04d}" for i in range(n_samples)]
        report_types = np.random.choice(["Monthly", "Quarterly", "Annual", "Special"], n_samples)
        departments = np.random.choice(["AI Lab", "R&D", "Sales", "Marketing", "HR", "Admin"], n_samples)
        
        # Tạo DataFrame
        data = {
            'report_id': report_ids,
            'date': dates,
            'type': report_types,
            'department': departments,
            'status': np.random.choice(["Draft", "Submitted", "Approved", "Rejected"], n_samples),
            'author': [f"User {i % 10 + 1}" for i in range(n_samples)]
        }
        
        df = pd.DataFrame(data)
        
        # Lọc theo report_type nếu có
        if report_type and report_type != "Tất cả":
            df = df[df['type'] == report_type]
            
        # Lọc theo department nếu có
        if department and department != "Tất cả":
            df = df[df['department'] == department]
            
        return df