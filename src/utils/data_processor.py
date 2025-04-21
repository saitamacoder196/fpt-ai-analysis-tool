import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import re

class DataProcessor:
    """
    Utility class để xử lý và phân tích dữ liệu từ các nguồn khác nhau
    """
    
    def __init__(self):
        """
        Khởi tạo DataProcessor
        """
        pass
    
    @staticmethod
    def clean_data(df):
        """
        Làm sạch dữ liệu cơ bản
        
        Args:
            df (DataFrame): DataFrame cần làm sạch
            
        Returns:
            DataFrame: DataFrame đã được làm sạch
        """
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        cleaned_df = df.copy()
        
        # Loại bỏ hàng trùng lặp
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Xử lý giá trị null
        for col in cleaned_df.columns:
            # Với cột số, thay thế null bằng giá trị trung bình hoặc 0
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # Nếu null > 50%, thay bằng 0
                if cleaned_df[col].isnull().mean() > 0.5:
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                # Nếu null <= 50%, thay bằng giá trị trung bình
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            # Với cột chuỗi, thay thế null bằng chuỗi rỗng
            elif pd.api.types.is_string_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna("")
            # Với cột datetime, giữ nguyên (sẽ xử lý tùy trường hợp)
            elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                pass
        
        return cleaned_df
    
    @staticmethod
    def detect_outliers(df, columns=None, method='zscore', threshold=3.0):
        """
        Phát hiện các giá trị ngoại lệ trong dữ liệu
        
        Args:
            df (DataFrame): DataFrame cần kiểm tra
            columns (list, optional): Danh sách các cột cần kiểm tra. Nếu None, kiểm tra tất cả cột số
            method (str): Phương pháp phát hiện ('zscore', 'iqr')
            threshold (float): Ngưỡng để xác định ngoại lệ
            
        Returns:
            DataFrame: DataFrame chứa các hàng có giá trị ngoại lệ
        """
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        data = df.copy()
        
        # Nếu không chỉ định cột, lấy tất cả cột số
        if columns is None:
            columns = data.select_dtypes(include=['number']).columns.tolist()
        
        # DataFrame để lưu kết quả
        outliers = pd.DataFrame()
        
        # Xử lý từng cột
        for col in columns:
            if col not in data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
                
            # Phương pháp Z-score
            if method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask = z_scores > threshold
                
            # Phương pháp IQR (Interquartile Range)
            elif method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            
            else:
                raise ValueError(f"Phương pháp không hợp lệ: {method}")
            
            # Thêm vào DataFrame kết quả
            col_outliers = data[outlier_mask].copy()
            if not outliers.empty and not col_outliers.empty:
                outliers = pd.concat([outliers, col_outliers])
            elif not col_outliers.empty:
                outliers = col_outliers
        
        # Loại bỏ trùng lặp
        outliers = outliers.drop_duplicates()
        
        return outliers
    
    @staticmethod
    def aggregate_by_time(df, date_column, time_period='day', agg_functions=None):
        """
        Tổng hợp dữ liệu theo thời gian
        
        Args:
            df (DataFrame): DataFrame cần tổng hợp
            date_column (str): Tên cột chứa dữ liệu ngày
            time_period (str): Chu kỳ thời gian (day, week, month, quarter, year)
            agg_functions (dict, optional): Dictionary định nghĩa các hàm tổng hợp cho từng cột
            
        Returns:
            DataFrame: DataFrame đã được tổng hợp
        """
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        data = df.copy()
        
        # Chuyển đổi cột ngày sang datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        
        # Tạo cột mới để nhóm dữ liệu
        if time_period == 'day':
            data['time_group'] = data[date_column].dt.date
        elif time_period == 'week':
            data['time_group'] = data[date_column].dt.to_period('W').dt.to_timestamp()
        elif time_period == 'month':
            data['time_group'] = data[date_column].dt.to_period('M').dt.to_timestamp()
        elif time_period == 'quarter':
            data['time_group'] = data[date_column].dt.to_period('Q').dt.to_timestamp()
        elif time_period == 'year':
            data['time_group'] = data[date_column].dt.to_period('Y').dt.to_timestamp()
        else:
            raise ValueError(f"Chu kỳ thời gian không hợp lệ: {time_period}")
        
        # Nếu không có hàm tổng hợp được chỉ định, tạo hàm mặc định
        if agg_functions is None:
            # Lấy tất cả cột số
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            # Loại bỏ cột thời gian nếu có trong danh sách
            if date_column in numeric_cols:
                numeric_cols.remove(date_column)
            
            # Tạo hàm tổng hợp mặc định (sum, mean, count)
            agg_functions = {}
            for col in numeric_cols:
                agg_functions[col] = ['sum', 'mean', 'count']
        
        # Thực hiện tổng hợp
        result = data.groupby('time_group').agg(agg_functions)
        
        # Làm phẳng tên cột nếu cần
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(col).strip() for col in result.columns.values]
        
        return result
    
    @staticmethod
    def extract_trends(df, date_column, value_column, window=7):
        """
        Trích xuất xu hướng từ dữ liệu theo thời gian
        
        Args:
            df (DataFrame): DataFrame cần phân tích
            date_column (str): Tên cột chứa dữ liệu ngày
            value_column (str): Tên cột chứa giá trị cần phân tích
            window (int): Kích thước cửa sổ cho việc làm mượt (moving average)
            
        Returns:
            DataFrame: DataFrame chứa xu hướng và dữ liệu làm mượt
        """
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        data = df.copy()
        
        # Chuyển đổi cột ngày sang datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        
        # Sắp xếp dữ liệu theo thời gian
        data = data.sort_values(by=date_column)
        
        # Tính moving average
        data[f'{value_column}_ma'] = data[value_column].rolling(window=window).mean()
        
        # Tính tỷ lệ thay đổi
        data[f'{value_column}_pct_change'] = data[value_column].pct_change() * 100
        
        # Tính xu hướng
        # 1. Tính giá trị trung bình của x (time) và y (value)
        data['x'] = range(len(data))
        x_mean = data['x'].mean()
        y_mean = data[value_column].mean()
        
        # 2. Tính hệ số góc (slope) của đường thẳng xu hướng
        numerator = ((data['x'] - x_mean) * (data[value_column] - y_mean)).sum()
        denominator = ((data['x'] - x_mean) ** 2).sum()
        slope = numerator / denominator if denominator != 0 else 0
        
        # 3. Tính điểm cắt trục y (intercept)
        intercept = y_mean - slope * x_mean
        
        # 4. Thêm giá trị xu hướng vào DataFrame
        data[f'{value_column}_trend'] = intercept + slope * data['x']
        
        # Loại bỏ cột x tạm thời
        data = data.drop(columns=['x'])
        
        return data
    
    @staticmethod
    def compare_datasets(df1, df2, key_columns, value_columns=None):
        """
        So sánh hai DataFrame và tìm sự khác biệt
        
        Args:
            df1 (DataFrame): DataFrame thứ nhất
            df2 (DataFrame): DataFrame thứ hai
            key_columns (list): Danh sách các cột khóa để so sánh
            value_columns (list, optional): Danh sách các cột giá trị cần so sánh. Nếu None, so sánh tất cả
            
        Returns:
            dict: Dictionary chứa kết quả so sánh
        """
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        data1 = df1.copy()
        data2 = df2.copy()
        
        # Nếu không chỉ định cột giá trị, lấy tất cả cột không phải khóa
        if value_columns is None:
            value_columns = [col for col in data1.columns if col not in key_columns]
            # Đảm bảo chỉ so sánh các cột xuất hiện trong cả hai DataFrame
            value_columns = [col for col in value_columns if col in data2.columns]
        
        # Dictionary lưu kết quả
        comparison = {
            'only_in_df1': [],
            'only_in_df2': [],
            'different_values': []
        }
        
        # Chuyển đổi các DataFrame sang định dạng dễ so sánh hơn
        data1_dict = data1.set_index(key_columns).to_dict(orient='index')
        data2_dict = data2.set_index(key_columns).to_dict(orient='index')
        
        # Tìm các khóa chỉ xuất hiện trong DataFrame thứ nhất
        only_in_df1 = set(data1_dict.keys()) - set(data2_dict.keys())
        for key in only_in_df1:
            comparison['only_in_df1'].append({
                'key': key,
                'data': data1.loc[data1.set_index(key_columns).index == key]
            })
        
        # Tìm các khóa chỉ xuất hiện trong DataFrame thứ hai
        only_in_df2 = set(data2_dict.keys()) - set(data1_dict.keys())
        for key in only_in_df2:
            comparison['only_in_df2'].append({
                'key': key,
                'data': data2.loc[data2.set_index(key_columns).index == key]
            })
        
        # Tìm các khóa xuất hiện trong cả hai DataFrame nhưng có giá trị khác nhau
        common_keys = set(data1_dict.keys()) & set(data2_dict.keys())
        for key in common_keys:
            diff_values = {}
            for col in value_columns:
                if col in data1_dict[key] and col in data2_dict[key]:
                    val1 = data1_dict[key][col]
                    val2 = data2_dict[key][col]
                    
                    # So sánh giá trị, xử lý cả trường hợp NaN
                    if pd.isna(val1) and pd.isna(val2):
                        continue
                    elif pd.isna(val1) or pd.isna(val2):
                        diff_values[col] = {'df1': val1, 'df2': val2}
                    elif val1 != val2:
                        diff_values[col] = {'df1': val1, 'df2': val2}
            
            if diff_values:
                comparison['different_values'].append({
                    'key': key,
                    'differences': diff_values
                })
        
        return comparison
    
    @staticmethod
    def export_to_csv(df, filename, output_dir='data'):
        """
        Xuất DataFrame ra file CSV
        
        Args:
            df (DataFrame): DataFrame cần xuất
            filename (str): Tên file (không bao gồm đường dẫn)
            output_dir (str): Thư mục đầu ra
            
        Returns:
            str: Đường dẫn đến file đã xuất
        """
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Đường dẫn đầy đủ
        filepath = os.path.join(output_dir, filename)
        
        # Xuất ra CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        return filepath
    
    @staticmethod
    def export_to_excel(df, filename, output_dir='data', sheet_name='Sheet1'):
        """
        Xuất DataFrame ra file Excel
        
        Args:
            df (DataFrame): DataFrame cần xuất
            filename (str): Tên file (không bao gồm đường dẫn)
            output_dir (str): Thư mục đầu ra
            sheet_name (str): Tên sheet
            
        Returns:
            str: Đường dẫn đến file đã xuất
        """
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Đường dẫn đầy đủ
        filepath = os.path.join(output_dir, filename)
        
        # Xuất ra Excel
        df.to_excel(filepath, sheet_name=sheet_name, index=False)
        
        return filepath
    
    @staticmethod
    def export_to_json(df, filename, output_dir='data', orient='records'):
        """
        Xuất DataFrame ra file JSON
        
        Args:
            df (DataFrame): DataFrame cần xuất
            filename (str): Tên file (không bao gồm đường dẫn)
            output_dir (str): Thư mục đầu ra
            orient (str): Định dạng JSON (records, split, index, columns, values)
            
        Returns:
            str: Đường dẫn đến file đã xuất
        """
        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Đường dẫn đầy đủ
        filepath = os.path.join(output_dir, filename)
        
        # Xuất ra JSON
        df.to_json(filepath, orient=orient)
        
        return filepath