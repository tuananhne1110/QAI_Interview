# Sử dụng hình ảnh cơ bản Python 3.8
FROM python:3.8-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp yêu cầu và cài đặt phụ thuộc
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng vào container
COPY . .

# Sao chép thư mục weights vào container
COPY logistic_regression_weights.h5 /app/Question_4/

# Mở cổng ứng dụng Flask
EXPOSE 5000

# Lệnh để chạy ứng dụng Flask
CMD ["python", "app.py"]