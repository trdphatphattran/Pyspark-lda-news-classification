# HỆ THỐNG PHÂN LOẠI CHỦ ĐỀ CỦA VĂN BẢN (LDA + TF-IDF)  

## 📌 Tổng quan dự án  
Dự án xây dựng mô hình học máy không giám sát (Unsupervised Learning) nhằm tự động khám phá và phân loại chủ đề từ kho dữ liệu tin tức tiếng Việt khổng lồ, ứng dụng thuật toán LDA và TF-IDF trên nền tảng tính toán phân tán.  

## 🚀 Điểm Nổi Bật  
* **Quy mô dữ liệu:** Xử lý hơn 7.000.000 bản ghi tin tức (Parquet format).  
* **Hiệu suất:** Tối ưu hóa tính toán trên cụm Spark Cluster (32 Cores).  
* **Công nghệ:** Kết hợp TF-IDF để trích xuất đặc trưng và LDA để phân cụm chủ đề ẩn.  
* **Giao diện:** Chatbot tương tác thời gian thực tích hợp biểu đồ trực quan hóa xác suất.

## 🛠 Kiến Trúc Hệ Thống  
* **1. Tiền xử lý:** Tách từ (Tokenization), loại bỏ từ dừng (Stopwords Removal) bằng thư viện NLP chuyên dụng.  
* **2. Trích xuất đặc trưng:** Sử dụng HashingTF và IDF để chuyển đổi văn bản thành vector số.  
* **3. Huấn luyện mô hình:** Thuật toán LDA (Latent Dirichlet Allocation) phân cụm dữ liệu thành 4 chủ đề chính.  
* **4. Ánh xạ chủ đề (Mapping):** Định danh Topic ID sang nhãn ngôn ngữ (Thể thao, Kinh doanh, Thế giới, Xã hội) dựa trên mật độ phân hữu nhãn đối chứng.

## 📊 Kết Quả  
* **Độ chính xác (Accuracy):** Đạt ~70% trên tập dữ liệu thô (unsupervised).  
* **Khả năng mở rộng:** Hệ thống có thể xử lý thêm dữ liệu mới mà không cần huấn luyện lại từ đầu nhờ Pipeline đã tối ưu.

## 🛠 Công Nghệ Sử Dụng  
* **Core:** Python, Apache Spark (PySpark), LDA, TF-IDF.
* **Model Management:** Pickle.
* **Data Processing:** Pandas, Regex.
* **Visualization:** Matplotlib, Seaborn.
* **Deployment:** Streamlit.

## Demo Web  
### 1. Chọn cách nhập văn bản  
- Ở đây có 3 cách nhập gồm nhập trực tiếp, gắn PDF/Word, gắn URL.
- Sau khi gắn xong, hãy nhập những câu hỏi yêu cầu hệ thống dự đoán thể loại văn bản trên.
<img width="1458" height="745" alt="image" src="https://github.com/user-attachments/assets/2d4e9d66-0386-457a-a36b-e2324c418f77" />

<img width="1114" height="633" alt="image" src="https://github.com/user-attachments/assets/a454e226-c634-4f0d-88c1-5ffe1b5c2965" />  

### 2. Lí do đưa ra tên chủ đề  
- Yêu cầu hệ thống đưa ra lí do vì sao lại chọn chủ đề trên, ở đây hệ thống thực hiện việc đếm các từ xuất hiện nhiều nhất.
<img width="915" height="319" alt="image" src="https://github.com/user-attachments/assets/fe7f6307-91e8-4094-a76b-ba71cf7d62d0" />

## 📂 Cấu trúc thư mục  
```text
├── app.py                            # Code giao diện Streamlit
├── LDA_Model_7M_Final.pkl            # Trọng số mô hình tốt nhất sau khi train
├── requirements.txt
└── README.md
```

## 💻 Hướng dẫn sử dụng  
### 1. Clone Repository  
```python
git clone https://github.com/trdphatphattran/Pyspark-lda-news-classification.git
cd Pyspark-lda-news-classification
```
### 2. Cài thư viện  
```python
pip install -r requirements.txt
```
### 3. Chạy Streamlit  
```python
streamlit run app.py
```
## 📬 Thông tin liên hệ

Nếu bạn có bất kỳ câu hỏi nào về dự án hoặc muốn hợp tác, vui lòng liên hệ với mình qua:

* **Họ và tên:** Trần Đại Phát
* **LinkedIn:** [Phat Tran](https://www.linkedin.com/in/phat-tran-9ba42a341/)
* **GitHub:** [trdphatphattran](https://github.com/trdphatphattran)
* **Email:** phattrandai15062005@gmail.com
* **Phone:** 0908647977 

