# Xử Lý Ngôn Ngữ Tự Nhiên - Phân Tích Cảm Xúc (Sentiment Analysis)

Bài lab Xử lý ngôn ngữ tự nhiên: xây dựng và so sánh hai mô hình học sâu dựa trên
mạng nơ-ron hồi quy (RNN) - **GRU** và **Bidirectional LSTM (BiLSTM)** - cho bài
toán phân loại cảm xúc nhị phân (tích cực / tiêu cực) trên bộ dữ liệu **Yelp
Polarity**.

## Nội dung chính

- **Tập dữ liệu:** [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity)
  (tải qua thư viện `datasets` của Hugging Face), lấy mẫu con 20.000 review cho
  tập train và 10.000 review cho tập test.
- **Tiền xử lý dữ liệu:**
  - Tokenize văn bản bằng `Tokenizer` của Keras (vocabulary tối đa 20.000 từ,
    xử lý từ hiếm bằng token `<OOV>`).
  - Phân tích thống kê độ dài review (trung bình, min, max, trung vị) và trực
    quan hóa phân bố độ dài bằng histogram.
  - Padding/truncating các chuỗi token về độ dài cố định `MAX_LENGTH = 200`.
- **Kiến trúc mô hình:**
  - Một hàm dùng chung `build_sequence_model()` cho phép dựng mô hình dạng
    `GRU` hoặc `BiLSTM` bằng cách đổi tham số `model_type`.
  - Gồm tầng `Embedding` (128 chiều, `mask_zero=True`), 2 tầng hồi quy
    (GRU hoặc Bidirectional LSTM), `Dropout` và `Dense` đầu ra.
- **Huấn luyện:**
  - Tối đa 30 epoch, batch size 128, `validation_split=0.2`.
  - Sử dụng `EarlyStopping` (theo dõi `val_loss`, patience 3, khôi phục
    trọng số tốt nhất) để tránh overfitting.
  - Đo thời gian huấn luyện của từng mô hình để so sánh hiệu năng.
- **Đánh giá mô hình:**
  - Vẽ biểu đồ loss/accuracy theo epoch cho cả tập train và validation.
  - Tính accuracy, precision, recall (tự cài đặt thủ công từ ma trận nhầm
    lẫn — không dùng sklearn) trên tập test.
  - Hiển thị confusion matrix dạng bảng cho cả hai mô hình.
  - Hàm `predict_review_sentiment()` cho phép nhập một câu review bất kỳ và
    xem xác suất/nhãn dự đoán từ cả GRU và BiLSTM.

## Yêu cầu môi trường

```bash
pip install tensorflow datasets numpy pandas matplotlib
```

Notebook được thiết kế để chạy trên môi trường có GPU (khuyến nghị GPU T4
trên Kaggle hoặc Google Colab) để tăng tốc độ huấn luyện.

## Cách chạy

1. Mở file `Xử_lý_ngôn_ngữ_tự_nhiên.ipynb` bằng Jupyter Notebook, Google Colab
   hoặc Kaggle Notebook.
2. Chạy tuần tự từng cell từ trên xuống dưới:
   - Cell tải và tiền xử lý dữ liệu.
   - Cell xây dựng và huấn luyện mô hình GRU.
   - Cell xây dựng và huấn luyện mô hình BiLSTM.
   - Cell đánh giá và so sánh kết quả hai mô hình.
3. (Tùy chọn) Thử dự đoán cảm xúc cho review tự nhập ở cell cuối cùng.

## Cấu trúc file

```
.
├── Xử_lý_ngôn_ngữ_tự_nhiên.ipynb   # Notebook chính
└── README.md
```

## Ghi chú

- `SEED = 42` được cố định cho `random`, `numpy` để đảm bảo khả năng tái lập
  kết quả.
- Nếu gặp lỗi tương thích cuDNN khi chạy GRU/LSTM trên một số môi trường GPU,
  có thể cần thiết lập `use_cudnn=False` hoặc kiểm tra phiên bản CUDA/cuDNN
  phù hợp với TensorFlow đang sử dụng.
