# Tìm hiểu sâu về CRNN & CTC Loss trong OCR

Trong bài viết này, chúng ta sẽ khám phá hoạt động bên trong của CRNN (Mạng nơ-ron tích chập tái phát) cho Nhận dạng ký tự quang học (OCR), đặc biệt tập trung vào **Hàm mất mát CTC (Connectionist Temporal Classification)**.

Chúng ta sẽ đi qua phần triển khai trong dự án này (`src/`) và phân tích một mẫu cụ thể ("233HP3") từ `pipeline_deep_dive.ipynb` để làm rõ cách tính toán gradient.

## 1. Quy trình xử lý dữ liệu (`src/dataset.py`)

Trước khi huấn luyện, chúng ta cần chuẩn bị hình ảnh. Lớp `OCRDataset` đảm nhiệm việc này:

1.  **Tải ảnh**: Ảnh được tải và chuyển đổi sang thang độ xám (chế độ `L`).
2.  **Thay đổi kích thước (Resizing)**: Tất cả ảnh được thay đổi kích thước về chiều cao cố định là **32px** và chiều rộng **128px**. Kích thước cố định này rất quan trọng để đưa vào batch.
3.  **Biến đổi (Transforms)**: Ảnh được chuyển đổi thành Tensor và chuẩn hóa về phạm vi `[-1, 1]`.
4.  **Mã hóa nhãn**: Kết hợp với danh sách từ vựng (vocab), nhãn văn bản được chuyển đổi thành chuỗi số nguyên (ví dụ: 'A' -> 1, 'B' -> 2). **Chỉ số 0 được dành riêng cho token Blank (Khoảng trắng) của CTC.**

## 2. Kiến trúc mô hình: CRNN (`src/model.py`)

Mô hình bao gồm ba thành phần chính:

### A. Trích xuất đặc trưng Convolutional (CNN)
Xương sống là một mạng CNN 7 lớp (kiểu VGG).
*   **Đầu vào**: `[Batch, 1, 32, 128]`
*   Nó sử dụng **Max Pooling** với bước nhảy (stride) `(2, 2)` ban đầu, nhưng sau đó sử dụng `(2, 1)`.
*   **Tại sao lại là `(2, 1)`?** Chúng ta muốn nén Chiều cao cụ thể (về 1) nhưng giữ nguyên **Chiều rộng** (Thời gian).
*   **Đầu ra**: Bản đồ đặc trưng có kích thước `[Batch, 512, 1, 26]`. Chiều rộng `26` trở thành độ dài chuỗi thời gian $T$ của chúng ta.

### B. Map-to-Sequence (Bản đồ sang Chuỗi)
Tensor được định hình lại để tương thích với RNN: `[Width, Batch, Channels]` = `[26, Batch, 512]`. Bây giờ, mỗi cột "pixel" trong bản đồ đặc trưng được coi là một bước thời gian.

### C. Các lớp Recurrent (RNN)
Hai lớp `BidirectionalLSTM` nắm bắt ngữ cảnh của chuỗi.
*   **Đầu vào**: `[26, Batch, 512]`
*   **Đầu ra**: `[26, Batch, n_class]`. Đây là các "Logits".

## 3. Giải thích CTC Loss với một mẫu thực tế

Hãy xem xét mẫu được xử lý trong `pipeline_deep_dive.ipynb`.
*   **Nhãn gốc (Ground Truth)**: `"233HP3"`
*   **Chiều rộng đầu vào mô hình ($T$)**: 26 bước thời gian.

Thách thức là **Gióng hàng (Alignment)**. Chúng ta không biết chữ "H" nằm ở *đâu* trong 26 bước đó. Nó có thể ở bước 10, 11 hoặc 12. CTC giải quyết vấn đề này bằng cách xem xét **tất cả các cách gióng hàng hợp lệ**.

<image>Chèn "Probability Heatmap (Model Output)" từ notebook để hiển thị các dự đoán thô</image>

### Bước 3a: Mục tiêu mở rộng (The Extended Target)
CTC giới thiệu một token **Blank (`-`)** để xử lý các ký tự lặp lại và sự phân tách.
*   Gốc: `2 3 3 H P 3`
*   Mở rộng ($L'$): `- 2 - 3 - 3 - H - P - 3 -`
Chúng ta tính toán các đường đi xác suất qua chuỗi mở rộng này.

### Bước 3b: Thuật toán Forward ($\alpha$)
**Ma trận Alpha** ($\alpha_{t,s}$) lưu trữ xác suất gióng hàng $s$ ký tự đầu tiên của mục tiêu tại bước thời gian $t$.
Tại mỗi bước, chúng ta có thể:
1.  **Ở lại (Stay)**: Tiếp tục xuất ra cùng một ký tự (ví dụ: `2` -> `2`).
2.  **Tiếp theo (Next)**: Di chuyển đến token tiếp theo (ví dụ: `2` -> `-`).
3.  **Bỏ qua (Skip)**: Nhảy qua một khoảng trắng (chỉ khi ký tự tiếp theo khác nhau, ví dụ: `2` -> `3`).

<image>Chèn trực quan hóa "Alpha Matrix" từ notebook hiển thị sự tích lũy xác suất forward</image>

### Bước 3c: Tính toán Gradient ("Tại sao")
Mô hình học như thế nào? Nó thực sự so sánh những gì nó **dự đoán** với những gì việc gióng hàng **yêu cầu**.

1.  **Gióng hàng mềm - Soft Alignment ($\gamma$)**: được tính toán bằng cách sử dụng biến Forward ($\alpha$) và Backward ($\beta$).
    $$ \gamma_{t,s} = \frac{\alpha_{t,s} \cdot \beta_{t,s}}{P(z|x)} $$
    Điều này cho chúng ta biết: "Xác suất chúng ta *phải* ở trạng thái $s$ tại thời điểm $t$ là bao nhiêu khi biết toàn bộ nhãn?"

    <image>Chèn biểu đồ "Soft Alignment (Gamma)" từ notebook hiển thị đường đi hợp lệ sáng nhất</image>

2.  **Công thức Gradient**:
    Đạo hàm của Loss đối với đầu ra mạng $y_k^t$ là:
    $$ \frac{\partial L}{\partial y_k^t} = y_k^t - \sum_{s \in \text{states}(k)} \gamma_{t,s} $$
    Hay đơn giản là: **Gradient = Dự đoán (Prediction) - Gióng hàng mục tiêu (Target Alignment)**.

    *   Nếu việc gióng hàng nói rằng chúng ta *nên* xuất ra '3' ($\gamma$ cao) nhưng mô hình dự đoán xác suất thấp ($y$), gradient sẽ âm, đẩy xác suất **lên**.
    *   Nếu mô hình dự đoán '3' ở nơi không có đường đi hợp lệ nào tồn tại ($\gamma$ thấp), gradient sẽ dương, đẩy xác suất **xuống**.

Cơ chế này cho phép CRNN làm rõ các dự đoán của nó theo thời gian mà không cần chú thích cấp độ pixel!

### Tài liệu tham khảo
*   [Sequence Modeling With CTC (Distill.pub)](https://distill.pub/2017/ctc/)
*   [Breaking down the CTC Loss (Ogunlao)](https://ogunlao.github.io/blog/2020/07/17/breaking-down-ctc-loss.html)
*   Notebook `pipeline_deep_dive.ipynb` của chúng tôi.
