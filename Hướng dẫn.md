# HƯỚNG DẪN CẤU HÌNH & CHẠY DỰ ÁN GREENCYCLE AI
## HỆ THỐNG SO SÁNH SONG SONG MÔ HÌNH KÉP: GREENCYCLE NET (TENSORFLOW) VS YOLO26 (PYTORCH)

Chào mừng bạn đến với tài liệu hướng dẫn vận hành chi tiết nhất của dự án **GreenCycle AI** - Hệ thống phân loại rác thải thông minh ứng dụng Học Sâu (Deep Learning). 

---

## 📂 Mục lục
1. [Giới thiệu tổng quan Dự án](#1-giới-thiệu-tổng-quan-dự-án)
2. [Cấu trúc Thư mục Dự án](#2-cấu-trúc-thư-mục-dự-án)
3. [Yêu cầu Hệ thống & Chuẩn bị Môi trường](#3-yêu-cầu-hệ-thống--chuẩn-bị-môi-trường)
4. [Hướng dẫn các bước Cài đặt Chi tiết](#4-hướng-dẫn-các-bước-cài-đặt-chi-tiết)
5. [Hướng dẫn Chạy dự án & Nghiệm thu Giao diện](#5-hướng-dẫn-chạy-dự-án--nghiệm-thu-giao-diện)
6. [Mổ xẻ 4 Đột phá Kỹ thuật & Toán học (Phục vụ thuyết trình)](#6-mổ-xẻ-4-đột-phá-kỹ-thuật--toán-học)
7. [Xử lý Sự cố & Câu hỏi thường gặp (FAQs)](#7-xử-lý-sự-cố--câu-hỏi-thường-gặp)

---

## 1. Giới thiệu tổng quan Dự án

**GreenCycle AI** là một ứng dụng Web sinh thái cao cấp được thiết kế theo phong cách **Glassmorphism (kính mờ)** hiện đại. Ứng dụng giải quyết bài toán phân loại rác thải sinh hoạt thành **7 nhóm chuyên biệt**:
*   `cardboard` (Giấy bìa Carton)
*   `compost` (Rác hữu cơ / Phân hủy sinh học)
*   `glass` (Thủy tinh)
*   `metal` (Kim loại / Lon nhôm)
*   `paper` (Giấy văn phòng / Báo chí)
*   `plastic` (Nhựa / Chai nhựa / Túi nilon)
*   `trash` (Rác thải còn lại, không tái chế)

### 💡 Điểm đặc sắc độc bản của dự án (Model Playground):
Không chỉ là một trang nhận diện rác thông thường, ứng dụng tích hợp **Mô hình kép song song (Dual-Model)** cho phép người dùng so sánh trực tiếp hiệu năng giữa 2 thế giới công nghệ:
1.  **GreenCycle Net (TensorFlow/Keras)**: Mô hình MobileNetV2 siêu nhẹ (~14MB) được cải tiến sâu học thuật với **khối chú ý kênh SE Block**, hàm loss **Focal Loss** tùy biến, huấn luyện rà phanh **Cosine Annealing** và trực quan hóa tính minh bạch bằng bản đồ nhiệt **Grad-CAM (Explainable AI - XAI)**.
2.  **YOLO26-cls (PyTorch/Ultralytics)**: Công nghệ nhận diện tân tiến nhất thế giới vừa ra mắt **đầu năm 2026**, tối ưu hóa cực kỳ mạnh mẽ cho thiết bị cạnh, loại bỏ DFL, suy luận biên dạng hình học siêu tốc và chính xác vượt trội.

---

## 2. Cấu trúc Thư mục Dự án

Cấu trúc thư mục thực tế của dự án của bạn được sơ đồ hóa chính xác dưới đây:

```text
Waste-or-Garbage-Classification-Using-Deep-Learning-main/
│
├── CNN - Architecture/           # Thư mục chứa các code nghiên cứu và so sánh mô hình gốc
│   ├── Waste or Garbage Classification Using ResNet.ipynb
│   ├── Waste or Garbage Classification Using ResNet.py
│   ├── Waste or Garbage Classification Using VGG16.ipynb
│   └── Waste or Garbage Classification Using VGG16.py
│
├── DataSets/                     # Thư mục chứa tập dữ liệu hình ảnh
│   ├── Train/                    # Tập huấn luyện (gồm 7 thư mục con lớp rác)
│   │   ├── cardboard/, compost/, glass/, metal/, paper/, plastic/, trash/
│   └── Test/                     # Tập kiểm thử (gồm 7 thư mục con lớp rác)
│       ├── cardboard/, compost/, glass/, metal/, paper/, plastic/, trash/
│
├── static/                       # Các tài nguyên tĩnh phục vụ Web App
│   ├── css/
│   │   └── style.css             # Thiết kế Glassmorphism & Neon Glow hiệu ứng
│   ├── js/
│   │   └── main.js               # Logic điều khiển, nạp biểu đồ Chart.js, gọi API
│   ├── uploads/                  # Nơi lưu trữ ảnh người dùng tải lên & ảnh Grad-CAM
│   └── images/                   # Đồ thị hiệu năng gốc (VGG16 & ResNet50)
│
├── templates/
│   └── index.html                # Giao diện chính của Web App dạng Single-Page
│
├── app.py                        # Trái tim dự án (Flask Backend, API, Training, Grad-CAM)
├── waste_model.h5                # Trọng số Custom MobileNetV2 sau huấn luyện (nếu có)
├── yolo26n-cls.pt                # Trọng số YOLO26 Classification ImageNet (tự động tải)
├── compare.txt                   # Ghi nhận cải tiến so với bản gốc 2019
├── improve.txt                   # Danh mục các đề xuất cải tiến deep học sâu
├── requirement.txt               # Danh sách thư viện nguyên bản của dự án
├── Waste_Garbage_Collection_improve.ipynb  # Notebook nghiên cứu phát triển mô hình của bạn
└── Hướng dẫn.md                  # Tệp hướng dẫn này (Tên chính xác hiện tại của bạn)
```

---

## 3. Yêu cầu Hệ thống & Chuẩn bị Môi trường

*   **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux.
*   **Trình thông dịch**: Python phiên bản từ **3.8 trở lên** đến **3.11** (khuyên dùng Python 3.10 hoặc 3.9).
*   **Các thư viện phần cứng (Tùy chọn)**: Nếu máy tính có GPU NVIDIA hỗ trợ CUDA, quá trình huấn luyện MobileNetV2 và YOLO26 sẽ diễn ra cực kỳ nhanh (chỉ mất vài giây đến vài phút). Nếu không có GPU, hệ thống sẽ tự động chạy trên CPU mượt mà nhờ kiến trúc siêu nhẹ của cả 2 mô hình.

---

## 4. Hướng dẫn các bước Cài đặt Chi tiết

Bạn hãy thực hiện lần lượt các bước dưới đây bằng cửa sổ terminal (PowerShell trên Windows hoặc Terminal trên macOS/Linux).

### Bước 1: Di chuyển vào thư mục dự án
Mở terminal và di chuyển đến thư mục chứa dự án của bạn:
```powershell
cd "d:\Trường Đại Học Công Thương\Năm 3_HUIT\Học kỳ 6\Học sâu\Thực Hành\Waste-or-Garbage-Classification-Using-Deep-Learning-main"
```

### Bước 2: Khởi tạo môi trường ảo Python (Virtual Environment - Khuyên dùng)
Để tránh xung đột thư viện giữa các dự án khác nhau trên máy tính, hãy tạo một môi trường ảo riêng biệt:
```powershell
# Tạo môi trường ảo có tên là .venv
python -m venv .venv

# Kích hoạt môi trường ảo trên Windows (PowerShell)
.venv\Scripts\Activate.ps1

# (Hoặc kích hoạt trên macOS/Linux)
# source .venv/bin/activate
```
*Lưu ý: Khi môi trường ảo được kích hoạt thành công, bạn sẽ thấy ký hiệu `(.venv)` xuất hiện ở đầu dòng lệnh trong terminal.*

### Bước 3: Nâng cấp pip lên phiên bản mới nhất
```powershell
python -m pip install --upgrade pip
```

### Bước 4: Cài đặt các thư viện lõi
Hãy chạy lệnh cài đặt tất cả các gói thư viện theo đúng danh sách `requirement.txt` của dự án và cài thêm thư viện `ultralytics` phục vụ YOLO26:
```powershell
# Cài đặt các thư viện nguyên bản từ requirement.txt
pip install -r requirement.txt

# Cài đặt thêm thư viện ultralytics và các gói bổ trợ
pip install ultralytics Flask-JSGlue matplotlib requests
```

**Mổ xẻ các thư viện đã cài đặt:**
*   `Flask`: Xây dựng máy chủ Web và các API đầu cuối `/classify`, `/train`, `/train_status`.
*   `tensorflow`: Nền tảng xây dựng mô hình GreenCycle Net, huấn luyện, compile và chạy Grad-CAM.
*   `torch` & `torchvision`: Nền tảng vận hành mạng nơ-ron PyTorch phục vụ YOLO26.
*   `ultralytics`: Thư viện lõi chứa kiến trúc YOLO26 mới nhất để thực hiện suy luận thời gian thực.
*   `opencv-python` & `pillow`: Xử lý, đọc ghi hình ảnh, hỗ trợ đè bản đồ nhiệt Grad-CAM màu sắc.
*   `matplotlib`: Hỗ trợ vẽ bản đồ nhiệt XAI trong luồng fallback.

---

## 5. Hướng dẫn Chạy dự án & Nghiệm thu Giao diện

### Bước 1: Khởi động máy chủ Flask
Tại thư mục gốc của dự án (khi môi trường ảo đã được kích hoạt), chạy lệnh sau:
```powershell
python app.py
```
Khi màn hình terminal xuất hiện thông báo sau, máy chủ đã hoạt động thành công:
```text
>>> Custom Waste Classifier Model loaded successfully!
>>> Pre-trained YOLO26 Classification model loaded successfully!
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Bước 2: Truy cập Giao diện Web
Mở trình duyệt web bất kỳ (Chrome, Edge, Firefox) và truy cập đường dẫn:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

### 🚀 Hướng dẫn Nghiệm thu thực tế trên Web

#### 1. Nhận Diện Rác thải & So sánh Mô hình Kép (Tab 1: Nhận Diện Rác)
*   **Bảng chọn mô hình (Model Selector Card)**: Bạn sẽ thấy hai thẻ Glassmorphism phát sáng tuyệt đẹp.
    *   Nhấp chọn **GreenCycle Net (TensorFlow)**: Thẻ sẽ phát sáng viền neon màu xanh lục bảo.
    *   Nhấp chọn **YOLO26-cls (PyTorch)**: Thẻ sẽ chuyển sang phát sáng viền neon màu vàng hổ phách.
*   **Tải ảnh & Dự đoán**:
    *   Kéo thả hoặc click chọn một hình ảnh rác thải bất kỳ (ví dụ: một chiếc chai nhựa hoặc vỏ hộp giấy) vào vùng nét đứt Drop-zone.
    *   Click nút **Bắt đầu phân loại**.
*   **Đọc kết quả tương tác**:
    *   **Nếu chọn GreenCycle Net**: Bạn sẽ nhận lại kết quả dự đoán kèm theo khung **Tính Minh Bạch AI (Explainable AI - XAI)**. Click nút **Bản Đồ Nhiệt (AI Focus)** để nhìn thấy bản đồ nhiệt Grad-CAM đỏ rực đè lên ảnh gốc (vị trí nắp chai, nhãn mác, cổ chai...) chứng minh AI thực sự học được hình dáng của rác! Click **Ảnh Gốc** để quay lại ảnh cũ.
    *   **Nếu chọn YOLO26-cls**: Kết quả trả về siêu tốc chỉ trong vài mili-giây với tên thuật toán `YOLO26 ImageNet Engine` hoặc `Custom Trained YOLO26 Model` cực kỳ hiện đại.
    *   Đồng thời hiển thị bảng chỉ dẫn tái chế sinh thái (Thời gian phân hủy, mẹo phân loại, sự thật thú vị) cùng biểu đồ phân phối xác suất Chart.js động mượt mà.

#### 2. Huấn Luyện Mô Hình Trực Tiếp (Tab 2: Huấn Luyện Mô Hình)
*   Đường dẫn dữ liệu `DataSets/Train` của bạn cần đã được nạp đầy đủ các ảnh.
*   Thiết lập số Epochs (ví dụ: 10 Epochs) và bấm **Bắt đầu Huấn Luyện**.
*   **Quan sát bảng điều khiển thời gian thực**:
    *   **Biểu đồ tự động vẽ (Chart.js)**: Sẽ tự động vẽ các điểm chấm Accuracy và Loss của bạn trực tiếp sau mỗi epoch kết thúc!
    *   **Console Logs**: Bạn sẽ thấy log in ra liên tục các bước chuẩn bị Generator, Augmentation, và đặc biệt là sự thay đổi Learning Rate của bộ điều phối Cosine Annealing ở Stage 2:
        `[LR Scheduler] Epoch 4: Set Learning Rate = 1.00e-05`
        `[LR Scheduler] Epoch 5: Set Learning Rate = 9.01e-06`
        ... Giảm dần rà phanh mượt mà theo đồ thị hình sin về `1.00e-07` ở Epoch 10!
*   Sau khi kết thúc 100%, hệ thống tự động lưu file `waste_model.h5` đè lên ổ đĩa và tải lại trang, ngay lập tức mô hình tùy chỉnh mới của bạn sẽ được kích hoạt để nhận diện!

#### 3. Phân Tích Kỹ Thuật (Tab 4)
*   Nhấp vào đây để xem bảng so sánh hiệu năng các mạng VGG16, ResNet50.
*   Xem phân khu **"Dấu Ấn Đột Phá Thuật Toán & Toán Học Học Sâu"** trình bày các công thức toán học sắc nét của 4 đột phá giúp bài thuyết trình của bạn đạt điểm tối đa.

---

## 6. Mổ xẻ 4 Đột phá Kỹ thuật & Toán học (Phục vụ thuyết trình)

Để thuyết trình xuất sắc trước hội đồng chấm thi, bạn cần nắm vững bản chất toán học của 4 công nghệ đột phá mà chúng ta đã tích hợp vào dự án:

### Đột phá 1: Khối Chú Ý Kênh Squeeze-and-Excitation (SE Block)
*   **Vấn đề của CNN truyền thống**: Các mạng CNN thông thường trích xuất hàng ngàn kênh đặc trưng nhưng đối xử với mọi kênh là quan trọng như nhau, dẫn đến mô hình bị phân tâm bởi nền nhiễu (thảm cỏ, mặt bàn gỗ).
*   **Nguyên lý SE Block**: Giúp mô hình tự học cách điều chỉnh trọng số (weighting) cho từng kênh đặc trưng, ép nó tập trung vào kênh chứa rác thải và bỏ qua kênh chứa nền nhiễu.
*   **Toán học 3 bước**:
    1.  **Squeeze (Nén)**: Áp dụng Global Average Pooling trên các đặc trưng không gian $H \times W$ của mỗi kênh để tạo ra một vector mô tả kênh toàn cục $\mathbf{z} \in \mathbb{R}^C$:
        $$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W u_c(i, j)$$
    2.  **Excitation (Kích hoạt)**: Đưa vector qua bộ lọc thắt nút cổ chai (Bottleneck) gồm 2 lớp Dense để học mối quan hệ phi tuyến giữa các kênh, sử dụng hàm activation ReLU ($\delta$) và kết thúc bằng Sigmoid ($\sigma$) để đưa trọng số về khoảng $[0, 1]$:
        $$\mathbf{s} = \sigma(\mathbf{W}_2 \delta(\mathbf{W}_1 \mathbf{z}))$$
    3.  **Scale (Tái recalibrate)**: Nhân vector trọng số thu được vào các bản đồ đặc trưng ban đầu để kích hoạt kênh quan trọng và dập tắt kênh nhiễu:
        $$\widetilde{\mathbf{x}}_c = s_c \cdot \mathbf{u}_c$$

### Đột phá 2: Hàm mất mát Focal Loss Đa Lớp (Multi-class Focal Loss)
*   **Vấn đề của Categorical Cross Entropy gốc**: Hàm mất mát thông thường đối xử với mọi bức ảnh như nhau. Trong tập dữ liệu rác, các ảnh bìa carton rất dễ đoán, trong khi chai thủy tinh phản xạ hoặc túi nilon trong suốt rất khó đoán. Mô hình dễ bị áp đảo bởi hàng ngàn mẫu dễ đoán, dẫn đến lười học các đặc trưng khó.
*   **Nguyên lý Focal Loss**: Tự động giảm trọng số mất mát của các mẫu dễ đoán và tăng cường hình phạt đối với các mẫu dự đoán sai (mẫu khó), buộc mô hình dồn toàn lực tối ưu các lớp rác phức tạp.
*   **Toán học**:
    $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
    Trong đó:
    *   $p_t$ là xác suất mô hình dự đoán đúng lớp thực tế.
    *   $(1 - p_t)^\gamma$ là **thành phần điều chế** (modulating factor). Nếu một bức ảnh rất dễ đoán ($p_t \approx 0.99$), thành phần điều chế $(1 - 0.99)^2 \approx 0.0001$ sẽ triệt tiêu gần như hoàn toàn độ lỗi của nó. Ngược lại, nếu ảnh cực khó bị đoán sai ($p_t \approx 0.1$), thành phần điều chế $(1 - 0.1)^2 \approx 0.81$ sẽ giữ nguyên lỗi và phạt cực nặng, buộc mô hình tập trung học lại.
    *   $\gamma = 2.0$ là tham số tập trung (focusing parameter).

### Đột phá 3: Bộ điều phối Learning Rate Cosine Annealing
*   **Vấn đề**: Giữ nguyên Learning Rate (tốc độ học) từ đầu đến cuối hoặc giảm bậc thang thô sơ dễ khiến mô hình bị mắc kẹt ở cực tiểu cục bộ (local minima), không đạt độ chính xác tối ưu.
*   **Nguyên lý**: Giảm tốc độ học từ từ theo chu kỳ nửa đường cong hình sin (Cosine). Ban đầu học nhanh, sau đó "rà phanh" siêu mượt khi tiếp cận điểm tối ưu toàn cục để tránh bị dao động lệch khỏi đáy hội tụ.
*   **Toán học**:
    $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$
    Trong đó:
    *   $\eta_{max}$: Tốc độ học cực đại ban đầu ($10^{-5}$ ở Stage 2).
    *   $\eta_{min}$: Tốc độ học cực tiểu đích đến ($10^{-7}$).
    *   $T_{cur}$: Epoch hiện tại.
    *   $T_{max}$: Tổng số epoch huấn luyện của chu kỳ (7 Epochs).

### Đột phá 4: Bản Đồ Nhiệt Grad-CAM (Explainable AI - XAI)
*   **Vấn đề**: Mô hình mạng nơ-ron sâu là một "hộp đen" (Black-box). Nó kết luận "đây là nhựa" nhưng ta không biết nó nhìn vào cái chai nhựa hay nhìn vào thảm cỏ xanh phía sau.
*   **Nguyên lý**: Đo lường dòng gradient chảy ngược từ lớp dự đoán cao nhất về lớp tích chập cuối cùng (`out_relu`) để vẽ bản đồ kích hoạt các vùng đặc trưng hình học mà mô hình dựa vào để đưa ra kết luận.
*   **Toán học**:
    1.  Tính toán gradient của điểm số lớp dự đoán $y^c$ đối với các bản đồ đặc trưng $A^k$ của lớp tích chập cuối cùng: $\frac{\partial y^c}{\partial A^k}$.
    2.  Tính trọng số tầm quan trọng của kênh đặc trưng $\alpha_k^c$ bằng cách lấy trung bình toàn cục không gian:
        $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
    3.  Tích hợp tuyến tính các bản đồ đặc trưng kèm trọng số và đưa qua hàm ReLU để giữ lại các vùng đóng góp tích cực cho lớp phân loại dự đoán:
        $$L_{Grad-CAM}^c = ReLU\left(\sum_{k} \alpha_k^c A^k\right)$$

---

## 7. Xử lý Sự cố & Câu hỏi thường gặp (FAQs)

### Q1: Gặp lỗi `'Adam' object has no attribute 'lr'` khi bấm Huấn luyện?
*   **Giải thích**: Lỗi này xảy ra trên các phiên bản TensorFlow mới (TF 2.11 trở lên) do cấu trúc lớp Optimizer của Keras được nâng cấp và thay đổi thuộc tính truy cập `.lr` thành `.learning_rate`.
*   **Xử lý**: Lỗi này **đã được chúng tôi vá triệt để** trong tệp `app.py`. Bộ điều phối tự động dò tìm thuộc tính thích ứng `.learning_rate` mới và gọi hàm `.assign(lr)` theo đúng chuẩn TensorFlow 2.x hiện đại nên bạn hoàn toàn yên tâm chạy hệ thống mà không gặp lỗi này nữa.

### Q2: Mô hình YOLO26 mất nhiều thời gian trong lần nhận diện đầu tiên?
*   **Giải thích**: Trong lần đầu tiên bạn bấm phân loại bằng mô hình YOLO26, thư viện `ultralytics` sẽ tự động kết nối internet để tải file trọng số pre-trained chính hãng `yolo26n-cls.pt` từ máy chủ của họ về máy tính của bạn (kích thước siêu nhẹ chỉ khoảng ~6MB).
*   **Xử lý**: Đảm bảo máy tính của bạn có kết nối internet trong lần chạy đầu tiên. Từ lần nhận diện thứ hai trở đi, mô hình đã được lưu cục bộ nên tốc độ xử lý sẽ diễn ra siêu tốc (chỉ khoảng vài mili-giây).

### Q3: Muốn tự huấn luyện mô hình YOLO26 tùy biến trên tập dữ liệu rác thì làm thế nào?
*   Bạn có thể dễ dàng huấn luyện một mô hình YOLO26 tùy biến hoàn toàn trên 7 lớp rác của bạn bằng cách chạy dòng lệnh đơn giản sau trong môi trường ảo Python:
    ```powershell
    python -c "from ultralytics import YOLO; model = YOLO('yolo26n-cls.pt'); model.train(data='./DataSets', epochs=10, imgsz=224)"
    ```
    Sau khi huấn luyện xong, Ultralytics sẽ xuất ra file trọng số tốt nhất `best.pt` nằm trong thư mục `runs/classify/train/weights/best.pt`. Bạn hãy copy tệp `best.pt` này ra thư mục gốc dự án của bạn và đổi tên thành `yolo26_waste.pt`.
    Hệ thống Flask của chúng ta sẽ tự động nhận diện tệp `yolo26_waste.pt` này và kích hoạt mô hình YOLO26 tùy biến cao cấp của riêng bạn thay thế cho mô hình ImageNet gốc!
