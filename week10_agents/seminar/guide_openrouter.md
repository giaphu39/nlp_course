# Hướng Dẫn Sử Dụng OpenRouter Cho Seminar Tuần 10

Tài liệu này hướng dẫn cách thiết lập tài khoản, lấy API Key từ OpenRouter, cách tích hợp vào mã nguồn của bài học, đồng thời giải thích chi tiết về nền tảng OpenRouter, chi phí và các chính sách miễn phí.

---

## 1. Hướng Dẫn Thao Tác Trên Giao Diện OpenRouter (UI)

Để sử dụng OpenRouter trong bài thực hành, bạn thực hiện theo các bước sau:

### Bước 1: Đăng ký / Đăng nhập tài khoản
1. Truy cập vào trang chủ: [https://openrouter.ai/](https://openrouter.ai/)
2. Nhấn nút **Login** ở góc trên cùng bên phải. Bạn có thể đăng nhập nhanh bằng tài khoản **Google**, **GitHub**, hoặc ví Web3.

### Bước 2: Tạo API Key
1. Sau khi đăng nhập, nhấp vào ảnh đại diện của bạn ở góc trên bên phải hoặc truy cập trực tiếp đường dẫn: [https://openrouter.ai/keys](https://openrouter.ai/keys).
2. Nhấn nút **Create Key**.
3. Đặt tên cho Key (ví dụ: `ysda-agent-key`) rồi nhấn **Create**.
4. **Sao chép API Key ngay lập tức** (mã này có tiền tố `sk-or-v1-...`). *Lưu ý: Bạn chỉ có thể nhìn thấy chuỗi Key này một lần duy nhất vì lý do bảo mật.*

### Bước 3: Tìm kiếm Model ID
1. Vào tab **Models** hoặc truy cập [https://openrouter.ai/models](https://openrouter.ai/models).
2. Tìm kiếm mô hình bạn muốn dùng (ví dụ: `x-ai/grok-4.3`, `google/gemini-2.5-flash`, hoặc các mẫu miễn phí như `meta-llama/llama-3.3-70b-instruct:free`).
3. Sao chép chính xác đoạn mã Model ID hiển thị ở tiêu đề mô hình để cấu hình vào code.

---

## 2. Cách Nhúng Thông Tin Vào Code Để Hoạt Động

Trong file notebook [ysda_agents_sem.ipynb](file:///d:/01_Study/03_SelfLearning/NLP/YSDA%20Natural%20Language%20Processing%20course/nlp_course/week10_agents/seminar/ysda_agents_sem.ipynb), OpenRouter được cấu hình tại các Cell thiết lập Model. Bạn hãy thay đổi như sau:

### Cấu hình Provider và Model ID (Cell 5 & Cell 6)
Chuyển đổi `PROVIDER` sang `'OpenRouter'` và điền các tham số tương ứng:

```python
# Thiết lập Provider là OpenRouter
PROVIDER = 'OpenRouter' 
```

Khi chạy Cell tiếp theo, notebook sẽ yêu cầu bạn nhập API Key qua hộp thoại bảo mật `getpass`:

```python
from getpass import getpass

if PROVIDER == 'vLLM':
    KEY = "EMPTY"
    HOST = "127.0.0.1"
    PORT = "8999"
    MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    URL = f"http://{HOST}:{PORT}/v1"
else:
    # Điền Model ID của OpenRouter mà bạn muốn sử dụng ở đây
    MODEL = "x-ai/grok-4.3" # Hoặc "meta-llama/llama-3.3-70b-instruct:free"
    KEY = getpass("Your openrouter api key: ") # Dán mã sk-or-v1-... khi hộp thoại hiện ra
    URL = "https://openrouter.ai/api/v1"
```

### Khởi tạo Client tương thích (Cell 8 & Cell 28)
OpenRouter hỗ trợ giao thức hoàn toàn tương thích với thư viện `openai`. Khi bạn khởi tạo client, thư viện sẽ tự động gọi đến endpoint của OpenRouter:

```python
from openai import OpenAI

client = OpenAI(
  base_url=URL, # Sẽ là 'https://openrouter.ai/api/v1'
  api_key=KEY,  # API Key sk-or-v1-... của bạn
)

# Lưu ý: Luôn nên truyền tham số `max_tokens` khi gọi OpenRouter
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=1000, # Giới hạn token đầu ra để tránh lỗi 402 (Insufficient credits)
)
print(response.choices[0].message.content)
```

---

## 3. Kiến Thức Mở Rộng Về OpenRouter

### 3.1. OpenRouter là gì và dùng để làm gì?
*   **Khái niệm:** OpenRouter là một **API Gateway/Aggregator** (cổng kết nối trung gian) dành cho các Mô hình ngôn ngữ lớn (LLM). Thay vì phải đăng ký tài khoản và tích hợp API riêng lẻ cho từng nhà cung cấp (OpenAI, Anthropic, Google, Meta, Cohere, xAI, v.v.), bạn chỉ cần kết nối tới OpenRouter.
*   **Mục đích sử dụng:**
    *   **Thử nghiệm nhanh:** Chuyển đổi giữa các mô hình khác nhau (như GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) chỉ bằng cách đổi tên tham số `MODEL` trong code, không cần sửa cấu trúc gọi API.
    *   **Tối ưu chi phí:** OpenRouter tự động định tuyến đến nhà cung cấp dịch vụ (host) có chi phí rẻ nhất hoặc tốc độ nhanh nhất tại thời điểm truy vấn.
    *   **Dễ thanh toán:** Chỉ cần nạp tiền vào một tài khoản duy nhất trên OpenRouter thay vì thanh toán riêng lẻ cho nhiều bên.

### 3.2. Chính sách tính phí (Pricing) như thế nào?
*   **Mô hình thanh toán:** Sử dụng phương thức **Pay-as-you-go** (dùng bao nhiêu trả bấy nhiêu). Bạn nạp tiền trước vào ví tài khoản (Credit) bằng thẻ tín dụng (Stripe) hoặc tiền mã hóa (crypto).
*   **Đơn vị tính:** Chi phí được tính trên **1 triệu tokens (1M tokens)** cho cả Prompt (đầu vào) và Completion (đầu ra).
*   **Giá cả:** Giá của từng mô hình được OpenRouter niêm yết công khai và bám sát giá gốc từ các nhà phát triển gốc, đôi khi rẻ hơn nhờ cơ chế gom lượng truy cập lớn (bulk discount) và bộ đệm kết quả (caching).
    *   Ví dụ: Các dòng mô hình nhẹ (như `Gemini Flash` hay `Llama 8B`) chỉ tốn vài cent (0.05$ - 0.1$) cho 1M tokens. Các mô hình cao cấp (như `Grok 4`, `Claude 3.5 Sonnet`) có giá khoảng 3$ - 15$ cho 1M tokens.

### 3.3. Các mô hình Miễn phí (Free Models) và Hạn chế
OpenRouter cung cấp một danh mục riêng các mô hình **hoàn toàn miễn phí** (được ký hiệu hậu tố `:free` trong tên mô hình, ví dụ: `meta-llama/llama-3.3-70b-instruct:free`, `mistralai/mistral-7b-instruct:free`, `google/gemma-2-9b-it:free`).

**Hạn chế của phiên bản/mô hình miễn phí:**
1.  **Tốc độ phản hồi (Rate Limits):** Các mô hình miễn phí bị giới hạn nghiêm ngặt về số lượng request trên phút (RPM) và số lượng token trên phút (TPM). Nếu gọi API liên tục (như trong vòng lặp ReAct Agent), bạn rất dễ gặp lỗi `429 Too Many Requests`.
2.  **Độ trễ cao (Latency):** Do chạy trên các tài nguyên chia sẻ công cộng, thời gian phản hồi (Time-to-first-token) của phiên bản free sẽ chậm hơn nhiều so với phiên bản trả phí trong giờ cao điểm.
3.  **Không có bảo đảm dịch vụ (SLA):** Các mô hình miễn phí có thể gặp tình trạng quá tải hoặc tạm thời ngoại tuyến mà không được thông báo trước.
4.  **Hạn chế quyền tiếp cận:** Bạn không thể sử dụng các mô hình cao cấp hàng đầu (như dòng GPT-4 hay Claude-3.5) nếu số dư tài khoản của bạn bằng 0.

### 3.4. Lỗi thường gặp: Error 402 (Payment Required / Insufficient credits)
*   **Nguyên nhân:** Khi gọi `client.chat.completions.create` mà không chỉ định `max_tokens`, OpenRouter sẽ mặc định giữ chỗ (reserve) số tiền tương ứng với số token đầu ra tối đa của model đó (ví dụ Grok 4.3 là 65,536 tokens). Nếu số dư credits của bạn không đủ chi trả cho 65,536 tokens, bạn sẽ nhận lỗi `402`.
*   **Cách xử lý:** Luôn truyền thêm tham số `max_tokens` (ví dụ `max_tokens=1000` hoặc `max_tokens=2048`) vào hàm gọi API, hoặc đổi sang mô hình miễn phí có đuôi `:free`.

---

## 4. Q&A: Các Câu Hỏi Thường Gặp Khi Thực Hành (Kaggle, GPU, vLLM, MCP)

### Q1: Chạy notebook này trên Kaggle thì có cần bật GPU không?
*   **Trả lời: KHÔNG CẦN GPU** nếu bạn sử dụng API bên ngoài (**Google AI Studio** hoặc **OpenRouter**).
*   **Lý do:** Khi dùng API, toàn bộ công việc tính toán nặng nhất của mô hình ngôn ngữ lớn (LLM inference) đều được thực hiện trên cụm máy chủ đám mây của Google hoặc OpenRouter. Môi trường Kaggle của bạn chỉ đóng vai trò là client gửi request qua giao thức HTTP, chạy các hàm tìm kiếm web, xử lý văn bản và điều phối logic ReAct. Vì vậy, bạn chỉ cần chọn máy ảo **CPU** trên Kaggle, vừa chạy mượt mà vừa **tiết kiệm 30 giờ GPU quota hàng tuần** của Kaggle.
*   **Khi nào mới cần GPU?** Bạn chỉ BẮT BUỘC cần GPU (NVIDIA T4 x2 hoặc P100) khi bạn chọn `PROVIDER = 'vLLM'` để tự tải trọng số và chạy mô hình trực tiếp trên máy ảo Kaggle.

---

### Q2: vLLM là thư viện gì? Tại sao trong code ban đầu lại cài nó? Cần làm những gì để sử dụng?
*   **vLLM là gì?** vLLM là thư viện mã nguồn mở chuyên dụng cho việc **phục vụ (serving) và suy luận (inference) LLM local với hiệu năng cực cao**. Điểm nổi bật nhất của vLLM là công nghệ **PagedAttention** (quản lý bộ nhớ Key-Value cache phân trang tương tự bộ nhớ ảo trong hệ điều hành), giúp tăng tốc độ sinh từ gấp 2 - 4 lần so với thư viện Hugging Face tiêu chuẩn và giảm tối đa lãng phí VRAM.
*   **Tại sao code lại cài vLLM?** Khóa học YSDA cung cấp phương án vLLM để học viên nào có GPU mạnh có thể tự chạy một mô hình local (như `Qwen/Qwen2.5-3B-Instruct`) mà không cần phụ thuộc vào API bên ngoài hay lo lắng về chi phí/hạn mức mạng. vLLM có sẵn endpoint tương thích chuẩn OpenAI (`vllm.entrypoints.openai.api_server`), cho phép code client gọi đến `http://127.0.0.1:8999/v1` hệt như đang gọi OpenAI thật.
*   **Để sử dụng vLLM cần làm những gì?**
    1.  **Bật GPU:** Trên Kaggle vào mục *Settings -> Accelerator -> GPU T4 x2*.
    2.  **Cài đặt:** Chạy `!pip install "vllm>=0.8.5"` (lưu ý gói này rất nặng, mất 3 - 5 phút tải).
    3.  **Khởi động Server:** Khởi chạy một tiến trình background qua `subprocess` hoặc terminal:
        ```bash
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-3B-Instruct \
            --max_model_len 8192 \
            --host 127.0.0.1 \
            --port 8999 \
            --gpu_memory_utilization 0.8
        ```
    4.  **Kết nối trong code:** Đặt `PROVIDER = 'vLLM'`, `URL = "http://127.0.0.1:8999/v1"`, `KEY = "EMPTY"`.
    5.  **Dọn dẹp:** Khi học xong, phải gọi lệnh `vllm_server.terminate()` để giải phóng VRAM cho GPU.
*   **Khuyến nghị:** Đối với bài học Tuần 10 về Agents, bạn **nên dùng Google AI Studio** để tránh rườm rà trong việc cài đặt và quản lý GPU, tập trung vào việc học cách viết Tool, MCP và luồng ReAct.

---

### Q3: Tại sao nên ưu tiên Google AI Studio thay vì OpenRouter cho bài lab này?
*   **Google AI Studio (Gemini 2.5 Flash):**
    *   Cung cấp **1,500 lượt gọi miễn phí mỗi ngày (RPD)** và tự hồi phục 100% sau 24h.
    *   Tốc độ phản hồi cực nhanh, cửa sổ ngữ cảnh rất lớn (1 triệu tokens).
    *   Hỗ trợ tương thích hoàn toàn chuẩn OpenAI SDK qua URL: `https://generativelanguage.googleapis.com/v1beta/openai/`.
    *   Không yêu cầu thẻ tín dụng, không lo cạn credit (lỗi 402) hay bị nghẽn shared pool (lỗi 429).
*   **OpenRouter:** Rất tuyệt vời khi cần trải nghiệm nhiều dòng model khác nhau (Grok, Claude, DeepSeek, LLaMA), nhưng các model `:free` thường bị giới hạn tốc độ và nghẽn mạng trong giờ cao điểm, còn model trả phí thì trừ tiền và đòi hỏi phải cấu hình `max_tokens` cẩn thận.

---

### Q4: Lỗi `McpError: Connection closed` khi chạy `client.connect_to_server('./ysda_tools.py')` là do đâu?
*   **Nguyên nhân:** Lệnh `connect_to_server` khởi chạy một tiến trình con qua giao thức dòng lệnh `stdio`. Nếu file `ysda_tools.py` không tồn tại trong thư mục hiện tại của Kaggle hoặc lệnh chạy ngầm `python` bị crash, đường ống `stdio` sẽ bị đóng ngay tức khắc, dẫn đến lỗi `Connection closed`.
*   **Cách khắc phục triệt để:**
    1.  Dùng `%%writefile ysda_tools.py` để tạo trực tiếp file mã nguồn MCP Server ngay trong notebook trên Kaggle.
    2.  Trong class `MCPClient`, sử dụng `command = sys.executable` thay vì `command = "python"` để tiến trình con dùng chính xác môi trường Python của Kernel hiện hành.
