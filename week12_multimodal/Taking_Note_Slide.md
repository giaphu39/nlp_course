# 📘 Taking Note — Bài giảng Multimodal LLMs (Week 12)

> **Nguồn:** `lecture_multimodal.pdf` — Yandex School of Data Analysis, Asya Fadeeva (30.11.2025)
> **Ngôn ngữ ghi chú:** Tiếng Việt, có bổ sung **tri thức đối chiếu thực tế (practice)** và links tới các bài báo gốc.

---

## 1. Giới thiệu Multimodal Models

**Multimodal LLMs (MLLMs)** là các mô hình cần kết hợp thông tin từ **nhiều modality** khác nhau: hình ảnh, âm thanh, video, văn bản. Mục tiêu chính là thêm **khả năng hiểu (understanding)** để suy luận và trả lời chính xác.

- Cách xây dựng phổ biến: **mở rộng một mô hình nền tảng văn bản (textual foundation model) đã được pre-train mạnh mẽ**.
- Ý tưởng cốt lõi: LLM hoạt động trên **chuỗi token**, ta muốn nhúng các modality khác vào chuỗi token đó. Tiền thân: **Flamingo (2022)** → [arXiv:2204.14198](https://arxiv.org/pdf/2204.14198).

### ⚠️ Phạm vi (Out of scope)
- **Omni models** (vừa hiểu vừa sinh) không nằm trong bài giảng → xem [From Specific-MLLMs to Omni-MLLMs](https://arxiv.org/pdf/2412.11694v3).
- **Audio understanding** cũng không bàn tới. Gợi ý các audio encoder phổ biến:
  - **whisper** — encoder-decoder của OpenAI (2022)
  - **SoundStream** — encoder-decoder với Residual Vector Quantization (2021)

> 💡 **Đối chiếu thực tế:** MLLM "từ văn bản trước" là triết lý của hầu hết các model hiện đại (LLaVA, Qwen-VL, LLaMA-3.2-Vision). Hướng Omni (Qwen3-Omni, GPT-4o) đang là xu hướng hot nhất cuối 2025.

---

## 2. Căn chỉnh Modality (Aligning modalities)

Câu hỏi lớn: **"Làm thế nào để căn chỉnh hai modality khác nhau?"**

### 2.1. Huấn luyện visual encoder tổng quát
- Train ResNet/ViT cho image classification chỉ có **dữ liệu nhãn hạn chế**.
- Một **caption** (mô tả văn bản) cung cấp **nhiều thông tin hơn một nhãn lớp**.
- Ý tưởng: **scale dữ liệu caption** để cải thiện khả năng tổng quát hóa của visual encoder.
  → Paper: [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/pdf/2103.00020)

### 2.2. Dữ liệu ảnh kèm caption
| Dataset | Năm | Đặc điểm |
|---------|-----|----------|
| **MS-COCO** | 2015 | ~330k cặp (ảnh–caption), caption do con người gán nhãn |
| **CLIP WebImageText** | 2021 | ~400M cặp thu thập từ web |
| **LAION-5B** | 2022 | 5 tỷ cặp ảnh–văn bản ([arXiv:2210.08402](https://arxiv.org/pdf/2210.08402)) |

**LAION-5B** được xây dựng:
- Bắt đầu từ **Common Crawl**
- Lấy ảnh + **alt-text** từ HTML
- **Tự động lọc** dựa trên độ tương đồng (similarity)

> 💡 **Đối chiếu thực tế:** LAION-5B là nguồn dữ liệu khổng lồ nhưng có *độ nhiễu* cao (alt-text không chuẩn). Nhiều model hiện đại (Qwen-VL, InternVL) dùng hỗn hợp dữ liệu sạch + web để cân bằng. Vấn đề cấp phép/nội dung nhạy cảm của LAION cũng từng gây tranh cãi (bị gỡ năm 2023).

---

## 3. Contrastive Language-Image Pre-training (CLIP)

**Ý tưởng:** học biểu diễn ảnh và văn bản **cùng nhau trong một không gian embedding chung**.

- Học **image encoder `f_image`** và **text encoder `f_text`**.
- Cặp ảnh–text **khớp nhau** → embedding phải **gần nhau**.
- Cặp ảnh–text **không khớp** → embedding phải **xa nhau**.

**Cách huấn luyện (batch N cặp):**
- Với batch N cặp, xem như bài toán **phân loại N×N** (mỗi ảnh so với N text, mỗi text so với N ảnh).
- **Trung bình cộng hai cross-entropy loss** (image→text và text→image) cho task này.

**Quy mô huấn luyện CLIP:**
- Batch size **32k**, **592 GPU V100** → [CLIP](https://arxiv.org/pdf/2103.00020), [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/pdf/2103.00020)

### CLIP trong visual understanding
- Image tokens cũng có thể **"lookahead"** nhìn trước task (prefix) để cập nhật biểu diễn — ví dụ **PaLIGemma** ([arXiv:2407.07726](https://arxiv.org/abs/2407.07726)) tốt cho fine-tuning.

---

## 4. Visual Language Models (VLMs)

### 4.1. Tiến trình phát triển VLM
- Tham khảo: [Comprehensive Review of MLLMs](https://arxiv.org/pdf/2408.01319)
- Leaderboard: [lmarena.ai/leaderboard/vision](https://lmarena.ai/leaderboard/vision)

**Các model nổi bật (tháng 12/2025):**
| Loại | Model |
|------|-------|
| **Proprietary** | Gemini (3, 2.5 Pro), ChatGPT (4o, 4.5), Claude |
| **Open-source** | Qwen3-VL (Nov 2025), Mistral, Gemma 3 |

### 4.2. Visual Encoder

**So sánh loại encoder:**
| Loại | Huấn luyện | Ví dụ model |
|------|-----------|-------------|
| **Language supervised** | Ngôn ngữ giám sát | OpenAI CLIP, SigLIP |
| **Self-supervised** | Tự giám sát | DINOv2, I-JEPA |
| **Class supervised** | Nhãn lớp | SupViT |

**Kết luận từ nghiên cứu:**
- **Contrastive training** hiệu quả hơn self-supervised methods.
- Khoảng cách lớn nhất nằm ở **các task liên quan OCR** (text-heavy images).
  → [arXiv:2406.16860](https://arxiv.org/pdf/2406.16860), [arXiv:2503.15621](https://arxiv.org/pdf/2503.15621)

**DINOv2 vs CLIP:**
- DINOv2 ([arXiv:2304.07193](https://arxiv.org/pdf/2304.07193)): train trên **cặp ảnh giống nhau**, chỉ tập trung vào **đặc trưng thị giác**, không tốt cho ảnh chứa nhiều chữ.
- CLIP: train với **language supervision** → tốt hơn cho text-heavy.

### 4.3. SigLIP — Sigmoid loss for language-image pre-training
**Đề xuất:** thay **softmax-based approach** bằng **binary classification** trên tập hợp tất cả cặp (all pair combinations).
→ [arXiv:2303.15343](https://arxiv.org/pdf/2303.15343)

**Lợi ích SigLIP so với CLIP:**
- ✅ **Dễ song song hóa** (dễ parallelised)
- ✅ **Ổn định hơn với nhiễu dữ liệu** (data noise)

> 💡 **Đối chiếu thực tế:** SigLIP hiện là visual encoder phổ biến trong nhiều VLM thế hệ mới. Vì loss là binary per-pair, nó không phụ thuộc vào batch size để có *negative samples* tốt như CLIP → hiệu quả hơn khi batch nhỏ.

### 4.4. Encoder Resolution
- Mỗi ảnh được biểu diễn bằng **số token cố định** (vd **256 token** với Gemma 3).

### 4.5. Pan & Scan (tỷ lệ khung hình cực đoan)
- Độ phân giải vuông cố định gây **artifact** khi xử lý ảnh tỷ lệ khung hình không vuông / độ phân giải cao.
- Giải pháp: **chia ảnh thành các crop không chồng lấp bằng nhau**, phủ toàn bộ ảnh, resize về **896×896** để đưa vào encoder.
- → [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786)

> 💡 **Đối chiếu thực tế:** Đây là kỹ thuật "native resolution" dùng trong nhiều VLM (Qwen-VL, LLaMA-3.2-Vision). Bằng cách chia ảnh thành nhiều tile + đặt thêm token "thumbnail" toàn cảnh, model vừa giữ được chi tiết vừa có ngữ cảnh chung. Chi phí là tăng số token mỗi ảnh.

### 4.6. VL Connectors
- Phổ biến nhất: **Linear layer** hoặc **MLP**.
- Giảm số visual token để **inference nhanh hơn**:
  - **Average pooling**
  - **C-Abstractor** (convolution) → [Locality-enhanced projector for multimodal LLM](https://arxiv.org/abs/2312.06742)
  - **BLIP-2: Q-Former** → [arXiv:2301.12597](https://arxiv.org/pdf/2301.12597)

> 💡 **Đối chiếu thực tế:** Q-Former (BLIP-2) là thiết kế cầu nối (bridge) nổi tiếng giúp "khóa" text backbone và chỉ học connector — giảm chi phí train rất nhiều. MLP (kiểu LLaVA) vẫn là chuẩn do đơn giản và hiệu quả.

### 4.7. Textual Backbone (LLM nền)
Tiêu chí chọn LLM cho VLM:
- **Khả năng suy luận (reasoning)**
- **Long context** (đặc biệt cho interleaved)
- **Thế giới tri thức (world knowledge)**
  → [Cambrian-1](https://arxiv.org/pdf/2406.16860)

---

## 5. Visual Instruction Tuning

> Paper gốc: [Visual Instruction Tuning (LLaVA, 2023)](https://arxiv.org/pdf/2304.08485)

- **Ý tưởng:** tạo **vision-language instruction-following data** dùng **text-only ChatGPT** dựa trên dữ liệu MS-COCO.
- Dataset: **158K unique** language-image instruction-following samples.

**Hai bước huấn luyện:**
1. **Tối ưu projection `W`** từ image tokens sang embedding space của LLM trên **500k image captions**.
2. **Fine-tune W và LLM** cùng nhau trên dataset **150k** (conversation nhiều lượt / multi-turn).

### Interleaved Multi-Image Instruction Tuning
- Mở rộng cho nhiều ảnh xen kẽ trong hội thoại.
- → [arXiv:2405.01483](https://arxiv.org/pdf/2405.01483)

> 💡 **Đối chiếu thực tế:** LLaVA khởi nguồn trào lưu instruction tuning cho VLM. Ngày nay các model lớn (Qwen-VL, InternVL) dùng pipeline tương tự: stage-1 học connector trên caption, stage-2 SFT toàn bộ trên hội thoại đa lượt, thậm chí thêm RLHF/DPO để khớp sở thích người dùng.

---

## 6. Benchmarking / Evaluation

**Câu hỏi then chốt khi đánh giá:** *"Ai đang trả lời câu hỏi: LLM hay MLLM?"* — vì model có thể trả lời đúng bằng tri thức văn bản thay vì thực sự hiểu ảnh.

- **MMBench** ([arXiv:2307.06281](https://arxiv.org/pdf/2307.06281)): benchmark đa dạng task, tách bạch việc model dựa vào ngôn ngữ hay thị giác.

> 💡 **Đối chiếu thực tế:** Xu hướng đánh giá hiện đại: dùng **LLM-as-a-judge**, các benchmark chống "shortcut" (ảnh bị xóa/suy biến để kiểm tra model có thực sự nhìn ảnh không), và các leaderboard cập nhật liên tục như OpenCompass. Lưu ý model có thể học thuộc benchmark → cần "test-time" đánh giá chéo.

---

## 7. Video Understanding

### 7.1. Đặc điểm VideoLLM
| Thành phần | Thường dùng |
|-----------|-------------|
| **Video Encoder** | CLIP ViT-L/14 |
| **Số frame nhìn lúc train** | Phụ thuộc dữ liệu & ứng dụng |
| **Adapter** | Linear Layer / Q-Former |
| **Kích thước LLM** | 7B / 13B / 34B |
| **Sử dụng audio?** | Thường là **không** |

→ [arXiv:2408.01319](https://arxiv.org/pdf/2408.01319), [arXiv:2312.17432](https://arxiv.org/pdf/2312.17432)

### 7.2. An Image Grid Can Be a Video
- Có thể tạo **grid ảnh từ các frame video** và xem như một "ảnh lớn" để VLM xử lý.
- → [arXiv:2403.18406](https://arxiv.org/abs/2403.18406)

### 7.3. Temporal & Spatial Pooling (extension LLaVA không tham số)
- **Spatial pooling** → kết quả tốt.
- **Temporal pooling** → suy giảm hiệu năng cho captioning.
- → [Parameter-free LLaVA Extension](https://arxiv.org/pdf/2404.16994)

> 💡 **Đối chiếu thực tế:** "Training-free adaptation" (không cần train thêm) là cách rẻ nhất để biến VLM ảnh thành VLM video: lấy mẫu frame, ghép grid hoặc nối temporal tokens. Các model chuyên video (Qwen2.5-VL, LLaVA-Video) thêm velocity embedding + video encoder riêng để đạt chất lượng cao hơn.

---

## 8. Tổng kết & Tri thức đối chiếu thực tế (bổ sung)

### Bản đồ kiến thức
```
Multimodal LLMs
├── Aligning modalities (contrastive learning, language supervision)
│   ├── CLIP (2021) → SigLIP (2023)
│   └── Datasets: MS-COCO, WebImageText, LAION-5B
├── Visual Language Models (VLMs)
│   ├── Visual encoder: CLIP/SigLIP/DINOv2/SupViT + Pan&Scan
│   ├── VL connector: Linear/MLP, average pooling, C-Abstractor, Q-Former
│   ├── Textual backbone: LLM mạnh về reasoning/long-context/world-knowledge
│   ├── Instruction tuning: LLaVA (158K), Interleaved multi-image
│   └── Evaluation: MMBench, "LLM hay MLLM trả lời?"
└── Video Understanding
    ├── VideoLLM: CLIP ViT-L/14 + adapter + LLM
    └── Training-free: image grid, spatial/temporal pooling
```

### Một số kinh nghiệm practice (đối chiếu thực tế)
1. **Chọn visual encoder** theo task: OCR/text-heavy → ưu tiên **CLIP/SigLIP** (language-supervised); task thuần thị giác/cấu trúc → **DINOv2**.
2. **Resolution** quyết định nhiều: ảnh độ phân giải cao cần kỹ thuật **Pan&Scan / native resolution** (chia tile) để giữ chi tiết, đặc biệt với văn bản nhỏ.
3. **Số visual token** → tốc độ inference: muốn nhanh thì dùng **average pooling / C-Abstractor** để giảm token.
4. **Đánh giá cẩn thận**: đảm bảo model thực sự nhìn ảnh chứ không "trả lời mò" bằng tri thức văn bản (như ví dụ trái sầu riêng trong notebook).
5. **Xu hướng 2025**: Qwen3-VL, PaliGemma, Kimi-VL (reasoning), các model Omni (đầu vào đa modality + đầu ra văn bản/giọng nói).

### Papers tham khảo chính
| Paper | Mã arXiv | Vai trò |
|-------|----------|---------|
| CLIP | 2103.00020 | Contrastive language-image pre-training |
| LAION-5B | 2210.08402 | Dataset 5B |
| SigLIP | 2303.15343 | Sigmoid loss |
| DINOv2 | 2304.07193 | Self-supervised visual encoder |
| LLaVA (Visual Instruction Tuning) | 2304.08485 | Instruction tuning |
| BLIP-2 | 2301.12597 | Q-Former connector |
| Gemma 3 | 2503.19786 | Pan & Scan, encoder resolution |
| MMBench | 2307.06281 | Benchmarking |
| Cambrian-1 | 2406.16860 | Vision-centric MLLMs |
| MM1 | 2403.09611 | Multimodal LLM pre-training |
| Flamingo | 2204.14198 | Gốc rễ interleaved MLLM |
| Interleaved Multi-Image | 2405.01483 | Multi-image instruction tuning |
| Image Grid | 2403.18406 | Video từ grid ảnh |
| Parameter-free LLaVA | 2404.16994 | Spatial/temporal pooling |

---

*Ghi chú soạn từ lecture slides Yandex, có bổ sung góc nhìn thực tế (practice) và link papers gốc.*
