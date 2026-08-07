# 💻 Taking Note — Luồng Code Qwen2.5-Omni (Week 12)

> **Nguồn:** `vlm_practice_qwen25_omni.ipynb`
> **Model:** `Qwen/Qwen2.5-Omni-3B` (hoặc -7B) — model đa modality quy mô nhỏ: nhận **text + audio + image + video** đầu vào, xuất **text + audio** đầu ra.
> **Ngôn ngữ ghi chú:** Tiếng Việt, có bổ sung **kỹ thuật và bài báo liên quan**.

---

## 1. Tổng quan & Bối cảnh

Qwen2.5-Omni là model **"decent (for its size)"** — tốt so với kích thước nhưng *không phải tốt nhất*. Khi chọn model cho ứng dụng thực tế, nên tham khảo leaderboard:

- [lmarena.ai/leaderboard](https://lmarena.ai/leaderboard) — phần **vision** (cả proprietary lẫn open-source)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) — riêng cho mô hình vision+language

### 🎯 Các model phổ biến cuối 2025 (gợi ý từ notebook)
| Model | Điểm mạnh |
|-------|-----------|
| **Qwen3-Omni** family | Phiên bản mới hơn, đa modality |
| **Qwen3-VL** (4B-Instruct) | Tối ưu riêng cho vision |
| **PaliGemma family** | Đa ngôn ngữ rộng |
| **Kimi-VL-A3B-Thinking-2506** (16B) | Vision + reasoning (chuỗi suy luận) |
| **BAGEL-7B-MoT** | Tạo ảnh bằng LLM |

> 📚 **Bài báo liên quan:** Bài giảng slide đề cập Omni models → [From Specific-MLLMs to Omni-MLLMs](https://arxiv.org/pdf/2412.11694v3), [Qwen3-Omni (2509.17765)](https://arxiv.org/pdf/2509.17765). Qwen2.5-Omni có báo riêng: *"Qwen2.5-Omni: End-to-End Omni-MLLM for Understanding and Generation"*.

---

## 2. Cài đặt Môi trường

```python
!pip install accelerate==0.32.1 transformers==4.57.3 bitsandbytes==0.48.2 qwen-omni-utils[decord]==0.0.8

# colab might ask you to restart session after installing this.
%env HF_HUB_DISABLE_XET=1
```

### Giải thích các thư viện & version
| Thư viện | Vai trò |
|----------|---------|
| `accelerate` | Quản lý phân bổ `device_map="auto"`, tăng tốc inference/training |
| `transformers` | Load model `Qwen2_5OmniForConditionalGeneration` + processor |
| `bitsandbytes` | **Quantization 4-bit** (cho model 7B) |
| `qwen-omni-utils[decord]` | Xử lý `process_mm_info` (audio/video), `decord` để đọc video |
| `HF_HUB_DISABLE_XET=1` | Tắt cơ chế tải XET (tránh lỗi tải checkpoint trên Colab) |

> 💡 **Kỹ thuật liên quan:** **Quantization 4-bit** (bitsandbytes) giúp chạy model 7B trên GPU ít VRAM. Đây là kỹ thuật nén tham số về 4-bit (thường dùng `load_in_4bit=True` + NF4 quantization), giảm ~4x bộ nhớ so với FP16, đánh đổi nhẹ về chất lượng. Khái niệm liên quan: precision (FP32/FP16/BF16), `torch_dtype="auto"`.

---

## 3. Load Model & Processor

```python
import torch
import transformers
from qwen_omni_utils import process_mm_info
from IPython.display import display, Audio, Image

MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"  # hoặc -7B với quantization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = transformers.Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
    quantization_config=None if MODEL_NAME.endswith("3B") else transformers.BitsAndBytesConfig(load_in_4bit=True)
)

processor = transformers.Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME, max_pixels=640 * 480)

SYSTEM_MESSAGE = {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],}
```

### Giải thích từng phần
- **`from_pretrained(...)`** — tải checkpoint HuggingFace. `torch_dtype="auto"` dùng dtype phù hợp checkpoint; `device_map="auto"` phân phối layers qua GPU/CPU.
- **Quantization config** chỉ áp dụng cho model **7B** (3B không cần):
  - `MODEL_NAME.endswith("3B")` → `None` (không quantize)
  - Ngược lại → `BitsAndBytesConfig(load_in_4bit=True)`
- **Processor** = tokenizer + image processor + audio processing. `max_pixels=640*480` giới hạn **memory** (ảnh được resize về tối đa này).
- **`SYSTEM_MESSAGE`** — system prompt định nghĩa vai trò "virtual human" của Qwen.

> 💡 **Kỹ thuật liên quan:** Trong kiến trúc Omni, model có **Thinker** (xử lý và sinh văn bản) và **Talker** (chuyển văn bản → giọng nói). Việc tách 2 module giúp điều chỉnh độc lập. Xem thêm phần 7 dưới đây.

---

## 4. LUỒNG XỬ LÝ CHUNG (Input → Output)

Mọi ví dụ trong notebook đều theo cùng một luồng chuẩn:

```
1. Tạo conversations (danh sách role + content đa modality)
2. processor.apply_chat_template(...)  → text prompt
3. process_mm_info(conversations)      → (audios, images, videos)
4. processor(text=..., audio=..., images=..., videos=...) → inputs tensor
5. model.generate(**inputs, speaker=..., thinker_do_sample=..., talker_do_sample=...)
6. decode + display (text và/hoặc audio)
```

```python
text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                   padding=True, use_audio_in_video=True).to(model.device)

text_ids, audio = model.generate(**inputs, speaker=speaker,
                                 use_audio_in_video=True,
                                 thinker_do_sample=False, talker_do_sample=True)
text_ids = text_ids[:, inputs['input_ids'].shape[1]:]  # remove prompt
text, = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

### Giải thích các bước
- **`apply_chat_template`** — biến list các message thành chuỗi prompt theo template của model (thêm `add_generation_prompt=True` để thêm phần mở đầu câu trả lời).
- **`process_mm_info`** — đọc/tiền xử lý các file ảnh, audio, video thành dạng model dùng được.
- **`processor(...)`** — encode toàn bộ về tensor (`return_tensors="pt"`).
- **`model.generate(...)`**:
  - `speaker="Chelsie"` (hoặc `"Ethan"` cho giọng nam) — chọn giọng, đặc trưng Qwen2.5-Omni.
  - `thinker_do_sample=False` — **greedy decoding** cho phần suy luận (ổn định, xác định).
  - `talker_do_sample=True` — **sampling** cho phần tạo giọng nói (tự nhiên hơn).
  - Trả về **`text_ids`** (token văn bản) và **`audio`** (tensor âm thanh).
- **Bỏ phần prompt:** `text_ids[:, inputs['input_ids'].shape[1]:]` — chỉ giữ token sinh ra, loại token đầu vào.
- **`batch_decode`** — giải mã token → chuỗi văn bản cuối cùng.

> 💡 **Kỹ thuật liên quan:** `thinker_do_sample`/`talker_do_sample` minh họa khái niệm **decoding strategies** — greedy decoding (lấy token xác suất cao nhất) vs sampling (lấy token ngẫu nhiên theo phân phối, có thể kèm temperature/top-p) để tăng độ đa dạng. Với suy luận cần chính xác dùng greedy; với sinh sáng tạo/giọng nói dùng sampling.

---

## 5. Ví Dụ Cụ Thể

### 5.1. Image Understanding (QA trên ảnh)
```python
conversations = [SYSTEM_MESSAGE, {"role": "user", "content": [
    {"type": "image", "image": "./img.jpg"},
    {"type": "text", "text": "What on Earth is this?"}]}]
```
- Ảnh được **resize** theo `max_pixels` đã đặt.
- Model trả text + audio (giọng nói).
- ⚠️ **Lưu ý từ notebook:** *"if the model says it's a durian, don't trust it!"* → model có thể nhầm (hallucination).

### 5.2. Audio Input & Output (transcribe đa ngôn ngữ)
```python
conversations = [SYSTEM_MESSAGE, {"role": "user", "content": [
    {"type": "text", "text": "Transcribe the spoken phrase in English, Spanish and French."},
    {"type": "audio", "audio": "sound.wav"}]}]
```
- Model phiên âm (transcribe) audio sang **3 ngôn ngữ** và đọc lại bằng giọng nói.

### 5.3. Multiple Images (so sánh 2 ảnh)
```python
conversations = [SYSTEM_MESSAGE, {"role": "user", "content": [
    {"type": "image", "image": "./img_A.jpg"},
    {"type": "image", "image": "./img_B.jpg"},
    {"type": "text", "text": "Which one is larger?"}]}]
```
- Hai ảnh đưa cùng lúc, model suy luận so sánh.

### 5.4. Playground (ảnh tự upload)
```python
from google.colab import files
user_prompt = "Find the length of the longest side in the triangle."
uploaded = files.upload("./uploaded/")
image_path = next(iter(uploaded.keys()))
```
- Cho phép upload ảnh riêng và hỏi model (vd giải toán hình học).
- Notebook **cố định 1 ảnh** — muốn nhiều ảnh thì theo format ví dụ 5.3.

---

## 6. Kỹ thuật Tiết kiệm Tài nguyên

```python
model.disable_talker()  # tắt speech module để tiết kiệm GPU time/memory
# ⚠️ Các cell ở trên sẽ KHÔNG hoạt động khi talker bị tắt
```

- **`disable_talker()`** tắt module tạo giọng nói → chỉ xuất text, tiết kiệm VRAM/GPU cho các query khó hơn.
- Nhưng các cell trước yêu cầu đầu ra audio sẽ lỗi nếu talker tắt.

> 💡 **Kỹ thuật liên quan:** Đây là minh họa cho việc **modular design** trong Omni model — chỉ kích hoạt những module cần thiết. Tương tự việc tắt audio encoder khi không cần audio input để giảm chi phí.

---

## 7. Kiến trúc Qwen2.5-Omni (bổ sung từ bài báo)

Qwen2.5-Omni áp dụng kiến trúc **Thinker-Talker** — một trong những thiết kế Omni nổi bật:

- **Thinker**: LLM nền chịu trách nhiệm hiểu toàn bộ modality (text, ảnh, video, audio) và **sinh văn bản**.
- **Talker**: module chuyên biệt nhận đầu ra từ Thinker và **tạo giọng nói (TTS)** real-time.
- Nhờ tách riêng, Talker có thể **streaming** (phát giọng nói trong khi Thinker vẫn đang suy luận) → giảm latency.

**Luồng xử lý đa modality:**
```
Image/Video → ViT (visual encoder) ─┐
Audio → audio encoder (kèm Q-Former) ─┼─→ Thinker (LLM) → text tokens / audio tokens
Text → tokenizer ────────────────────┘                   └→ Talker → giọng nói
```

> 📚 **Bài báo:** *Qwen2.5-Omni: End-to-End Omni-MLLM for Understanding and Generation* (Alibaba Group). Kiến trúc tương tự đã được mở rộng trong **Qwen3-Omni** ([arXiv:2509.17765](https://arxiv.org/pdf/2509.17765)).

---

## 8. Bổ sung Kỹ thuật & Bài báo Liên quan

### 8.1. Các kỹ thuật nền tảng được minh họa trong notebook
1. **Vision-Language connector** — bridge ảnh ↔ LLM. Trong notebook, processor/model xử lý tự động. Khái niệm gốc: **BLIP-2 Q-Former** ([arXiv:2301.12597](https://arxiv.org/pdf/2301.12597)), **LLaVA MLP projector** ([arXiv:2304.08485](https://arxiv.org/pdf/2304.08485)).
2. **Instruction following / Chat template** — chuẩn hóa hội thoại đa modality thành prompt (hệ thống chat template của HuggingFace).
3. **Quantization (4-bit)** — `BitsAndBytesConfig` để giảm VRAM (cần cho model 7B).
4. **Decoding strategies** — `thinker_do_sample` vs `talker_do_sample` (greedy vs sampling).
5. **Multi-image & Interleaved prompting** — đưa nhiều ảnh cùng ngữ cảnh văn bản. Liên quan [Interleaved Multi-Image Instruction Tuning](https://arxiv.org/pdf/2405.01483).

### 8.2. Bảng tóm tắt bài báo liên quan
| Kỹ thuật / Model | Paper / Tài liệu | Ghi chú |
|------------------|------------------|---------|
| Qwen2.5-Omni | [HF model card](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | Thinker-Talker, đa modality |
| Qwen3-Omni | [arXiv:2509.17765](https://arxiv.org/pdf/2509.17765) | Thế hệ Omni mới |
| From Specific-MLLMs to Omni-MLLMs | [arXiv:2412.11694](https://arxiv.org/pdf/2412.11694v3) | Định nghĩa Omni models |
| BLIP-2 (Q-Former) | [arXiv:2301.12597](https://arxiv.org/pdf/2301.12597) | Vision-language connector |
| LLaVA | [arXiv:2304.08485](https://arxiv.org/pdf/2304.08485) | Visual instruction tuning, MLP projector |
| Interleaved Multi-Image | [arXiv:2405.01483](https://arxiv.org/pdf/2405.01483) | Nhiều ảnh xen kẽ |
| Gemma 3 Pan&Scan | [Gemma 3 Tech Report](https://arxiv.org/pdf/2503.19786) | Xử lý ảnh độ phân giải cao |
| PaliGemma | [arXiv:2407.07726](https://arxiv.org/abs/2407.07726) | Encoder "lookahead", fine-tuning |
| Kimi-VL | [HF](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506) | Vision + reasoning |
| BAGEL-7B-MoT | [HF](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) | Image generation với LLM |

### 8.3. Lưu ý thực hành (practice)
1. **Chọn model theo use-case**: đa modality tổng hợp → Qwen-Omni; ưu tiên vision → Qwen3-VL; cần reasoning → Kimi-VL; cần đa ngôn ngữ → PaliGemma.
2. **Giảm VRAM**: dùng 3B thay 7B, hoặc 4-bit quantization, hoặc `disable_talker()`, hoặc giảm `max_pixels`.
3. **Số token ảnh** ảnh hưởng bộ nhớ: `max_pixels` giới hạn kích thước ảnh → cân nhắc giữa chi tiết và memory.
4. **Hallucination**: model có thể mô tả, đặt tên sai đồ vật (vd gọi trái cây là "durian"). Luôn kiểm chứng kết quả với input thực tế.
5. **Streaming / latency**: kiến trúc Thinker-Talker cho phép phát giọng nói sớm — hữu ích cho ứng dụng realtime.

---

*Ghi chú soạn từ notebook thực hành Yandex, có bổ sung kỹ thuật và bài báo liên quan.*
