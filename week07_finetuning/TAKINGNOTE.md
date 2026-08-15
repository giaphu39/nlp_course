# 📘 Week 7: Fine-tuning (PEFT & RLHF) - Tổng hợp Kiến thức

## 📋 Tổng quan Seminar

Seminar này tập trung vào **Parameter-Efficient Fine-Tuning (PEFT)** và **Reinforcement Learning from Human Feedback (RLHF)** - hai kỹ thuật quan trọng để fine-tune các Large Language Models (LLMs) với tài nguyên hạn chế.

---

## 1. 🔧 4-bit Quantization (BitsAndBytes)

### Kỹ thuật
- **Mục tiêu:** Giảm dung lượng bộ nhớ của model weights từ 32-bit float xuống 4-bit integer
- **Công cụ:** `transformers.BitsAndBytesConfig`

### Chi tiết kỹ thuật
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 - chuẩn nén tốt nhất cho LLM
    bnb_4bit_use_double_quant=True,       # Lượng tử hóa kép (giảm xuống ~3.6 bit)
    bnb_4bit_compute_dtype=torch.float16, # Kiểu tính toán để tiết kiệm VRAM
    llm_int8_enable_fp32_cpu_offload=True # Offload CPU khi hết VRAM
)
```

### Biến đổi dữ liệu
| Đầu vào | → | Đầu ra |
|---------|---|--------|
| Model weights 32-bit float (FP32) | → | 4-bit NormalFloat (NF4) |
| ~16GB cho model 7B | → | ~3.5GB cho model 7B |
| 1 tham số = 32 bit | → | 1 tham số ≈ 3.6-4 bit |

### Khi nào áp dụng
- ✅ Khi VRAM GPU hạn chế (< 8GB)
- ✅ Khi muốn chạy model 7B+ trên GPU consumer
- ✅ Khi chỉ cần fine-tune adapter (không cần update full weights)

### Lợi ích
- 🟢 Giảm dung lượng model xuống ~4x lần
- 🟢 Có thể chạy model 7B trên GPU 4-6GB VRAM
- 🟢 Kết hợp tốt với LoRA để fine-tune hiệu quả

### Tác hại / Hạn chế
- 🔴 Mất độ chính xác nhẹ do quantization noise
- 🔴 Chậm hơn FP16/FP32 do cần dequantize khi tính toán
- 🔴 Không thể train full model weights (chỉ train adapter)

---

## 2. 🔄 Gradient Checkpointing

### Kỹ thuật
- **Mục tiêu:** Giảm VRAM bằng cách không lưu toàn bộ activations, thay vào đó tính lại khi backward
- **Công thức:** Lưu một subset nhỏ activations, tính lại phần còn lại trong backward pass

### Biến đổi dữ liệu
| Đầu vào | → | Đầu ra |
|---------|---|--------|
| Forward pass: lưu mọi activation | → | Forward pass: chỉ lưu checkpoint activations |
| Backward pass: dùng activations đã lưu | → | Backward pass: tính lại activations từ checkpoints |

### Khi nào áp dụng
- ✅ Khi VRAM không đủ cho batch size mong muốn
- ✅ Khi training model lớn (> 1B parameters)
- ✅ Luôn bật khi fine-tune với quantization

### Lợi ích
- 🟢 Giảm VRAM ~30-50% cho activations
- 🟢 Cho phép batch size lớn hơn
- 🟢 Dễ implement (chỉ cần 1 dòng code)

### Tác hại / Hạn chế
- 🔴 Tăng thời gian training ~20-30% do tính lại activations
- 🔴 Cần tắt khi inference (model.generate) vì không cần backward

---

## 3. 🎯 Prompt Tuning (Soft Prompts)

### Kỹ thuật
- **Mục tiêu:** Thay vì fine-tune toàn bộ model, thêm các token embeddings có thể học được vào đầu input
- **Paper:** [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
- **Số tham số trainable:** Chỉ `num_prompts × embedding_dim` (ví dụ: 16 × 4096 = ~65K params)

### Kiến trúc
```
Input: [PAD, PAD, ..., PAD, "A", "quick", "brown", "fox"]
         ↑ learnable prompts (16 tokens)  ↑ original token embeddings
         self.learnable_prompts            self.original_word_embeddings
```

### Biến đổi dữ liệu
| Đầu vào | → | Đầu ra |
|---------|---|--------|
| Token IDs: `[PAD]*16 + [original_tokens]` | → | Embeddings: `[learnable_prompts] + [original_embeddings]` |
| 16 PAD tokens | → | 16 learnable prompt vectors (được tối ưu) |
| Gốc: model chỉ dùng embeddings có sẵn | → | Thay thế embedding layer bằng wrapper có prompts |

### Khi nào áp dụng
- ✅ Khi cần điều chỉnh hành vi model (ví dụ: nói sự thật về con cáo)
- ✅ Khi tài nguyên cực kỳ hạn chế (chỉ train ~65K params)
- ✅ Khi task đơn giản, chỉ cần thay đổi nhỏ trong output

### Lợi ích
- 🟢 Cực kỳ nhẹ: chỉ train vài nghìn đến vài trăm nghìn tham số
- 🟢 Không sửa đổi model weights gốc
- 🟢 Dễ dàng chuyển đổi giữa các task (chỉ cần đổi prompts)

### Tác hại / Hạn chế
- 🔴 Performance thấp hơn LoRA trên task phức tạp
- 🔴 Chỉ ảnh hưởng đến phần đầu của sequence
- 🔴 Cần prepend PAD tokens, phải điều chỉnh attention mask

---

## 4. 📚 PEFT Library (HuggingFace)

### Kỹ thuật
- **Mục tiêu:** Cung cấp interface thống nhất cho các phương pháp PEFT
- **Công cụ:** `peft.PromptTuningConfig`, `peft.get_peft_model`, `peft.LoraConfig`

### Cách hoạt động
```python
peft_config = peft.PromptTuningConfig(
    task_type=peft.TaskType.CAUSAL_LM, 
    num_virtual_tokens=16
)
model = peft.get_peft_model(model, peft_config)
```

### Biến đổi dữ liệu
| Đầu vào | → | Đầu ra |
|---------|---|--------|
| Model gốc (full weights) | → | Model wrapped với PEFT adapters |
| Forward pass thông thường | → | Forward pass + adapter injection tự động |
| Không cần prepad PAD tokens | → | PEFT tự động xử lý virtual tokens |

### Khi nào áp dụng
- ✅ Khi muốn dùng PEFT methods mà không implement thủ công
- ✅ Khi cần chuyển đổi nhanh giữa các phương pháp (Prompt Tuning ↔ LoRA)
- ✅ Khi muốn tận dụng các optimization có sẵn

### Lợi ích
- 🟢 Interface thống nhất cho nhiều PEFT methods
- 🟢 Tự động xử lý virtual tokens, attention masks
- 🟢 Tích hợp tốt với Transformers, Trainer, TRL
- 🟢 Hỗ trợ quantized models (4-bit, 8-bit)

### Tác hại / Hạn chế
- 🔴 Che giấu implementation details (khó debug)
- 🔴 Phụ thuộc vào version của thư viện
- 🔴 Một số tùy chỉnh nâng cao khó thực hiện

---

## 5. 📉 LoRA (Low-Rank Adaptation)

### Kỹ thuật
- **Mục tiêu:** Thêm adapter ma trận hạng thấp song song với các linear layers, chỉ train adapter
- **Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
- **Công thức:** `h = Wx + BAx` với `W ∈ ℝ^(d×k)`, `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, `r << min(d,k)`

### Kiến trúc
```
         ┌─────────────┐
x ──────▶│  W (frozen) │──────▶ h = Wx + BAx
         └─────────────┘
                +
         ┌─────────────┐
         │  A (train)  │──▶ B (train) ──▶
         └─────────────┘
         
         rank r = 8 (thường)
         Số tham số adapter = 2 × d × r × k (rất nhỏ so với W)
```

### Biến đổi dữ liệu
| Đầu vào | → | Đầu ra |
|---------|---|--------|
| `W ∈ ℝ^(d×k)` (frozen, hàng triệu params) | → | `W` giữ nguyên, thêm `A∈ℝ^(k×r)`, `B∈ℝ^(r×d)` |
| `y = Wx` | → | `y = Wx + BAx` |
| Attention: Q, K, V projections | → | Q, K, V + LoRA adapters (hạng r=8) |
| MLP: up, gate, down projections | → | Có thể thêm adapter vào MLP layers |

### Khi nào áp dụng
- ✅ Task phức tạp cần nhiều capacity hơn prompt tuning
- ✅ Khi muốn fine-tune attention và/hoặc FFN layers
- ✅ Khi kết hợp với 4-bit quantization (QLoRA)
- ✅ Khi cần fine-tune model 7B+ trên GPU consumer

### Lợi ích
- 🟢 Số tham số trainable rất nhỏ (0.1-1% của model)
- 🟢 Không tăng latency đáng kể khi inference (có thể merge adapters vào W)
- 🟢 Performance gần bằng full fine-tuning trên nhiều task
- 🟢 Có thể chuyển đổi task nhanh (chỉ cần swap adapters)
- 🟢 Kết hợp tốt với quantization (QLoRA)

### Tác hại / Hạn chế
- 🔴 Cần chọn rank r phù hợp (quá thấp → underfitting, quá cao → lãng phí)
- 🔴 Chỉ áp dụng cho linear layers, không áp dụng cho embeddings
- 🔴 Hơi chậm hơn so với prompt tuning (nhiều params hơn)
- 🔴 Với rank rất thấp, có thể không capture được task-specific knowledge

### So sánh các target layers
| Layer | Tác động | Khi nào nên dùng |
|-------|----------|------------------|
| `self_attn.q_proj` | Ảnh hưởng đến attention queries | ✅ Luôn nên dùng |
| `self_attn.k_proj` | Ảnh hưởng đến attention keys | ✅ Luôn nên dùng |
| `self_attn.v_proj` | Ảnh hưởng đến attention values | ✅ Luôn nên dùng |
| `self_attn.o_proj` | Ảnh hưởng đến output projection | ✅ Có thể dùng thêm |
| `mlp.up_proj` | FFN up projection | ✅ Task phức tạp |
| `mlp.gate_proj` | FFN gate (Gated activation) | ✅ Task phức tạp |
| `mlp.down_proj` | FFN down projection | ✅ Task phức tạp |
| `lm_head` | Output LM head | ✅ Task đặc thù (code generation) |

---

## 6. 🏆 RLHF (Reinforcement Learning from Human Feedback)

### Tổng quan Pipeline

```
Stage 1: Supervised Fine-Tuning (SFT)
┌─────────────────────────────────────────────────────┐
│ Dữ liệu: (prompt, demonstration) pairs              │
│ Mô hình: LM được fine-tune supervised trên demos    │
│ Mục tiêu: Học format và style của output mong muốn  │
└─────────────────────────────────────────────────────┘

Stage 2: Reward Modeling
┌─────────────────────────────────────────────────────┐
│ Dữ liệu: (chosen, rejected) pairs từ human feedback │
│ Mô hình: Reward model (thường là BERT-like)         │
│ Loss: -log(σ(r_chosen - r_rejected))               │
│ Mục tiêu: Học cách scoring outputs theo preference  │
└─────────────────────────────────────────────────────┘

Stage 3: PPO Fine-tuning
┌─────────────────────────────────────────────────────┐
│ 1. Rollout: LM generate responses từ queries        │
│ 2. Evaluation: Reward model score các responses      │
│ 3. Update: PPO update LM để maximize reward          │
│ Mục tiêu: LM tự tạo output được reward cao           │
└─────────────────────────────────────────────────────┘
```

### 6.1 🎯 Reward Modeling

#### Kỹ thuật
- **Mục tiêu:** Huấn luyện model phân loại để predict human preferences
- **Model thường dùng:** DistilBERT, BERT, RoBERTa (sequence classification)
- **Loss function:** Pairwise ranking loss từ InstructGPT paper

#### Công thức Loss
```
Loss = -log(σ(r(x, y_w) - r(x, y_l)))
```
- `r(x, y_w)`: reward cho chosen response
- `r(x, y_l)`: reward cho rejected response
- `σ`: sigmoid function

#### Biến đổi dữ liệu
| Đầu vào | → | Đầu ra |
|---------|---|--------|
| Chosen text + Rejected text pairs | → | Reward scores (scalar) |
| Human preference labels | → | Pairwise ranking loss |
| Token IDs (chosen & rejected) | → | Logits → scalar reward |

#### Khi nào áp dụng
- ✅ Khi có human preference data (hoặc có thể synthetic)
- ✅ Khi muốn align model với human values
- ✅ Khi cần controlled generation (positive/negative reviews)

#### Lợi ích
- 🟢 Có thể học complex preferences từ human feedback
- 🟢 Reward model có thể generalize đến unseen responses
- 🟢 Có thể dùng reward-guided sampling (best-of-N)

#### Tác hại / Hạn chế
- 🔴 Cần nhiều human feedback data (hoặc synthetic)
- 🔴 Reward model có thể bị reward hacking
- 🔴 Quality của RLHF phụ thuộc nhiều vào reward model

### 6.2 🔄 PPO (Proximal Policy Optimization) Fine-tuning

#### Kỹ thuật
- **Mục tiêu:** Fine-tune LM để maximize reward từ reward model
- **Thuật toán:** PPO clip từ [OpenAI paper](https://arxiv.org/abs/1707.06347)
- **Công cụ:** `trl.PPOTrainer`, `trl.AutoModelForCausalLMWithValueHead`

#### 3 Stages trong mỗi PPO step
```
1. Rollout: 
   Query → LM → Generate response
   
2. Evaluation:
   Response → Reward Model → Reward score
   
3. Update:
   PPO loss = -E[ min(ratio × A, clip(ratio, 1-ε, 1+ε) × A) ]
   - ratio = π_new(a|s) / π_old(a|s)
   - A = advantage (reward - baseline)
   - ε = clip threshold (thường 0.2)
```

#### Biến đổi dữ liệu
| Đầu vào | → | Đầu ra |
|---------|---|--------|
| Query tokens (partial review) | → | Full response (LM generate) |
| Full response (text) | → | Reward score (scalar) |
| LM logits + Reward | → | PPO loss → Gradient update |
| Model weights (frozen) | → | LoRA adapters (updated) |

#### Khi nào áp dụng
- ✅ Khi muốn optimize một metric cụ thể (sentiment, length, toxicity)
- ✅ Khi supervised fine-tuning không đủ (cần exploration)
- ✅ Khi muốn align model với human preferences

#### Lợi ích
- 🟢 LM tự generate responses, không cần supervised data
- 🟢 Có thể optimize complex objectives (reward = f(text))
- 🟢 KL penalty giữ model gần với original (tránh catastrophic forgetting)
- 🟢 Có thể kết hợp với LoRA để fine-tune hiệu quả

#### Tác hại / Hạn chế
- 🔴 Reward hacking: LM có thể "cheat" reward function
- 🔴 Cần tuning nhiều hyperparameters (learning rate, KL coefficient, clip range)
- 🔴 Training unstable nếu reward model không tốt
- 🔴 Tốn thời gian: mỗi step cần generate + evaluate
- 🔴 Cần nhiều VRAM (model + value head + reference model cho KL)

---

## 7. 📊 So sánh các phương pháp

### Bảng so sánh tổng quan

| Phương pháp | Số params trainable | VRAM yêu cầu | Performance | Speed | Implement |
|-------------|:-------------------:|:------------:|:-----------:|:-----:|:---------:|
| **Full Fine-tuning** | 100% (7B) | ~28GB+ (FP32) | ★★★★★ | ★★ | Dễ |
| **Prompt Tuning** | ~0.001% (65K) | ~4GB | ★★★ | ★★★★★ | Trung bình |
| **LoRA (r=8)** | ~0.1% (7M) | ~6GB | ★★★★ | ★★★★ | Trung bình |
| **QLoRA (4-bit + LoRA)** | ~0.1% (7M) | ~4GB | ★★★★ | ★★★ | Khó |
| **RLHF + LoRA** | ~0.1% (7M) | ~8GB | ★★★★ | ★★ | Rất khó |

### Khi nào chọn phương pháp nào

| Tình huống | Phương pháp khuyến nghị | Lý do |
|------------|------------------------|-------|
| VRAM < 4GB | Prompt Tuning hoặc QLoRA | Ít params nhất, quantization giảm VRAM |
| VRAM 4-8GB | QLoRA + LoRA (r=8) | Cân bằng performance và memory |
| VRAM > 8GB | LoRA (r=16-32) | Performance cao hơn |
| Task đơn giản (1 câu) | Prompt Tuning | Đủ cho task đơn giản |
| Task phức tạp (code, reasoning) | LoRA/QLoRA | Cần nhiều capacity hơn |
| Alignment với human preference | RLHF + LoRA | Cần reward optimization |
| Controlled generation | RLHF (PPO) | Có thể optimize any metric |

---

## 8. 🧠 Luồng code tổng thể

### Seminar Flow (PEFT focus)

```
1. LOAD MODEL
   └── 4-bit quantization (BitsAndBytes)
   └── Gradient checkpointing
   └── Freeze all weights
   
2. PROMPT TUNING (Manual)
   └── WordEmbeddingsWithLearnedPrompts class
   └── Replace embedding layer
   └── Train only learnable prompts
   └── Task: Make model say truth about fox
   
3. PEFT LIBRARY
   └── peft.PromptTuningConfig
   └── peft.get_peft_model
   └── Same task but with library
   
4. LORA IMPLEMENTATION
   └── LoRALayer class (manual)
   └── adapter_A, adapter_B initialization
   └── Apply to Q/K/V projections
   └── Test gradients
   
5. TRAINING WITH LORA
   └── Dataset: codeparrot-clean (Python code)
   └── Trainer: transformers.Trainer
   └── Generate code samples
   └── Compare before/after
```

### Homework Flow (RLHF focus)

```
1. LOAD MODELS
   └── Main model: GPT2-IMDB (causal LM)
   └── Reward model: DistilBERT (sequence classification)
   
2. STAGE 1: REWARD MODEL
   └── IMDBPairwiseDataset (chosen/rejected pairs)
   └── trl.RewardTrainer
   └── Pairwise ranking loss
   └── Evaluate accuracy on test set
   
3. REWARD-GUIDED GENERATION
   └── Generate N=16 samples
   └── Select highest reward
   └── Best-of-N sampling
   
4. STAGE 2: PPO FINE-TUNING
   └── AutoModelForCausalLMWithValueHead
   └── LoRA config (r=32)
   └── PPOTrainer
   └── Rollout → Evaluate → Update
   └── KL penalty to stay close to original
```

---

## 9. ⚠️ Common Pitfalls & Tips

### Quantization
- ⚠️ `load_in_4bit=True` deprecated → dùng `quantization_config=bnb_config`
- ⚠️ Cần tắt gradient checkpointing khi inference (`model.generate`)
- ⚠️ `enable_input_require_grads()` cần thiết cho gradient checkpointing

### Prompt Tuning
- ⚠️ Phải prepad PAD tokens và điều chỉnh attention mask
- ⚠️ Loss tính từ `logits[:, num_prompts:-1]` (bỏ qua prompt tokens)
- ⚠️ Không dùng item assignment (`tensor[i] = x`) vì không autograd-friendly

### LoRA
- ⚠️ Initialization: A = kaiming_uniform, B = zeros (để BA=0 ban đầu)
- ⚠️ Cần test gradient trước khi train full pipeline
- ⚠️ `model.config.use_cache = False` khi train, True khi inference

### RLHF
- ⚠️ Reward model phải ở eval mode (disable dropout) khi PPO
- ⚠️ Cần inspect data thủ công sau mỗi stage
- ⚠️ Reward hacking: KL penalty rất quan trọng
- ⚠️ Nếu reward model không tốt → RLHF sẽ fail

---

## 10. 📚 Tài liệu tham khảo

### Papers
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
- [Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [Prefix Tuning](https://arxiv.org/abs/2101.00190)
- [InstructGPT (RLHF)](https://arxiv.org/pdf/2203.02155.pdf)
- [PPO](https://arxiv.org/abs/1707.06347)
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)
- [Gradient Checkpointing](https://arxiv.org/abs/1604.06174)
- [Post-training Quantization](https://arxiv.org/abs/2208.07339)

### Libraries
- [PEFT](https://huggingface.co/docs/peft/index)
- [TRL](https://huggingface.co/docs/trl)
- [BitsAndBytes](https://huggingface.co/docs/transformers/main_classes/quantization)
- [Accelerate](https://huggingface.co/docs/accelerate/package_reference/big_modeling)

### Video
- [EMNLP tutorial on PEFT](https://www.youtube.com/watch?v=KoOlcX3XLd4)
- [Hugging Face RLHF tutorial](https://www.youtube.com/watch?v=2MBJOuVq380)
- [MunichNLP short version](https://www.youtube.com/watch?v=StdrAJZsmw4)

---

## 11. 🔑 Key Takeaways

1. **PEFT là giải pháp cho fine-tuning LLM với tài nguyên hạn chế** - Chỉ train 0.1-1% tham số
2. **Quantization + LoRA (QLoRA) là combo mạnh nhất** - Giảm VRAM ~4x, performance gần bằng full fine-tune
3. **Prompt Tuning phù hợp task đơn giản** - Rất nhẹ nhưng limited capacity
4. **LoRA là lựa chọn cân bằng nhất** - Performance tốt, VRAM vừa phải, dễ implement
5. **RLHF là kỹ thuật mạnh cho alignment** - Nhưng phức tạp, cần tuning nhiều
6. **Reward model quality quyết định quality của RLHF** - Garbage in, garbage out
7. **Luôn inspect data và model output thủ công** - Không tin tưởng metrics mù quáng
8. **KL penalty trong PPO rất quan trọng** - Ngăn model degenerate và reward hacking
