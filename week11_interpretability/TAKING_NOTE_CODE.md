# Ghi chú kiến thức & luồng hoạt động — Seminar + Homework

> Nguồn: `seminar_interpretability.ipynb`, `homework_interpretability.ipynb`
> Chủ đề: Week 11 — Interpretability thực hành (transformer_lens, activation patching, steering, SAE)

---

## PHẦN A. KIẾN THỨC CỐT LÕI

### A1. Thư viện sử dụng

| Thư viện | Vai trò |
|----------|---------|
| `transformer_lens` (HookedTransformer) | Load model GPT-2, chạy forward có gắn **hooks** để can thiệp vào activation từng layer/head |
| `circuitsvis` (cv) | Trực quan hóa attention pattern dạng tương tác (HTML) |
| `torch` / `matplotlib` | Tensor + vẽ biểu đồ |
| `datasets` (HF) | Load dataset Wikitext-2 cho việc thu thập activation (homework) |

### A2. HookedTransformer — khái niệm `hook` (móc)

- `model.run_with_cache(tokens)` → chạy forward và **trả về cache** chứa mọi activation trung gian.
- Cache có dạng `cache[f"blocks.{layer}.attn.hook_pattern"]`, `cache[f"blocks.{layer}.hook_resid_pre"]`, v.v.
- `model.run_with_hooks(tokens, fwd_hooks=[...])` → chạy forward và **gọi các hàm hook** tại các điểm được chỉ định.
- **fwd_hooks** = list `(tên_hook, hàm_hook)`; hàm hook có dạng `fn(activation_tensor, hook)` và phải **trả về** activation mới (đã sửa).

**Các tên hook quan trọng (GPT-2 small, 12 layers):**
- `blocks.{L}.attn.hook_pattern` — attention pattern `(batch, heads, Q_len, K_len)`
- `blocks.{L}.attn.hook_v` — value vectors `(batch, heads, Q_len, d_head)`
- `blocks.{L}.hook_resid_pre` — residual stream **trước** block L `(batch, seq, d_model)`
- `blocks.{L}.hook_resid_post` — residual stream sau block L

### A3. Logit Difference (hiệu logit)

> Metric trung tâm của bài IOI: đo "mức độ" model muốn chọn token nào.

```python
def logits_to_logit_diff(logits):
    john_id = model.to_single_token(" John")
    mary_id = model.to_single_token(" Mary")
    final_logits = logits[0, -1, :]          # logit của token cuối
    return float(final_logits[john_id] - final_logits[mary_id])
```

- **Logit diff > 0** → model nghiêng về "John"; **< 0** → nghiêng về "Mary".
- `model.to_single_token(" John")` — lưu ý **dấu cách** phía trước (token của từ).

### A4. Activation Patching (kỹ thuật can thiệp nhân quả)

> **Ý tưởng:** chạy prompt **clean** (đúng) và prompt **corrupt** (sai). Sau đó lấy activation tại một thành phần của *clean run* thay vào *corrupt run*. Nếu output "được sửa" → thành phần đó **có vai trò nhân quả** cho hành vi.

```python
def patch_resid_hook(resid, hook, position=pos, layer=L):
    resid[:, position, :] = clean_cache[f"blocks.{L}.hook_resid_pre"][0, position, :]
    return resid

patched_logits = model.run_with_hooks(
    clean_tokens,
    fwd_hooks=[(f"blocks.{L}.hook_resid_pre", patch_resid_hook)]
)
```

- Trong seminar: patch **residual stream** tại vị trí `pos=10` (vị trí từ "Mary"/"John" cuối cùng) của từng layer L.
- Sau đó vẽ scatter logit diff theo layer → tìm layer "nút thắt" quyết định pronoun.

### A5. Head Ablation & Scaling (vô hiệu hóa / khuếch đại head)

> Để **steer** model theo một hành vi, ta can thiệp vào **value vector** của các head quan trọng.

```python
def make_hook(head_list=head_list):
    def hook(v, hook):
        v[:, head_list, :, :] = 0.0          # ablate: zero hóa output các head
        # hoặc v[:, head_list, :, :] *= scale  # scale: nhân hệ số
        return v
    return hook

hooks.append((f"blocks.{layer}.attn.hook_v", make_hook()))
```

- `hook_v` shape: `(batch, heads, Q_len, d_head)`.
- **Ablate** (scale=0): xóa ảnh hưởng của head → xem model còn hành vi đó không.
- **Scale** (VD ×7.0): khuếch đại head → làm hành vi mạnh hơn.

### A6. Steering Vector (vector chỉ hướng cảm xúc)

> **Ý tưởng (homework):** trung bình activation của các câu positive trừ trung bình activation của các câu negative → được vector đại diện cho "sắc thái tích cực". Cộng vector này vào residual stream trong lúc generate → model bị "kéo" về hướng tích cực.

```python
steering_vectors = [pos_avg - neg_avg for pos_avg, neg_avg in zip(pos_avgs, neg_avgs)]
```

- Với `coef > 0` → steer tích cực; `coef < 0` → steer tiêu cực; `coef = 0` → trung tính.
- Lấy vector từ **residual pre hook** (`hook_resid_pre`) của **từng layer**.

### A7. Sparse Autoencoder (SAE)

> **Mục tiêu:** phân rã activation `x` thành tổ hợp sparse của các đặc trưng tiềm ẩn (feature dictionary).

```python
class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        f = torch.relu(self.encoder(x))   # hidden sparse (hầu hết = 0)
        x_hat = self.decoder(f)
        return x_hat, f
```

**Loss:**
```
loss = MSE(x, x_hat) + λ · ||f||₁
```
- `MSE` — tái tạo lại activation (reconstruction).
- `||f||₁` — **L1 penalty** ép `f` thưa (sparse), chỉ vài đặc trưng bật.
- `hidden_dim = input_dim * k` (VD ×4, ×8...) → **overcomplete** dictionary (nhiều đặc trưng hơn số chiều).

**Giải thích neuron:** sau khi train, với mỗi neuron, tìm các token/sentence có activation cao nhất → xem neuron đó "mã hóa" đặc trưng gì.

---

## PHẦN B. LUỒNG HOẠT ĐỘNG CÔNG VIỆC (SEMINAR)

> Mục tiêu: (1) trực quan hóa attention heads của GPT-2, (2) activation patching trên IOI task, (3) feature steering bằng head ablation/scaling.

### Luồng tổng quát:

```
[1] Setup: cài đặt + load GPT-2 qua HookedTransformer
    ↓
[2] Generate thử + trực quan hóa attention pattern (circuitsvis)
    ↓
[3] Tìm 5 loại attention heads
    ├── Earliest head attend to first Mary
    ├── Attention sink head
    ├── Positional-bias head
    ├── Induction head
    └── Punctuation-tracking head
    ↓
[4] IOI Activation Patching
    ├── Tạo clean/corrupt prompts
    ├── Đo logit diff baseline
    └── Patch residual stream từng layer → scatter plot
    ↓
[5] Feature steering
    ├── generate_with_hooks (vòng lặp sinh từng token)
    ├── ablate_heads (NAME_MOVER) → xóa hành vi
    └── scale_heads (NEG_NM × 7.0) → tăng hành vi
```

### Chi tiết từng bước:

**1. Setup model**
```python
model = HookedTransformer.from_pretrained("gpt2")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```
- `HookedTransformer.from_pretrained` tự tải config + weights GPT-2 và cho phép gắn hooks.

**2. Visualize attention**
```python
text = "When Mary and John went to the store, John gave a drink to Mary."
tokens = model.to_tokens(text).to(device)
_, cache = model.run_with_cache(tokens)
pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # (batch, heads, Q_len, K_len)
str_tokens = model.to_str_tokens(tokens)
a = cv.attention.attention_patterns(tokens=str_tokens, attention=pattern[0])
```

**3. Tìm các loại attention heads (1 điểm)**
- **Attention sink head:** trung bình attention vào **token 0** (`<|endoftext|>` / BOS) rất cao ở mọi vị trí. Duyệt từng layer, tính `pattern[:, :, :, 0].mean()`.
- **Induction head:** pattern cho thấy token hiện tại attend ngược lại **token xuất hiện trước đó** (dạng `A...B → A` — "lặp lại token trước"). Thường thấy ở layer giữa (L4-L5 trong GPT-2 small).
- **Positional-bias head:** attend theo **khoảng cách tuyệt đối/tương đối** (VD: luôn attend vào token trước đó 1 vị trí — anti-diagonal trong heatmap).
- **Punctuation-tracking head:** attend mạnh vào dấu câu / các token đặc biệt.
- **Earliest head attending to first Mary:** tìm layer + head đầu tiên có attention đáng kể vào vị trí "Mary" đầu câu khi đang sinh `*` (second `*`).

**4. IOI Activation Patching (2 điểm)**
```python
ioi_clean   = "After John and Mary went to the store, Mary gave a bottle of milk to"
ioi_corrupt = "After John and Mary went to the store, John gave a bottle of milk to"
```
- `torch.where(clean_tokens != corrupt_tokens)` → vị trí khác nhau (chỗ "Mary" vs "John").
- Baseline: clean cho logit diff **âm** (nghiêng Mary), corrupt cho diff **dương**.
- Patching: với mỗi layer L, thay residual stream ở vị trí `pos=10` của clean run vào corrupt run, đo lại logit diff.
- Kết quả: scatter plot cho thấy vài layer (VD ~8-10) **có ảnh hưởng lớn** → là nơi quyết định pronoun.

**5. Feature steering via head ablation (1 điểm)**
- `NAME_MOVER = [(9,6), (9,9), (10,0)]` — các head "di chuyển tên" → ablate (zero hóa) chúng → model mất khả năng điền đúng tên.
- `NEG_NM = [(10,7), (11,10)]` — negative name heads → **scale ×7** → model bị đẩy mạnh về hướng chọn đúng tên kia.

**6. Bonus:** thử nghiệm với các bài báo về IOI:
- https://arxiv.org/pdf/2211.00593 (Interpretability in the Wild — IOI circuit)
- https://arxiv.org/pdf/2310.04625 (có thể là paper về SAE/activation patching nâng cao)

---

## PHẦN C. LUỒNG HOẠT ĐỘNG CÔNG VIỆC (HOME WORK)

> Mục tiêu: (1) Sentiment steering bằng activation addition, (2) Sparse Autoencoder trên residual activations.

### Luồng tổng quát:

```
[1] Setup: load GPT-2 (giống seminar)
    ↓
[2] Part 1 — Sentiment Steering
    ├── Bộ câu positive/negative
    ├── collect_average_residuals → pos_avgs, neg_avgs
    ├── steering_vectors = pos - neg (từng layer)
    └── generate_with_steering(prompt, coef)
        ├── coef = 0.0  → neutral
        ├── coef = 0.3  → positive
        └── coef = -0.3 → negative
    ↓
[3] Part 2 — Sparse Autoencoder
    ├── Load Wikitext-2 → lọc câu rỗng
    ├── Thu thập residual activation (layer cuối) cho từng token
    ├── Định nghĩa SAE (encoder + ReLU + decoder)
    ├── Train với MSE + λ·L1
    └── Interpret: tìm top-k sentence kích hoạt mạnh nhất mỗi neuron
```

### Chi tiết từng bước:

**Part 1 — Sentiment Steering (3 điểm)**

*Bước 1: Thu thập residual activations trung bình*
```python
def collect_average_residuals(sent_list):
    avgs = []
    with torch.no_grad():
        for sent in sent_list:
            tokens = model.to_tokens(sent).to(device)
            _, cache = model.run_with_cache(tokens)
            layer_avgs = []
            for L in range(model.cfg.n_layers):
                resid = cache[f"blocks.{L}.hook_resid_pre"]  # (1, seq, d_model)
                layer_avgs.append(resid[0].mean(dim=0))      # trung bình theo seq
            avgs.append(torch.stack(layer_avgs))
    return torch.stack(avgs).mean(dim=0)   # trung bình theo câu
```
- **Hint quan trọng:** dùng **residual pre hook** của **từng layer** → vector cho mỗi layer.

*Bước 2: Tạo steering vectors*
```python
steering_vectors = [pos - neg for pos, neg in zip(pos_avgs, neg_avgs)]
```

*Bước 3: Generate có steering*
```python
def generate_with_steering(prompt, max_new_tokens=20, coef=0.0):
    tokens = model.to_tokens(prompt).to(device)
    for _ in range(max_new_tokens):
        hooks = []
        for L in range(model.cfg.n_layers):
            def make_hook(L=L):
                def hook(resid, hook):
                    resid += coef * steering_vectors[L]   # cộng vector vào residual
                    return resid
                return hook
            hooks.append((f"blocks.{L}.hook_resid_pre", make_hook()))
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        next_token = logits[0, -1].argmax().unsqueeze(0)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if next_token == model.tokenizer.eos_token_id:
            break
    return model.to_string(tokens[0, 1:])
```
- **Steering hiệu quả hơn tinkering với attention heads** vì tác động trực tiếp lên residual stream ở mọi layer.

**Part 2 — Sparse Autoencoder (4 + bonus điểm)**

*Bước 1: Thu thập activation dataset*
```python
last_layer = model.cfg.n_layers - 1
for sent in dataset_sentences:
    tokens = model.to_tokens(sent).to(device)
    _, cache = model.run_with_cache(tokens)
    resid = cache[f"blocks.{last_layer}.hook_resid_post"]  # residual layer cuối
    activations.append(resid.squeeze(0))
    token_to_sentence.extend([sent_id] * resid.shape[1])

activations = torch.cat(activations, dim=0)   # (total_tokens, d_model)
```
- Lưu `token_to_sentence` để biết mỗi token thuộc câu nào → phục vụ bước interpret sau.

*Bước 2: Định nghĩa SAE*
```python
class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        f = torch.relu(self.encoder(x))
        x_hat = self.decoder(f)
        return x_hat, f
```

*Bước 3: Train*
```python
sae = SAE(input_dim, hidden_dim).to(device)
optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
mse = nn.MSELoss()
l1_coef = ...   # hệ số sparsity

for epoch in range(num_epochs):
    optimizer.zero_grad()
    x_hat, f = sae(activations_device)
    loss = mse(x_hat, activations_device) + l1_coef * f.abs().sum()
    loss.backward()
    optimizer.step()
```
- `hidden_dim` thường = `input_dim * k` (VD 4×) → dictionary overcomplete.
- Có thể dùng batch nhỏ nếu dataset lớn (compute bound), hoặc giảm `num_epochs`.

*Bước 4: Interpret neuron*
```python
sae.eval()
with torch.no_grad():
    _, h = sae(activations)
    h = h.cpu().numpy()

for neuron in selected_neurons:
    top_idx = h[:, neuron].argsort()[::-1][:5]   # top-5 token kích hoạt mạnh nhất
    # dùng token_to_sentence để truy ra câu + vị trí token → in ra ngữ cảnh
```
- Mỗi neuron sẽ "bắt" một đặc trưng: VD neuron phát hiện từ số, từ hiếm, dấu câu, tên riêng...
- **Bonus:** thử nghiệm nhiều biến thể (Top-K SAE, JumpReLU, tăng hidden_dim, đổi λ...) để cải thiện interpretability.

---

## PHẦN D. TỔNG KẾT CÁC KIẾN THỨC CẦN NẮM

| Kiến thức | Kỹ thuật / Thư viện | Seminar | Homework |
|-----------|---------------------|:-------:|:--------:|
| Attention visualisation | `transformer_lens.run_with_cache` + `circuitsvis` | ✅ | - |
| Attention sinks / induction / positional heads | Phân tích `hook_pattern` | ✅ | - |
| Activation patching | `run_with_hooks` + cache | ✅ | - |
| Logit difference | `model.to_single_token` | ✅ | - |
| Head ablation / scaling | Hook trên `attn.hook_v` | ✅ | - |
| Steering vector | `hook_resid_pre` + cộng vector | - | ✅ |
| Residual activation collection | `run_with_cache` | - | ✅ |
| Sparse Autoencoder | PyTorch `nn.Module` + MSE + L1 | - | ✅ |
| Interpret neurons | top-k activation + token mapping | - | ✅ |

---

## PHẦN E. NHỮNG ĐIỂM CẦN LƯU Ý

1. **Tên hook phải chính xác:** `blocks.{L}.attn.hook_pattern`, `blocks.{L}.attn.hook_v`, `blocks.{L}.hook_resid_pre/post`. Sai tên → lỗi ngay.
2. **Hook function phải trả về activation** (`return resid` / `return v`), nếu không sẽ fail.
3. **`model.to_single_token(" John")` có dấu cách** — token của từ đi kèm khoảng trắng, không có dấu cách sẽ ra token khác.
4. **IOI logit diff:** clean → âm (Mary), corrupt → dương (John). Patching giúp xác định layer nhân quả.
5. **`NAME_MOVER` ablate → mất hành vi; `NEG_NM` scale cao → tăng hành vi** — hai chiều can thiệp ngược nhau.
6. **Steering vector** lấy từ residual pre **từng layer**, cộng với hệ số `coef`; quá lớn → text vỡ (gibberish), quá nhỏ → không hiệu quả.
7. **SAE là compute intensive** — có thể giảm `num_epochs`, dùng batch, hay giảm số câu; vẫn đạt max điểm nếu pipeline đúng và có interpret kết quả.
8. **Loss SAE = MSE (tái tạo) + λ·L1 (sparsity)** — cân bằng 2 mục tiêu này là chìa khóa.
9. **`token_to_sentence` rất quan trọng** — nếu mất mapping token→câu, không thể in ra ngữ cảnh interpret neuron.
10. Hai bài báo bonus: IOI circuit (2211.00593) và paper về patching nâng cao (2310.04625).

