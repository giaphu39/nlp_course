# RAG - Retrieval-Augmented Generation

## Tóm tắt bài giảng Week 09: Retrieval Stuff (NLP Course - YSDA)

**Giảng viên:** Michael Chertushkin

---

## Mục lục

1. LLM là "cỗ máy ảo giác" hoàn hảo (Hallucination)
2. Information Retrieval (IR) và khoảng cách phù hợp (Relevance Gap)
3. Naive RAG
4. Cách huấn luyện Naive RAG ("Baby RAG")
5. Advanced RAG
6. Iterative RAG
7. "RAG is Dead"?
8. Tương lai của retrieval và agentic RAG

---

## 1. LLM là "cỗ máy ảo giác" hoàn hảo

### 1.1 Hallucination (ảo giác) là gì?

- Hallucination là hiện tượng LLM trả lời bằng văn bản **trông "có vẻ đúng" nhưng thực tế không đúng**.
- Không quá nguy hiểm trong các ứng dụng B2C thông thường, **trừ các lĩnh vực nhạy cảm** như tư vấn sức khỏe / tài chính.
- Được coi là **rào cản lớn nhất cho thị trường B2B** — lý do chính khiến Gen AI chưa được áp dụng rộng rãi trong doanh nghiệp.

### 1.2 Vì sao hallucination xảy ra?

Việc huấn luyện LLM (năm 2025) thường được phân rã thành 2 pha:

- **Pre-training** (tiền huấn luyện)
- **Post-training** (hậu huấn luyện) — gồm SFT và căn chỉnh theo con người (human alignment)

#### Trong Pre-training

- Là bài toán tự hồi quy (autoregressive) dự đoán token tiếp theo với hàm mục tiêu **CrossEntropy**; (X, Y) := (context, token).
- Pre-training **không cố gắng**:
  - làm cho kết quả chính xác tuyệt đối,
  - kiểm tra với database,
  - nói "tôi không biết".
- Chính bản chất tối thiểu CrossEntropy trong dự đoán token tiếp theo sẽ **khuyến khích việc "đoán"**.

#### Trong SFT (Supervised Fine-Tuning)

- Là bài toán autoregressive với CrossEntropy; (X, Y) := (instruction, instruction_result).
- SFT **cũng không làm gì** để đảm bảo câu trả lời "đúng về mặt sự kiện".

#### Trong Post-training (human alignment)

- Là nỗ lực căn chỉnh theo con người: chọn giữa phản hồi bị từ chối (rejected) và phản hồi được chấp nhận (accepted).
- Tùy phương pháp:
  - **RLHF**: dùng Reward Model + PPO, tối thiểu pairwise ranking loss.
  - **DPO**: dùng chính LLM làm reward model.
- Điểm quan trọng: mô hình thiên về thứ annotator ưa thích, mà annotator lại thích các câu trả lời **"tự tin"**, **"nghe hợp lý"**, **"đi sâu vào chi tiết"**.

### 1.3 Tổng kết

- Ở giai đoạn phát triển hiện tại, **hallucination được khuyến khích ở mọi giai đoạn huấn luyện** và gần như không thể tránh khỏi.
- Hallucination (cùng với lo ngại bảo mật và khó kết nối hệ thống bên thứ 3) chặn việc áp dụng Gen AI vào B2B.
- Có những nỗ lực đưa "hình phạt" cho câu trả lời tự tin nhưng sai vào loss function, nhưng **chưa phổ biến**.

### 1.4 Giải pháp

Nếu không thể giảm hallucination trong lúc huấn luyện (hiện tại), ta có thể làm gì đó **ở test-time** để giảm thiểu → dẫn đến ngành **Information Retrieval (IR)**.

---

## 2. Information Retrieval (IR)

### 2.1 Góc nhìn 1000 feet

- IR là tìm các mục **liên quan** trong một kho ngữ liệu lớn, dựa trên câu truy vấn (query).
- Ta có:
  - Tập tài liệu (collection of documents),
  - Câu truy vấn (query),
  - Cần tạo danh sách xếp hạng d₁, d₂, ..., dₙ sao cho tài liệu liên quan nhất đứng đầu.
- Đạt được bằng cách **học hàm chấm điểm s(q, d)**.

### 2.2 TF-IDF

- Vector hóa toàn bộ corpus bằng `TfidfVectorizer()`.
- Khi có query, biến đổi query thành vector rồi tìm các vector gần nhất.
- Hoạt động rất tốt làm **baseline**.
- **Nhược điểm**: bỏ qua thứ tự từ (bag of words), quá đơn giản.

### 2.3 BM25 — phương pháp IR cổ điển đáng chú ý nhất

- Chấm điểm mỗi tài liệu D cho cặp (D, Q) theo công thức riêng.
- Hoạt động tốt ("25" là số phiên bản — họ đã lặp 25 lần).
- Sẽ quay lại BM25 ở phần **Hybrid Search**.

### 2.4 Ưu / nhược điểm của phương pháp cổ điển

| Ưu điểm | Nhược điểm |
|---|---|
| Rất nhanh | Không nhận diện từ đồng nghĩa (doctor ≠ physician) |
| Xuất sắc cho khớp chính xác (exact match) | Phụ thuộc tokenization / stemming / lemmatization |
| Dễ diễn giải (có điểm cho từng term) | Không làm được multi-hop reasoning |

### 2.5 Metrics trong IR

- **Precision@k**
- **Recall@k**
- **MRR** (Mean Reciprocal Rank)
- **NDCG@k**

---

## 3. Từ IR đến RAG

### 3.1 Khái niệm RAG

- **RAG (Retrieval-Augmented Generation)** là kỹ thuật **"làm giàu" context của LLM tại test-time** bằng thông tin được truy hồi dựa trên query.
- Nói cách khác, ta **conditioning** lên các tài liệu được truy hồi.
- "Condition" đơn giản chỉ là **nối (append) tài liệu vào context** — nhưng nối ở đâu: đầu, giữa hay cuối? → liên quan bài báo *Lost in the Middle: How Language Models Use Long Contexts* (arXiv:2307.03172).

### 3.2 Hai loại RAG

1. **Joint RAG** (Lewis et al., 2020) — huấn luyện chung retriever + generator:
   - **RAG-Sequence**: một tài liệu z được chọn (như một biến tiềm ẩn) và dùng để sinh **toàn bộ chuỗi** đầu ra.
   - **RAG-Token**: **mỗi token y có thể đến từ một tài liệu khác nhau**.
   - Ý tưởng joint training rất hợp lý và được thử ở nhiều công trình, nhưng có lý do cơ bản khiến nó không được áp dụng rộng rãi: **LLM thường là hộp đen (black box)** trong môi trường công nghiệp.

2. **Modular RAG** — trọng tâm của bài giảng:
   - LLM là hộp đen → có nhu cầu rất lớn về phương pháp **làm giàu context mà không đụng vào LLM**.
   - Có **retriever** (có thể tinh chỉnh qua Embedding Tuning) để đạt precision@k, recall@k cao.
   - Có **reranker** (cross-encoder) cũng có thể tinh chỉnh để tối ưu MRR/NDCG.
   - Toàn bộ setup này tạo ra vấn đề gọi là **Relevance Gap**.

---

## 4. Relevance Gap (khoảng cách phù hợp)

- **Retriever** được tối ưu để đưa về tài liệu trông "liên quan" theo một metric IR nào đó.
- Nhưng **generator** quan tâm đến tài liệu thực sự giúp nó trả lời câu hỏi một cách **trung thực và chính xác**.
- Hệ quả:
  - Retriever trả về tài liệu có đúng keyword nhưng **không chứa bằng chứng** cần thiết.
  - Hoặc trả về đúng tài liệu nhưng **bị chôn vùi** giữa các đoạn gần trùng lặp / nhiễu, khiến LLM không tập trung vào phần quan trọng.
- **"Advanced RAG" thực chất là tập hợp nhiều phương pháp khác nhau được dán lại với nhau** để thu hẹp relevance gap này.

---

## 5. Cách huấn luyện "Baby RAG" (tự tay huấn luyện RAG)

Để huấn luyện RAG của riêng mình, cần 3 thứ:

1. Định nghĩa **mô hình ML** (kiến trúc NN hoặc chọn họ phương pháp).
2. Định nghĩa **hàm mục tiêu / loss**.
3. Định nghĩa **bộ dữ liệu (X, Y)**.

### 5.1 Dữ liệu

- Muốn tài liệu liên quan **gần** query trong không gian tiềm ẩn, tài liệu không liên quan thì **xa**.
- Lựa chọn hiển nhiên:
  - (q, d, **1**) cho cặp liên quan (q, d).
  - (q, d, **0**) cho cặp không liên quan.
- **Câu hỏi cho người đọc kỹ:** có thể dùng **-1** cho cặp không liên quan hay không?

### 5.2 Contrastive (Siamese) Loss

- Dùng khi có label 1 / 0.
- Ví dụ **Margin Based Contrastive Loss**:
  - D — khoảng cách Euclidean giữa embedding của query và embedding của document.
  - M — margin.
- Đẩy các cặp liên quan lại gần nhau, đẩy các cặp không liên quan ra xa.

### 5.3 Triplet Loss (loss dùng trong seminar)

- Biến thể không dùng label tường minh:

  L = max(0, δ − sim[V_q(q), V_a(a⁺)] + sim[V_q(q), V_a(a⁻)])

- Diễn giải: **câu trả lời đúng phải gần câu hỏi hơn câu trả lời sai ít nhất một khoảng δ**.
- Đây chính là cách huấn luyện mô hình **DSSM** (Deep Semantic Similarity Model).

### 5.4 Kiến trúc mô hình

- Cần một thành phần nhận (Query, Doc) và tạo ra embedding.
- Chi tiết kiến trúc sẽ được thảo luận trong seminar (thực hành: dùng BERT pretrained + MLP head).

---

## 6. Advanced RAG — 3 giai đoạn tối ưu

Advanced RAG coi pipeline là **3 giai đoạn có thể tối ưu**:

1. **Pre-Retrieval** — indexing, chunking nâng cao
2. **Retrieval** — truy hồi thực tế
3. **Post-Retrieval** — nén, lọc, xếp hạng lại

### 6.1 Pre-Retrieval

- Mục tiêu chính: sửa lỗi **"Garbage in / Garbage out"**.
  - Nếu index cấu trúc kém hoặc query mơ hồ, không retrieval/tuning nào cứu được output.
  - *"Hệ thống RAG của bạn không tệ. Câu hỏi (query) của bạn mới tệ."*
- Hai chủ đề chính:

#### a) Query Transformation (biến đổi query, gồm query expansion)

- **HyDE — Hypothetical Document Embeddings** (arXiv:2212.10496): dùng LLM khác tạo ra một **"tài liệu giả định"** từ query, rồi nhúng tài liệu đó để truy hồi.
- **Query rewrite / expand / decompose** (arXiv:2305.03653): dùng LLM viết lại, mở rộng hoặc phân rã query.
- **Multi-query (synonym queries)**: tạo nhiều query đồng nghĩa và gửi tất cả vào retriever.
  - Tăng **recall** nhưng giảm **precision**.
  - Được dùng trong nhiều hệ thống production, đặc biệt khi gửi song song (concurrently).

#### b) Index & Corpus Design (thiết kế index và kho ngữ liệu)

- **Hierarchical Indexing (chỉ mục phân cấp)**:
  - Tóm tắt ngắn toàn bộ tài liệu (top level),
  - Tóm tắt chi tiết hơn từng mục (middle level),
  - Các chunks chi tiết (bottom level).
  - Truy hồi kiểu **top-down**: query khớp với summary cấp cao → tìm thấy → "zoom vào" lấy các chunks chi tiết liên quan, đảm bảo context vừa liên quan vừa đầy đủ.
- **Multi-representation / Multi-vector indexing**:
  - Tạo **nhiều biểu diễn vector** cho một tài liệu/chunk, ví dụ: văn bản gốc, tóm tắt văn bản, **"câu hỏi tiềm năng"** về văn bản đó.
  - Cách này **hoạt động cực kỳ tốt** trong hệ QnA thời gian thực.

### 6.2 Retrieval

- Truy hồi thực tế chỉ là bước tính similarity giữa các docs và query đến.
- Nhưng có 2 phương pháp cải thiện chất lượng đáng kể:

#### a) Hybrid Search (tìm kiếm lai)

- Anthropic công bố kết hợp **BM25 + semantic similarity RAG** (Contextual Retrieval).
- Tóm tắt phát hiện:
  - Embeddings + BM25 **tốt hơn** chỉ dùng embeddings.
  - Đưa **top-20 chunks** vào model hiệu quả hơn top-10 hay top-5.
  - **Thêm context vào chunks** cải thiện độ chính xác rất nhiều.
  - **Reranking tốt hơn không reranking**.
  - Các lợi ích **cộng dồn**: contextual embeddings + contextual BM25 + reranking + 20 chunks trong prompt.

#### b) Fusion Algorithms (thuật toán hợp nhất)

- Sau khi kết hợp BM25 + embedding similarity, ta gặp **bài toán hợp nhất hai danh sách** (fusion problem).
- Thuật toán SOTA: **Reciprocal Rank Fusion (RRF)**.
  - RRF **bỏ qua hoàn toàn điểm số gốc** (chưa chuẩn hóa).
  - Chỉ dựa vào **vị trí (rank)** của tài liệu trong mỗi danh sách.

### 6.3 Post-Retrieval

- Retrieval tối ưu cho **recall**, nhưng có thể có nhiều **false positive** (precision thấp).
- Post-Retrieval có mục đích **tăng precision** trong khi giữ nguyên recall.
- Hai lớp phương pháp đáng chú ý:

#### a) Reranking

- Lấy top-k tài liệu từ retriever, áp mô hình mạnh hơn (nhưng tốn tính toán) để tạo ranking mới chính xác hơn.
- **Cross-encoders**: rất đắt. Là mô hình tương tác đầy đủ: nối (query, doc) rồi đưa qua toàn bộ transformer.
- **Late-interaction models**: rẻ hơn, hoạt động 2 bước:
  - Lúc indexing: mã hóa từng token của document (chỉ làm 1 lần).
  - Lúc test-time: mã hóa query từng token rồi tìm **max similarity** giữa token của query và token của document.

#### b) Context Compression (nén context)

- Loại bỏ các câu/token không liên quan bên trong tài liệu đã chọn → "khử nhiễu" context trước khi vào LLM.
- Có thể thực hiện bằng một LLM khác.
- Hoặc bằng **xRAG** (arXiv:2405.13792): diễn giải lại document embeddings trong dense retrieval — vốn chỉ dùng để truy hồi — thành **features từ modality retrieval**, nén context cực đoan thành **một token**.

### 6.4 Kết luận tạm thời

- Vì LLM là "hộp đen", nếu thực sự làm việc trong lĩnh vực này, khả năng cao việc cải thiện Retriever trở thành **bài toán kỹ thuật (engineering)** hơn là bài toán tối ưu (optimization).

---

## 7. Iterative RAG

- Ý tưởng: vượt qua pipeline tuyến tính "retrieve-then-generate" và **lặp lại** trên nó.
- 3 phiên bản đáng chú ý:

### 7.1 Self-RAG (Asai et al., 2023)

- Huấn luyện **một LLM duy nhất** tự kiểm soát quá trình RAG của chính nó thông qua **self-reflection** (tự phản ánh).
- LLM tự quyết định khi nào cần retrieve, khi nào sinh, và **phê bình chính output** của mình.

### 7.2 GraphRAG (Wang et al., 2023)

- Vector-search thuần **không "nối được các dấu chấm"** giữa các mẩu thông tin rời rạc.
- Ý tưởng chính: dùng **Knowledge Graph (KG)** thay vì vector thuần.
  - Lúc query, hệ thống truy hồi **sub-graphs** và **semantic communities** từ KG, không chỉ các chunk văn bản cô lập.
  - Context có cấu trúc cho phép GraphRAG trả lời các câu hỏi phức tạp mà vector search không thể.
- Lợi ích chính: **multi-hop reasoning có sẵn** (out-of-the-box).

### 7.3 Agentic RAG (Wang et al., 2023)

- Nhúng toàn bộ pipeline RAG vào **framework agent tự trị** — "Self-RAG on steroids".
  - Agent có thể **lên kế hoạch** retrieve gì tiếp theo.
  - Agent có thể **dùng tools** để lấy thêm thông tin.
  - Agent có thể **gọi evaluation** để đánh giá hiệu suất hiện tại.

---

## 8. "RAG is Dead"?

- Khi **Long Context Window LLMs** xuất hiện, có tuyên bố rằng RAG đã chết. Có thật vậy không?
- Long Context LLM và RAG phục vụ **cùng mục tiêu**:
  - Grounding câu trả lời dựa trên context.
  - Giảm hallucination.
- Nhưng có **khác biệt rất rõ ràng** giữa hai cách tiếp cận:
  - **Lost in the Middle problem** (arXiv:2307.03172): **vị trí đặt tài liệu trong context ảnh hưởng chất lượng** câu trả lời.
  - Nhét mọi thứ vào cửa sổ context dài không phải lúc nào cũng tốt; truy hồi chọn lọc của RAG vẫn có giá trị riêng.

---

## 9. Đánh giá hệ thống RAG

- Các metrics IR cổ điển vẫn được dùng.
- Thêm các metrics Gen AI mới:
  - **Faithfulness / Groundedness**: mức độ trung thành với context được cung cấp.
  - **Answer Relevance**: mức độ liên quan của câu trả lời.
- Framework đánh giá chuyên cho RAG:
  - **RAGAS**
  - **DeepEval**

---

## 10. Tương lai của RAG

Không thể đoán chính xác, nhưng có thể suy đoán:

1. Ở giai đoạn hiện tại, RAG là một **"cái nạng" (costyl)** — nó không giải quyết triệt để vấn đề hallucination.
2. RAG hiện tại thiên về **heuristics và engineering/pipelining** hơn là optimization.
3. Năm 2025, giới nghiên cứu **quay lại bài báo Joint RAG (2020)** để tối ưu **chung generator + retriever**.

---

## Từ khóa quan trọng

`Hallucination` · `Information Retrieval` · `TF-IDF` · `BM25` · `Precision@k` · `Recall@k` · `MRR` · `NDCG@k` · `RAG-Sequence` · `RAG-Token` · `Modular RAG` · `Relevance Gap` · `Contrastive Loss` · `Triplet Loss` · `DSSM` · `HyDE` · `Query Expansion` · `Hierarchical Indexing` · `Multi-vector Indexing` · `Hybrid Search` · `Reciprocal Rank Fusion` · `Cross-encoder` · `Late-interaction` · `Context Compression` · `xRAG` · `Self-RAG` · `GraphRAG` · `Agentic RAG` · `Lost in the Middle` · `RAGAS` · `DeepEval`

