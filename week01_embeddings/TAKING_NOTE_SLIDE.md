# Tổng Kết Kiến Thức Tuần 1: Word Embeddings & Text Classification

> **Nguồn slide tham khảo:** YSDA NLP Course (Word Embeddings & Text Classification - Lena Voita)
> **Chủ đề:** Biểu diễn từ (Word Representations) & Phân loại văn bản cổ điển + neural (Text Classification)

---

## PHẦN 1. WORD EMBEDDINGS (NHÚNG TỪ)

### 1. Ý tưởng cốt lõi & Một số phương pháp biểu diễn từ
- **Cách mô hình "nhìn" từ:** Máy tính cần vector đặc trưng số để xử lý văn bản. Hệ thống ánh xạ từng từ thông qua bảng tra cứu (Look-up Table) bằng chỉ số (index) trong từ điển từ vựng (Vocabulary). Từ chưa biết (Out of Vocabulary - OOV) được biểu diễn bằng token đặc biệt `UNK` hoặc gán vector không.
- **One-hot Vectors:**
  - *Cách hoạt động:* Vector có kích thước bằng từ điển $|V|$, có giá trị $1$ ở vị trí index của từ đó, còn lại là $0$.
  - *Hạn chế:* Số chiều quá lớn khi từ điển tăng; các vector trực giao với nhau $\Rightarrow$ không phản ánh được mối quan hệ ngữ nghĩa (ví dụ: khoảng cách giữa "cat" và "dog" bằng khoảng cách giữa "cat" và "table").
- **Distributional Semantics (Ngữ nghĩa phân bố):**
  - **Giả thuyết phân bố (Distributional Hypothesis):** *"Các từ thường xuất hiện trong ngữ cảnh tương tự nhau thì có nghĩa tương tự nhau"* (Firth 1957; Harris 1954).
  - *Mục tiêu:* Đưa thông tin ngữ cảnh xung quanh vào chính vector biểu diễn từ.

---

### 2. Các phương pháp dựa trên đếm (Count-Based Methods - Pre-neural)
Xây dựng ma trận từ - ngữ cảnh (word-context matrix) thủ công dựa trên thống kê tần suất toàn cục của ngữ liệu, sau đó giảm số chiều (dimensionality reduction) để nén và khử nhiễu.

- **Co-occurrence Counts (Đếm đồng xuất hiện):**
  - Định nghĩa ngữ cảnh là các từ xuất hiện trong cửa sổ kích thước $L$ xung quanh từ mục tiêu.
  - Ma trận chứa tần suất xuất hiện đồng thời trực tiếp của các cặp từ.
- **PPMI (Positive Pointwise Mutual Information):**
  - Thay vì đếm trực tiếp, mô hình đo lường mức độ liên kết chặt chẽ giữa từ $w$ và ngữ cảnh $c$:

$$PMI(w, c) = \log \frac{P(w, c)}{P(w)P(c)}$$

$$PPMI(w, c) = \max(0, PMI(w, c))$$

  - Giúp loại bỏ ảnh hưởng của các từ quá phổ biến nhưng ít mang thông tin (như "the", "a").
- **LSA (Latent Semantic Analysis):**
  - Áp dụng phân tích suy hao trị riêng SVD (Singular Value Decomposition) lên ma trận term-document nhằm biểu diễn các tài liệu và thuật ngữ dưới dạng các chủ đề ẩn (latent topics).

---

### 3. Phương pháp dựa trên dự đoán: Word2Vec
Ý tưởng: **Học** vector từ bằng cách tối ưu hóa các tham số để đi dự đoán từ ngữ cảnh xung quanh.

- **Hàm mục tiêu (Loss function):** Average Negative Log-Likelihood trên một cửa sổ trượt kích thước $m$:

$$J(\theta) = -\frac{1}{T} \sum_{t=1}^T \sum_{-m \le j \le m, j \neq 0} \log P(w_{t+j} | w_t)$$

- **Cách tính xác suất qua Softmax:**
  Mỗi từ có 2 vector: $v_w$ khi đóng vai trò là từ trung tâm (center word) and $u_w$ khi đóng vai trò là từ ngữ cảnh (context word).

  $$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

- **Tối ưu hóa tốc độ huấn luyện:**
  - **Negative Sampling (SGNS):** Việc tính mẫu số của Softmax trên toàn bộ từ điển $V$ cực kỳ tốn kém ($O(|V|)$). SGNS chuyển bài toán đa phân loại thành phân loại nhị phân: tăng độ tương đồng giữa từ trung tâm với từ ngữ cảnh thật (1 positive), đồng thời giảm độ tương đồng với $K$ từ ngẫu nhiên được chọn (K negatives).

$$J_{t, j}(\theta) = -\log \sigma(u_o^T v_c) - \sum_{w\in \text{Neg}} \log \sigma(-u_w^T v_c)$$

  - **Lựa chọn mẫu tiêu cực (Negative Examples):** Chọn ngẫu nhiên theo phân phối tần suất lũy thừa $U^{3/4}(w)$ để tăng cơ hội lấy các từ hiếm làm mẫu tiêu cực.
- **Hai biến thể của Word2Vec:**
  - **Skip-Gram:** Dự đoán các từ ngữ cảnh từ từ trung tâm (phổ biến hơn và hiệu quả hơn với từ hiếm).
  - **CBOW (Continuous Bag-of-Words):** Dự đoán từ trung tâm dựa trên tổng các vector ngữ cảnh xung quanh.
- **Ảnh hưởng của kích thước cửa sổ:**
  - Cửa sổ nhỏ (2-5): Học quan hệ cú pháp và chức năng (ví dụ: các danh từ chỉ loài chó đi cùng nhau, động từ cùng chia thì).
  - Cửa sổ lớn (5-10): Học quan hệ chủ đề và ngữ nghĩa rộng hơn (ví dụ: "dog" đi cùng "bark", "leash", "walk").

---

### 4. GloVe (Global Vectors for Word Representation)
- Kết hợp cả hai hướng tiếp cận: dựa trên đếm (counts) và dựa trên dự đoán (prediction).
- Hàm loss tối ưu trực tiếp trên ma trận đồng xuất hiện toàn cục $X_{ij}$:

$$J = \sum_{i,j=1}^V f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

  Trong đó $f(X_{ij})$ là hàm trọng số nhằm phạt các cặp từ quá hiếm và giới hạn ảnh hưởng của các cặp từ quá phổ biến.

---

### 5. Đánh giá và Phân tích không gian ngữ nghĩa
- **Phương pháp đánh giá:**
  - **Intrinsic (Nội tại):** Đánh giá trực tiếp trên các bài test phụ như tương đồng từ (Word Similarity) bằng Cosine Similarity hoặc phép loại suy (Word Analogy): $v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$.
  - **Extrinsic (Ngoại tại):** Đánh giá dựa trên hiệu năng của vector nhúng khi làm đầu vào cho các tác vụ downstream thực tế (như phân loại văn bản).
- **Đặc trưng hình học không gian nhúng:**
  - Mối quan hệ đa ngôn ngữ: Các không gian ngữ nghĩa của các ngôn ngữ khác nhau có tính chất tuyến tính tương đối $\Rightarrow$ có thể dịch chuyển tuyến tính qua ma trận quay $W$ tìm bởi phương pháp Orthogonal Procrustes.
  - Bias (Thiên kiến): Vector nhúng bị nhiễm các định kiến xã hội từ dữ liệu huấn luyện (ví dụ: "man" đi với "computer programmer" tương ứng "woman" đi với "homemaker"). Có thể loại bỏ bằng phương pháp INLP (Iterative Nullspace Projection) để khử thông tin định kiến mà ít làm giảm chất lượng biểu diễn chung.

---

## PHẦN 2. TEXT CLASSIFICATION (PHÂN LOẠI VĂN BẢN)

### 1. Các phương pháp cổ điển (Classical Methods)
- **Naive Bayes (Generative Model):**
  - Dự đoán nhãn $y$ thông qua ước lượng xác suất đồng thời $P(x, y)$ nhờ quy tắc Bayes: $P(y|x) \propto P(y) P(x|y)$.
  - Giả định "Naive": Các từ độc lập có điều kiện khi biết nhãn và thứ tự từ không quan trọng (Bag of Words).
  - Áp dụng **Laplace smoothing (Add-$\delta$ smoothing)** để tránh xác suất bằng 0 khi gặp từ chưa xuất hiện trong tập huấn luyện.
  - Sử dụng tổng các log xác suất để tránh tràn số (underflow) do nhân quá nhiều số thập phân nhỏ.
- **Logistic Regression / MaxEnt (Discriminative Model):**
  - Mô hình hóa trực tiếp xác suất điều kiện $P(y|x)$ bằng cách nhân trọng số đặc trưng và đưa qua hàm Softmax.
  - Việc cực đại hóa hàm Likelihood của tập dữ liệu tương đương với cực tiểu hóa hàm cross-entropy loss.
- **SVM (Support Vector Machine):**
  - Phổ biến với đặc trưng Bag-of-Words hoặc Bag-of-Ngrams kết hợp với nhân tuyến tính (Linear Kernel).

---

### 2. Phân loại văn bản bằng Neural Networks
Học tự động đặc trưng biểu diễn văn bản từ chuỗi vector nhúng thay vì định nghĩa đặc trưng thủ công.

- **Luồng hoạt động chung:**

$$\text{Từ} \rightarrow \text{Word Embeddings} \rightarrow \text{Neural Network (RNN/CNN)} \rightarrow \text{Vector biểu diễn văn bản cố định } (d) \rightarrow \text{Linear Layer} \rightarrow \text{Softmax} \rightarrow \text{Xác suất nhãn}$$

- **Các kiến trúc phổ biến:**
  - **Bag of Embeddings (BOE):** Cộng hoặc trung bình cộng các vector nhúng của từ trong văn bản. Không giữ thứ tự từ nhưng giữ được quan hệ ngữ nghĩa tốt hơn nhiều so với Bag of Words.
  - **Mô hình tuần tự (Recurrent - RNN/LSTM/GRU):** Đọc văn bản theo thứ tự từ trái qua phải. Lấy trạng thái ẩn cuối cùng $h_{\text{last}}$ làm vector đại diện cho toàn bộ văn bản. Có thể xếp chồng nhiều lớp (Multi-layer) hoặc chạy hai chiều (Bidirectional) để tối ưu khả năng nhớ thông tin.
  - **Mô hình tích chập (Convolutional - CNN):**
    - Áp dụng các bộ lọc trượt một chiều (1D Convolution) đóng vai trò như các bộ phát hiện N-gram.
    - Sử dụng **Global-over-time Max-Pooling** để nén chuỗi kết quả có chiều dài biến thiên thành một vector đặc trưng có kích thước cố định (đạt được translation invariance - tính bất biến dịch chuyển).
    - Có thể dùng song song nhiều kích thước filter khác nhau (ví dụ: filter size 3, 4, 5 tương ứng với 3, 4, 5-gram).

---

### 3. Phân loại đa nhãn (Multi-Label Classification)
Tác vụ mà một văn bản có thể có nhiều nhãn đúng đồng thời (ví dụ: một tweet có nhiều hashtag).
- **Thay đổi kiến trúc:** Thay thế hàm Softmax ở đầu ra bằng hàm Sigmoid độc lập cho mỗi lớp.
- **Thay đổi Loss:** Thay thế Categorical Cross-Entropy bằng tổng các hàm Binary Cross-Entropy trên từng lớp nhãn.

---

### 4. Một số thủ thuật thực tế (Practical Tips)
- **Xử lý pretrained embeddings:** Có 3 cách tiếp cận chính:
  - *Train từ đầu (scratch):* Dễ overfit nếu tập data nhỏ.
  - *Giữ nguyên (Static):* Hữu ích khi tập dữ liệu nhỏ và muốn giữ lại kiến thức tổng quát từ tập ngữ liệu lớn.
  - *Fine-tune:* Cập nhật vector nhúng trong quá trình huấn luyện phân loại giúp giải quyết bài toán đồng âm/trái nghĩa tốt hơn (ví dụ: trong không gian tĩnh của Word2Vec, "good" và "bad" rất gần nhau vì chung ngữ cảnh; fine-tune giúp tách rời chúng dựa trên nhãn phân loại cảm xúc).
- **Data Augmentation cho văn bản:**
  - *Word Dropout:* Thay thế ngẫu nhiên một số từ bằng `UNK` hoặc từ ngẫu nhiên khác để mô hình không phụ thuộc quá mức vào các từ khóa riêng lẻ.
  - *Synonym Replacement:* Thay thế từ bằng từ đồng nghĩa nhờ các nguồn từ điển (như WordNet).
  - *Back-translation (Dịch ngược):* Dịch văn bản sang ngôn ngữ trung gian rồi dịch ngược lại để tạo câu diễn đạt tương tự (paraphrase).

---

## PHẦN 3. LIÊN HỆ KIẾN THỨC MỚI TRONG CÁC BÀI BÁO HIỆN TẠI (2021 - 2026)

Dưới đây là sự phát triển và liên hệ các lý thuyết tuần 1 với các nghiên cứu hiện đại trong NLP:

### 1. Sự dịch chuyển từ Static sang Contextualized Embeddings và LLM Embeddings
- **Liên hệ lý thuyết tuần 1:** Các phương pháp cổ điển (Word2Vec, GloVe) gán cho mỗi từ một vector tĩnh duy nhất (Static Embedding) bất kể ngữ cảnh (từ "bank" trong "river bank" và "money bank" có cùng vector).
- **Cập nhật hiện đại:**
  - Không gian biểu diễn từ hiện nay chủ yếu được tạo ra từ các mô hình Transformer (BERT/DeBERTa hoặc các LLM Decoder-only như LLaMA/GPT) tạo ra **Contextualized Embeddings** (Nhúng phụ thuộc ngữ cảnh) động tại mỗi bước attention.
  - **MTEB (Massive Text Embedding Benchmark)** là bảng xếp hạng chuẩn hóa đánh giá chất lượng vector biểu diễn văn bản hiện nay trên nhiều tác vụ khác nhau (Retrieval, Classification, Clustering, Analogy).

### 2. Học biểu diễn văn bản bằng học tương phản (Contrastive Learning)
- **SimCSE (Simple Contrastive Learning of Sentence Embeddings - EMNLP 2021):**
  - Đưa ra phương pháp học nhúng câu tự giám sát (self-supervised) bằng cách áp dụng **Dropout** như một kỹ thuật tăng cường dữ liệu (data augmentation) tối giản.
  - Đưa cùng một câu qua Encoder hai lần với các mặt nạ dropout khác nhau để tạo ra cặp mẫu tích cực (positive pair), và coi các câu khác trong batch là mẫu tiêu cực (negative samples).
  - Phương pháp này giải quyết trực tiếp hiện tượng sụp đổ không gian nhúng (**representation collapse/anisotropy**) - điều mà các tác giả của bài báo *"All but the top"* (EMNLP 2017) trong slide đã giải quyết bằng cách trừ vector trung bình và loại bỏ các thành phần PCA lớn nhất.
- **Contriever (2022) & BGE Embeddings (BAAI - 2023):**
  - Sử dụng học tương phản trên quy mô lớn để huấn luyện các bộ Bi-Encoder phục vụ cho hệ thống RAG (Retrieval-Augmented Generation).

### 3. Nhúng văn bản định hướng bằng chỉ dẫn (Instruction-Tuned Text Embeddings)
- **E5 (EmbEddings from Bidirectional Encoder Representations - 2022/2024):**
  - Nghiên cứu chỉ ra rằng việc thêm chỉ dẫn (instruction) trước văn bản (ví dụ: *"Query: retrieve documents that explain..."* hoặc *"Document: ..."*) giúp mô hình nhúng biết cách định hình không gian vector phù hợp với mục đích của người dùng.
  - Điều này nâng cấp ý tưởng "tuyến tính hóa và ánh xạ không gian ngữ nghĩa" trong bài giảng (dịch chuyển không gian ngôn ngữ thông qua ma trận $W$) thành việc **điều khiển động không gian nhúng** bằng ngôn ngữ tự nhiên.

### 4. Khử thiên kiến (Debiasing) trong kỷ nguyên LLM
- **Liên hệ lý thuyết tuần 1:** Slide giới thiệu thuật toán chiếu nullspace tuyến tính tuyến tính INLP để debias thông tin giới tính.
- **Cập nhật hiện đại:**
  - Việc loại bỏ thiên kiến (debiasing) nay được thực hiện ở cấp độ LLM thông qua các phương pháp căn chỉnh phản hồi người dùng như **RLHF (Reinforcement Learning from Human Feedback)** và **DPO (Direct Preference Optimization - NeurIPS 2023)**.
  - **Mechanistic Interpretability & Representation Engineering (Zou et al., 2023):** Thay vì chiếu nullspace tĩnh, nghiên cứu hiện đại tìm các "vector chỉ đạo" (steering vectors) bên trong các lớp ẩn của LLM và thực hiện can thiệp trực tiếp (activation patching) để dẫn dắt hành vi của mô hình (ví dụ: khử độc tố, giảm bias về giới tính hoặc sắc tộc khi sinh văn bản).

### 5. LLM as Classifiers & PEFT (Parameter-Efficient Fine-Tuning)
- **Liên hệ lý thuyết tuần 1:** Dùng RNN/CNN với linear layer để phân loại văn bản.
- **Cập nhật hiện đại:**
  - **Zero-shot / Few-shot Classification với LLMs:** Sử dụng prompt chỉ dẫn trực tiếp cho LLM để phân loại văn bản mà không cần huấn luyện lại trọng số (In-Context Learning).
  - **PEFT (LoRA, QLoRA - 2021/2023):** Khi cần tối ưu hóa độ chính xác cho mô hình phân loại cỡ nhỏ hoặc trung bình (như LLaMA-8B hay DeBERTa-v3), người ta chèn thêm các ma trận tích hợp hạng thấp (Low-Rank Adapters) để fine-tune, giúp tiết kiệm bộ nhớ và tài nguyên tính toán nhưng đạt độ chính xác vượt trội so với các kiến trúc LSTM/CNN truyền thống.
