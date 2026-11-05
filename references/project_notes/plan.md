# 1. Kiến trúc NRMS gốc (Original NRMS)

Trước tiên, mình sẽ tập trung vào **kiến trúc NRMS chuẩn và cách xử lý dữ liệu MIND-small**. Chưa cần bàn đến các biến thể hay cải tiến.

NRMS (*Neural News Recommendation with Multi-Head Self-Attention*) có ba thành phần:

1. **News Encoder:** Chuyển tiêu đề bài báo thành vector.
2. **User Encoder:** Học sở thích người dùng từ các bài báo đã đọc.
3. **Click Predictor:** Tính điểm phù hợp giữa người dùng và bài báo.

## 1.1. News Encoder

**Input:** Tiêu đề bài báo (news title).

Kiến trúc:

```text
News Title
    │
    ▼
Tokenization
    │
    ▼
Word Embedding
    │
    ▼
Multi-Head Self-Attention
    │
    ▼
Additive Attention
    │
    ▼
News Representation
```

Ví dụ:

```text
"NVIDIA announces new AI chips"
                │
                ▼
        Word Embeddings
                │
                ▼
     Multi-Head Self-Attention
                │
                ▼
        Additive Attention
                │
                ▼
        News Vector (400)
```

Trong NRMS gốc, word embeddings được khởi tạo bằng pretrained GloVe embeddings.

**Multi-Head Self-Attention** học quan hệ giữa các từ trong tiêu đề:

\[
H=\operatorname{MultiHeadAttention}(X,X,X)
\]

Sau đó **Additive Attention** tổng hợp các token thành một news vector:

\[
e_n=\sum_{i=1}^{L}\alpha_i h_i
\]

Trong đó \(\alpha_i\) là attention weight của token thứ \(i\).

Lưu ý: NRMS gốc không sử dụng category embedding, Sentence-BERT hay Knowledge Graph trong News Encoder.

## 1.2. User Encoder

**Input:** Lịch sử các bài báo người dùng đã click.

Ví dụ:

```text
User History
    │
    ├── News A ──► News Encoder ──► Vector A
    ├── News B ──► News Encoder ──► Vector B
    ├── News C ──► News Encoder ──► Vector C
    └── News D ──► News Encoder ──► Vector D
                                    │
                                    ▼
                         Multi-Head Self-Attention
                                    │
                                    ▼
                            Additive Attention
                                    │
                                    ▼
                             User Vector (400)
```

Điểm đặc biệt là **News Encoder được chia sẻ trọng số** giữa lịch sử đọc và candidate articles.

User Encoder dùng self-attention để học mối liên hệ giữa những bài báo đã đọc:

\[
Z=\operatorname{MultiHeadAttention}(E_H,E_H,E_H)
\]

Sau đó additive attention tổng hợp lịch sử:

\[
u=\sum_{i=1}^{P}\beta_i z_i
\]

Trong đó:

- \(P\): Số bài báo trong lịch sử.
- \(z_i\): Biểu diễn của bài báo thứ \(i\) sau self-attention.
- \(\beta_i\): Attention weight.

Kết quả là một vector đại diện cho sở thích người dùng.

## 1.3. Click Predictor

NRMS sử dụng **dot product**, không phải MLP hay sigmoid classifier.

Với user vector \(u\) và candidate news vector \(e_c\):

\[
\boxed{s(u,c)=u^Te_c}
\]

Trong quá trình huấn luyện, NRMS sử dụng negative sampling và softmax:

\[
P(c^+\mid u,C)=
\frac{\exp(s(u,c^+))}
{\sum_{c\in C}\exp(s(u,c))}
\]

Loss:

\[
\boxed{\mathcal{L}=-\log P(c^+\mid u,C)}
\]

Ở đây, \(C\) là nhóm gồm một positive article và các sampled negative articles.

**Phân biệt:** Dot product tạo ra ranking score. Softmax được sử dụng trong training để học cách phân biệt positive và negative candidates. Không nên xem softmax score này là CTR đã được hiệu chuẩn.

---

Các thành phần trên được xác nhận trong paper gốc của Wu et al. (EMNLP 2019). [ACL Anthology](https://aclanthology.org/D19-1671/?utm_source=chatgpt.com)

**Đính chính một chi tiết:** Vector 400 chiều ở ví dụ phía trên không phải cấu hình của NRMS gốc. Paper sử dụng 16 attention heads, mỗi head 16 chiều, nên news/user representation có **256 chiều**. Word embedding có 300 chiều, attention query 200 chiều và negative sampling ratio là 4. [ACL Anthology](https://aclanthology.org/D19-1671.pdf?utm_source=chatgpt.com)

# 2. Sửa Data Pipeline theo góp ý giảng viên

Đây là những thay đổi cần thiết cho toàn nhóm, không chỉ riêng NRMS.

| Vấn đề | Cách xử lý | Áp dụng cho NRMS |
|---|---|---|
| `slate_length` | Loại khỏi input features | Không sử dụng |
| `article_recency` | Loại khỏi mô hình chính vì thiếu publication time đáng tin cậy | Không sử dụng |
| `inactivity_gap` missing 41% | Tạo missing indicator và trạng thái UNKNOWN | NRMS gốc không cần |
| Temporal split | Thống nhất split và lưu cố định | Bắt buộc |
| Historical CTR leakage | Chỉ tính từ interactions trong quá khứ | NRMS gốc không cần |
| Unobserved articles | Không gán nhãn negative | Bắt buộc |
| Dynamic rotation | Tách ranking evaluation khỏi rotation simulation | Đánh giá riêng |

Ba vấn đề đầu và temporal split xuất phát trực tiếp từ các nhận xét trong `advise.md`. advise

## 2.1. Loại bỏ `slate_length`

Giảng viên nhận xét rằng slate length giống nhau đối với tất cả candidates trong cùng impression.

Ngoài ra, MIND-small của nhóm không có zero-click impressions, khiến exposure-level CTR giảm theo slate length một phần do cơ chế chọn mẫu.

**Cách sửa:**

- Xóa `slate_length` khỏi danh sách predictive features.
- Vẫn giữ slate length trong dữ liệu để padding, batching và phân tích thống kê.
- Không dùng slate length làm input cho MLP.

NRMS gốc vốn không cần feature này.

## 2.2. Loại bỏ `article_recency`

Trong Interim Report, nhóm tính recency dựa trên thời điểm bài báo xuất hiện lần đầu trong logs.

Nhưng:

\[
t_{\text{first observed}}\neq t_{\text{published}}
\]

Ví dụ, một bài báo xuất bản ngày 7/11 nhưng xuất hiện lần đầu trong dữ liệu ngày 9/11 sẽ bị nhận diện nhầm là bài báo mới.

**Cách sửa:**

Loại `article_recency` khỏi mô hình chính.

Nếu vẫn muốn nghiên cứu freshness, có thể sử dụng `time_since_first_observed` như một proxy, nhưng phải ghi rõ hạn chế và đánh giá riêng.

Không được gọi proxy này là publication recency.

## 2.3. Xử lý `inactivity_gap`

Không nên thay toàn bộ missing values bằng median mà không có indicator.

Đề xuất:

```python
df["inactivity_missing"] = (
    df["inactivity_gap"].isna().astype(int)
)

df["inactivity_gap_filled"] = (
    df["inactivity_gap"].fillna(0)
)
```

Đối với MLP, sử dụng đồng thời hai features:

```text
inactivity_gap_filled
inactivity_missing
```

Với NRMS gốc, không cần đưa hai features này vào.

Lưu ý rằng missing inactivity không đồng nghĩa với user cold-start: một người dùng có thể chưa có impression trước đó trong logs nhưng vẫn có lịch sử đọc.

## 2.4. Thống nhất Temporal Split

Interim Report hiện ghi:

- Train: Days 1–5.
- Validation: Day 6.
- Test: Day 7.

Tuy nhiên, `advise.md` có ghi chú về việc chỉnh lại mô tả 4-1-2 và 5-1-1.

**Cần xác nhận với nhóm split nào thực sự được tạo trong code.**

Sau đó, cả ba nhóm UBCF + CBF, MLP và NRMS phải sử dụng cùng một danh sách impression IDs cho train/validation/test.

Không được tự random split.

## 2.5. Negative Sampling

Đây là phần quan trọng nhất đối với NRMS.

Một impression:

```text
N100-0 N101-1 N102-0 N103-0 N104-0 N105-0
```

Tạo training sample:

```text
Positive: N101

Negatives:
N100
N102
N103
N104
```

Không lấy negative từ những articles chưa được hiển thị.

**Training:** 1 positive + 4 sampled negatives.

**Evaluation:** Sử dụng toàn bộ candidates và labels của từng impression.

Với impressions có nhiều positive clicks, tạo các training samples tương ứng và giữ nguyên impression grouping khi đánh giá.

---

# 3. Dataset NRMS thực sự cần

Mặc dù Interim Report có nhiều engineered features, NRMS gốc chỉ cần một phần nhỏ.

| Dữ liệu | Trường | Mục đích |
|---|---|---|
| `news.tsv` | `news_id` | Định danh bài báo |
| `news.tsv` | `title` | Input cho News Encoder |
| `behaviors.tsv` | `impression_id` | Nhóm candidates khi evaluation |
| `behaviors.tsv` | `history` | Input cho User Encoder |
| `behaviors.tsv` | `impressions` | Candidate articles + click labels |
| `behaviors.tsv` | `time` | Temporal split |

Các trường `category`, `abstract`, `entity_embedding`, `historical_ctr` không cần thiết cho Original NRMS.

### Dữ liệu sau preprocessing

```python
{
    "history": [
        [12, 45, 78, ...],
        [35, 90, 21, ...]
    ],
    "candidates": [
        [23, 56, 89, ...],
        [11, 42, 63, ...]
    ],
    "labels": [
        [0, 1, 0],
        [1, 0, 0]
    ]
}
```

Ở đây, các số đại diện cho token IDs hoặc news IDs tùy tầng xử lý. Nên giữ hai loại ID này riêng biệt trong code.

---

# 4. Cấu hình NRMS nên chạy

Để bắt đầu gần với paper gốc:

| Hyperparameter | Giá trị |
|---|---|
| Word embedding | GloVe |
| Word embedding dimension | 300 |
| Attention heads | 16 |
| Head dimension | 16 |
| News representation | 256 |
| User representation | 256 |
| Additive attention query | 200 |
| Negative sampling ratio | 4 |
| Dropout | 0.2 |
| Batch size | 64 |
| Optimizer | Adam |

Các thông số trên dựa trên thiết lập thực nghiệm được mô tả trong NRMS gốc. [aclanthology.org](https://aclanthology.org/D19-1671.pdf?utm_source=chatgpt.com)

Riêng learning rate, maximum title length, history truncation và early stopping, bạn nên xác định rõ trong config rồi lựa chọn dựa trên validation set.

**Không tuning hyperparameters bằng test set.**

---

# 5. Các kết quả bắt buộc

Vì nhóm còn hai hướng mô hình khác, bạn chỉ cần tập trung vào:

| Experiment | Output |
|---|---|
| Original NRMS | AUC, MRR, NDCG@5, NDCG@10 |
| Multiple seeds | Mean ± Std |
| User cold-start analysis | Ranking performance theo history length |
| Temporal analysis | NDCG@5 theo khung giờ |

Nếu còn thời gian, bổ sung attention ablation.

Chưa cần xây dựng NRMS + Category hay NRMS + Temporal.

---

## Kết luận

Trước mắt, bạn và Tan nên tập trung vào đúng ba việc:

**1. Chuẩn hóa MIND-small:** Dùng temporal split chung, xử lý history đúng thời điểm và negative sampling đúng impression.

**2. Reproduce Original NRMS:** GloVe → News Self-Attention → News Additive Attention → User Self-Attention → User Additive Attention → Dot Product.

**3. Benchmark:** AUC, MRR, NDCG@5/10, cold-start analysis và temporal analysis.

Sau khi ba phần này chạy ổn định mới quyết định có cần biến thể cải tiến hay không.

**Bước tiếp theo hợp lý nhất là đi sâu vào Data Pipeline:** cách đọc `news.tsv`, `behaviors.tsv`, xây dựng vocabulary, history, negative sampling và PyTorch DataLoader đúng với kiến trúc NRMS.