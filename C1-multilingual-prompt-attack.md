# 2026 — Challenge 1 (ĐỀ THẬT): Multilingual Prompt-Attack Classification

Trang đề: https://www.kaggle.com/competitions/rmit-hackathon-2026 · đọc trực tiếp lúc thi, 22/10/2026
Host: **Bowen** (SkywardAI). Tên cuộc thi: *"Security x Generative AI x Low-Resource Languages"*.
⏰ Mở cách đây 3 ngày, **đóng sau khoảng 9 tiếng kể từ 22/10 buổi sáng** — tức khớp với hạn 18:00 ngày 22/10 trong booklet. 242 người đăng ký, 67 đã nộp, 37 đội, 82 lượt nộp tính đến lúc đọc.

## 1. Đề là gì?
Đóng vai **AI Protector**: cho một câu prompt bằng **tiếng Bồ Đào Nha, Pháp, Ý hoặc Anh**, phân loại vào đúng 1 trong 3 nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `BENIGN` | yêu cầu bình thường, an toàn |
| `JAILBREAK` | cố thuyết phục model bỏ qua luật an toàn |
| `PROMPT_INJECTION` | cố chiếm quyền, ghi đè chỉ dẫn gốc của model |

Lý do chọn Bồ Đào Nha/Pháp/Ý: đây là ngôn ngữ **nhiều tài nguyên cho NLP nói chung nhưng ít tài nguyên cho an toàn AI** — hầu hết dataset phòng thủ prompt injection chỉ làm bằng tiếng Anh, nên bộ lọc an toàn hay bị vượt qua khi đổi ngôn ngữ. Đúng chủ đề *Low-Resource Languages* của năm nay.

## 2. Dữ liệu
| File | Số dòng | Cột |
|---|---|---|
| `train.csv` | 14.000 | `id, text, language, label` |
| `test.csv` | 6.000 | `id, text, language` |
| `sample_submission.csv` | 6.000 | `id, label` |

- `text`: 10–18.550 ký tự, **trung vị 294 ký tự**.
- `language`: `pt` 45% · `fr` 25% · `it` 20% · `en` 10%.
- **Mất cân bằng nhãn mạnh:** `BENIGN` 62% · `JAILBREAK` 31% · `PROMPT_INJECTION` 7%. Đoán toàn bộ là `BENIGN` chỉ được khoảng **0.26 điểm**, đoán ngẫu nhiên khoảng 0.33 — rõ ràng đề cố ý bẫy baseline ngây thơ.
- License: CC BY-NC-SA 4.0.

```python
import pandas as pd
train = pd.read_csv("train.csv")   # có label
test  = pd.read_csv("test.csv")    # không có label — cần dự đoán
sub   = pd.read_csv("sample_submission.csv")
sub["label"] = my_predictions      # BENIGN / JAILBREAK / PROMPT_INJECTION
sub.to_csv("submission.csv", index=False)
```

## 3. ⚠️ Bắt buộc: chỉ được chọn 1 trong 10 model có sẵn
Không được dùng backbone pretrained nào khác ngoài danh sách này (cấp sẵn dưới dạng Kaggle Dataset, cũng có trong tab **Models**):

| Model | Loại | Gợi ý dùng khi nào |
|---|---|---|
| `distilbert/distilbert-base-multilingual-cased` | encoder | baseline nhanh nhất |
| `jhu-clsp/mmBERT-small` | encoder | nhanh, đa ngôn ngữ |
| `jhu-clsp/mmBERT-base` | encoder | đa ngôn ngữ, mạnh |
| `FacebookAI/xlm-roberta-base` | encoder | đa ngôn ngữ, mạnh |
| `FacebookAI/xlm-roberta-large` | encoder | encoder tốt nhất, cần nhiều VRAM |
| `google/rembert` | encoder | đa ngôn ngữ (**dùng batch size 8**) |
| `protectai/deberta-v3-base-prompt-injection-v2` | encoder (đã tinh chỉnh an toàn) | đã biết sẵn prompt injection |
| `patronus-studio/wolf-defender-prompt-injection` | encoder (đã tinh chỉnh an toàn) | đã biết sẵn prompt injection |
| `ibm-granite/granite-3.3-2b-instruct` | LLM | fine-tune LLM / LoRA |
| `mistralai/Mistral-7B-Instruct-v0.3` | LLM | fine-tune LLM / LoRA |

**Phải ghi rõ model đã chọn ở đầu notebook.**

Đường dẫn Kaggle Dataset dạng: `/kaggle/input/models/aisuko/<model-slug>/transformers/default/1` (xem ví dụ code bên dưới dùng `xlm-roberta-base`).

## 4. Cách làm — 3 bước
1. **Chọn model** trong danh sách trên, nêu rõ ở đầu notebook.
2. **Fine-tune trên `train.csv`.**
   - Hướng encoder: `AutoModelForSequenceClassification` với `num_labels=3`.
   - Hướng LLM: LoRA/QLoRA, hoặc gắn thêm classification head.
   - **Bắt buộc xử lý mất cân bằng lớp**: class weights, focal loss, hoặc resampling.
   - Có thể dùng cột `language` làm feature phụ (BTC xác nhận hiệu năng khác nhau theo từng ngôn ngữ).
3. **Dự đoán `test.csv` rồi nộp CSV** đúng định dạng `id,label`, đủ **6.000 dòng + header**.

```
id,label
t000006,BENIGN
t000007,PROMPT_INJECTION
t000008,JAILBREAK
```

## 5. Chấm điểm
- Metric: **Macro-averaged F1** — tính F1 riêng từng lớp rồi lấy trung bình đều (không theo tỉ lệ), nên lớp hiếm `PROMPT_INJECTION` (7%) có **trọng số ngang bằng** lớp `BENIGN` (62%).
- Mốc tham chiếu: toàn `BENIGN` ≈ 0.26 · ngẫu nhiên ≈ 0.33 · hoàn hảo = 1.0.
- ✅ **Có điểm Challenge 1 ngay khi nộp 1 bài hợp lệ đầu tiên**, sau đó cứ cải thiện điểm trên leaderboard.
- ⚠️ **Chỉ có 6 lượt nộp** — ít hơn hẳn năm 2025 (10 lượt). Phải tự validate (hold-out hoặc k-fold) trước khi nộp.

## 6. Quy tắc đặt tên notebook (khác năm 2025!)
```
<university>_<campus>_<team>_C<challenge>        với <challenge> ∈ {1}
```
- `<university>`: chỉ chữ cái, ví dụ `RMIT`, `Taylor`
- `<campus>`: chỉ chữ cái, một trong `SGS`, `HNI`, `MEL`, `MYS`
- `<team>`: chữ, số, `_`, `-`

Ví dụ hợp lệ: `RMIT_SGS_a_C1`, `RMIT_HNI_beautiful_C1`, `Taylor_MYS_sunny_C1`, `RMIT_MEL_beautiful_C1`.

## 7. Nền tảng
Năm nay **lai giữa 2 nền tảng**:
- **Challenge 1** (duy nhất có trên Kaggle) → chấm tự động bằng leaderboard, macro-F1.
- **Các challenge khác** (nếu có — trang Overview chỉ nói *"handled by RMIT RACE Cloud"**, không mô tả thêm) → làm và nộp qua RMIT RACE Cloud, không qua Kaggle. Cần hỏi trực tiếp BTC/booklet tại chỗ hoặc theo dõi website https://rmit-hackathon.com/ để biết chi tiết, vì trang Kaggle không mô tả.

## 8. Code mẫu chính thức của BTC (từ bài FAQ, rút gọn)
File gốc do BTC đăng trong Discussion (**"Model to Data: Zero to Hero in 5 Minutes"**), dùng 2×T4:

```python
import torch
print(torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset

MODEL = "/kaggle/input/models/aisuko/xlm-roberta-base/transformers/default/1"
DATA  = "/kaggle/input/competitions/rmit-hackathon-2026"

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
L2I = {l: i for i, l in enumerate(LABELS)}

MAXLEN, EPOCHS, LR = 256, 3, 2e-5
BS_PER_GPU = 16              # 2 x T4 -> effective train batch 32
NGPU = max(torch.cuda.device_count(), 1)

train = pd.read_csv(f"{DATA}/train.csv")
test  = pd.read_csv(f"{DATA}/test.csv")

tr, va = train_test_split(train, test_size=0.1, stratify=train.label, random_state=42)

tok = AutoTokenizer.from_pretrained(MODEL)

def mk(df, labelled=True):
    d = df[["text"]].copy()
    if labelled:
        d["labels"] = df.label.map(L2I).values
    return (Dataset.from_pandas(d, preserve_index=False)
            .map(lambda b: tok(b["text"], truncation=True, max_length=MAXLEN),
                 batched=True, remove_columns=["text"]))

ds_tr, ds_va, ds_te = mk(tr), mk(va), mk(test, labelled=False)
print(len(ds_tr), len(ds_va), len(ds_te))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS),
    id2label={i: l for l, i in L2I.items()}, label2id=L2I
)
```
BTC dừng ví dụ ở đây (chưa có phần Trainer/predict). Phần tiếp theo ở mục 9 là gợi ý mình viết thêm để hoàn chỉnh.

## 9. Hoàn thiện pipeline (train → class weight → predict → nộp)

```python
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

# --- Xử lý mất cân bằng lớp bằng class weights ---
class_w = compute_class_weight("balanced", classes=np.arange(3), y=tr.label.map(L2I))
class_w = torch.tensor(class_w, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_w.to(outputs.logits.device))(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"f1_macro": f1_score(p.label_ids, preds, average="macro")}

args = TrainingArguments(
    output_dir="out", num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=BS_PER_GPU, per_device_eval_batch_size=BS_PER_GPU,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="f1_macro", logging_steps=50, fp16=True, report_to="none",
)

trainer = WeightedTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                          tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
                          compute_metrics=compute_metrics)
trainer.train()
print(classification_report(va.label.map(L2I), trainer.predict(ds_va).predictions.argmax(-1),
                            target_names=LABELS))

# --- Dự đoán test và nộp ---
test_preds = trainer.predict(ds_te).predictions.argmax(-1)
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
sub["label"] = [LABELS[i] for i in test_preds]
sub.to_csv("submission.csv", index=False)
```

## 10. Chiến thuật trong ~7 tiếng làm bài
1. **10 phút đầu:** chạy baseline `distilbert-base-multilingual-cased`, nộp ngay 1 bài hợp lệ để chắc chắn có điểm Challenge 1.
2. **Đo F1-macro cục bộ** trên tập hold-out 10% (stratify theo `label`) **trước khi** dùng lượt nộp — chỉ còn 6 lượt.
3. **Ưu tiên xử lý lớp hiếm `PROMPT_INJECTION`** (7%): oversampling, class weight hoặc focal loss; theo dõi riêng F1 của lớp này (`classification_report`).
4. Nếu đủ thời gian và VRAM: thử `xlm-roberta-large` hoặc 2 model safety-tuned (`deberta-v3-prompt-injection-v2`, `wolf-defender-prompt-injection`) — các model này **đã học sẵn khái niệm prompt injection**, có thể hội tụ nhanh hơn dù hiện tại chủ yếu huấn luyện trên tiếng Anh.
5. **Ensemble theo xác suất** (trung bình softmax) của 2 model khác kiến trúc thường tăng F1-macro ổn định hơn 1 model đơn.
6. Theo dõi riêng **F1 theo từng ngôn ngữ** (`pt/fr/it/en`) để biết ngôn ngữ nào đang yếu, từ đó augment dữ liệu hướng đúng chỗ (ví dụ dịch chéo prompt tiếng Anh/Việt jailbreak nổi tiếng sang pt/fr/it bằng chính LLM).
7. Đặt tên notebook đúng mẫu `<university>_<campus>_<team>_C1` **ngay từ đầu**, tránh quên lúc gấp giờ.
