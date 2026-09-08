# 📋 PLAN — Seminar: CTC Loss (Graves et al., ICML 2006) qua lăng kính CRNN OCR

> Cập nhật 08/09/2026. Chốt: **90 phút · khán giả là devs cùng team (đi sâu math) · tiếng Việt**.
> Demo dựa trên repo CRNN_OCR sẵn có (model đã train, checkpoint ở `checkpoints/`).
> Paper: `ctc_loss_icml_2006.pdf` (Graves, Fernández, Gomez, Schmidhuber — ICML 2006).

---

## PHẦN A — Bản đồ repo (mục đích từng file/folder, không gồm `.venv`)

| Đường dẫn | Mục đích | Vai trò trong seminar |
|---|---|---|
| `data/trainset/` | 26,155 ảnh captcha (Kaggle), **tên file = label** (vd `2B2847.jpeg` → `2B2847`) | Ví dụ bài toán temporal classification "real" mở đầu |
| `data/testset/` | 100 ảnh test (đang kiêm luôn validation) | Demo mục 7: chạy live + so sánh greedy vs beam |
| `eda.ipynb` | EDA dataset: xem sample, xây charset, phân bố độ dài label, kích thước ảnh | Chỉ tham khảo, không dùng trong buổi |
| `pipeline_deep_dive.ipynb` | **Asset quý nhất**: visualize nội bộ CTC trên sample `"233HP3"` — heatmap α (soft-alignment), β, gradient, greedy vs beam decode | **Demo chính** (mục 3–4 agenda: heatmap α, gradient; mục 7: decode) |
| `src/dataset.py` | `OCRDataset`: grayscale → resize 32×W, encode label thành int (index **0 dành cho blank**), `collate_fn` flatten labels + lengths | Map: định nghĩa label/vocab §2, §3 |
| `src/model.py` | `CRNN` = CNN 7 lớp VGG-style + 2 BiLSTM. Pooling stride `(2,1)` (dòng 44, 48) để nén height về 1 nhưng **giữ trục width làm trục thời gian T** | Map: network `N_w` §3.1; BLSTM §5.2 |
| `src/ctc_loss.py` | **Tim của seminar** — CTC loss viết tay từ scratch: build `extended_targets`, forward α, backward β, gradient | Map: §4.1 eq(5)–(11), §4.2 eq(12)–(16) |
| `src/train.py` | Train CRNN bằng `nn.CTCLoss` chính thức của PyTorch (baseline chuẩn) | Map: eq(12) training loop |
| `src/train_custom.py` | Train bằng `CTCLossFromScratch` — chứng minh loss tự viết cho kết quả tương đương | Bằng chứng "code hiểu đúng paper" |
| `src/utils.py` | `decode_greedy` (= best path decoding), `decode_beam_search` (≈ prefix search), `calculate_accuracy`, `plot_training_history` | Map: §3.2 decoding |
| `src/inference.py` | Load checkpoint + decode ảnh mới (beam search có sẵn nhưng chưa nối CLI) | Demo tổng hợp **cuối buổi** (mục 7 agenda) |
| `checkpoints/` | `best_model.pth` (đã train), `vocab.txt` (39 ký tự, **lẫn lowercase** `c j s u w x y`), `epoch1.pth` | Có sẵn → demo không cần train lại |
| `blogs/` | `draft.md` (bài deep-dive dài nhất), `blog_vi.md` / `blog_en.md` (bài viết CRNN+CTC) | **Tái dùng làm khung slide** (~70% nội dung sẵn) |
| `images/intro.png` | Ảnh minh họa kiến trúc | Slide mở đầu |
| `README.md` | Hướng dẫn setup / train / inference | Tham khảo lệnh |
| `ctc_loss_icml_2006.pdf` | Paper gốc | Tài liệu đọc lại theo Phần B |

### Mapping paper ↔ code (đã verify bằng fuzz-test so với `nn.CTCLoss`, khớp loss + gradient sai số ~1e-6 với `zero_infinity=True`)

| Paper | Code |
|---|---|
| eq(2) `p(π\|x) = Πₜ y_t^{π_t}` | `log_softmax(2)` — `src/train.py:53`, `src/train_custom.py:54` |
| Chèn blank vào `l'` (len `2\|l\|+1`) §4.1 | `extended_targets` — `src/ctc_loss.py:273-279` |
| eq(5)–(7) forward α + skip rule | `_compute_alpha_matrix` — `src/ctc_loss.py:6` (skip mask dòng 40–41) |
| eq(8) `p(l\|x) = αT(\|l'\|) + αT(\|l'\|−1)` | `logaddexp(final_alpha[2L], final_alpha[2L−1])` — `src/ctc_loss.py:294` |
| eq(9)–(11) backward β | `_compute_beta_matrix` — `src/ctc_loss.py:63` |
| eq(12) `O_ML = −Σ ln p(z\|x)` | `losses.append(-total_log_prob)` — `src/ctc_loss.py:297` |
| eq(15)–(16) gradient / error signal | `CTCLossFunction.backward` — `src/ctc_loss.py:314`; qua softmax: `∂O/∂u = y − posterior` |
| Rescaling `C_t` chống underflow §4.1 | Code dùng **log-domain + `logaddexp`** → điểm giảng: cách 2006 vs hiện đại |
| Best path decoding eq(4) | `decode_greedy` — `src/utils.py:10` |
| Prefix search decoding §3.2 | `decode_beam_search` — `src/utils.py:33` (thảo luận: exact nhưng exponential vs beam approximation) |

---

## PHẦN B — Lộ trình đọc lại paper theo prompt 13 mục (intuition first, depth after)

### B.0 — Mục lục paper & bảng phủ (tick ☐ khi đọc xong để không bị sót)

Paper 8 trang. Đọc **theo thứ tự B1→B13** (không theo thứ tự trang), dùng bảng này để đối chiếu phần nào của paper đã đi qua:

| ✓ | Mục paper (trang) | Nội dung chính | Mục B tương ứng |
|---|---|---|---|
| ☐ | Abstract (tr.1) | Tóm tắt: bài toán + phương pháp + kết quả TIMIT | B1 |
| ☐ | §1 Introduction (tr.1–2) | Nhược điểm HMM/CRF (3 điểm), hybrid HMM-RNN, ý tưởng CTC, temporal vs framewise | B2, B6 |
| ☐ | §2 Temporal Classification (tr.2) | Formalism: X, Z, điều kiện U ≤ T, temporal classifier `h` | B4 |
| ☐ | §2.1 Label Error Rate + eq(1) (tr.2) | LER = edit distance chuẩn hóa | B10 |
| ☐ | §3.1 + eq(2)(3) (tr.2–3) | Softmax `\|L\|+1` units, blank, path π, giả định independence, map B, `p(l\|x)` | B3, B4, B5 |
| ☐ | Fig 1 (tr.3) | Framewise vs CTC: spike vs align với segmentation | B2, B3 |
| ☐ | §3.2 + eq(4) (tr.3) | Decoding: best path, prefix search (Fig 2), heuristic chia section theo blank | B4, B9 |
| ☐ | §4 intro (tr.4) | Principle of maximum likelihood, BPTT | B5 |
| ☐ | §4.1 + eq(5)–(8) (tr.4–5) | Forward: biến `l′` chèn blank, α, skip rule, `p(l\|x) = αT(\|l′\|) + αT(\|l′\|−1)` | B4, B12 |
| ☐ | §4.1 + eq(9)–(11) (tr.5) | Backward β, điều kiện biên | B4 |
| ☐ | §4.1 rescaling `C_t`, `D_t` (tr.5) | Chống underflow; `ln p(l\|x) = Σ ln C_t` | B4 |
| ☐ | §4.2 + eq(12)–(14) (tr.5–6) | Objective `O_ML`, vai trò `α_t(s)β_t(s) / y_t^{l′_s}` | B4, B5 |
| ☐ | §4.2 + eq(15)–(16) (tr.6) | Gradient theo `y_t^k` và `u_t^k` (error signal) | B4 |
| ☐ | Fig 3 (tr.5) / Fig 2 (tr.4) / Fig 4 (tr.6) | Lattice "CAT" / prefix search tree / evolution of error signal | B4, B12 |
| ☐ | §5 intro (tr.6) | Thiết kế thí nghiệm: CTC vs HMM vs hybrid, chọn BLSTM | B7, B8 |
| ☐ | §5.1 Data (tr.6) | TIMIT, 61 phonemes, MFCC, chuẩn hóa | B10 |
| ☐ | §5.2 Setup (tr.7) | Chi tiết BLSTM + hyperparams, noise σ=0.6, baseline HMM/hybrid | B8, B9, B10 |
| ☐ | Table 1 + §5.3 Results (tr.7) | Bảng LER, phân tích significance | B7, B10 |
| ☐ | §6 Discussion (tr.7–8) | Không explicit segment, inter-label deps, hierarchical CTC, overfitting | B5, B7, B9, B11, B13 |
| ☐ | §7 Conclusions (tr.8) | Tóm tắt đóng góp | B13 |
| ☐ | References (tr.8) | Rabiner 89, Bourlard & Morgan 94, Hochreiter 97, Schuster & Paliwal 97… | B6 |

> Lưu ý phủ sóng: eq(14) nằm trong B4 (dẫn tới eq 15); Acknowledgements bỏ qua. Không còn mục nào của paper lọt ngoài bảng.

> Cách dùng: đọc theo thứ tự B1→B13. Mỗi mục ghi rõ **"Đọc:"** (mục paper + trang) và **mốc code** để đối chiếu.
> 👉 **Bản đọc + ghi chú tương tác** (checkbox tick được, ô note, mục đề highlight màu): xem `PAPER_READING_NOTES.md`.

### 1. Big picture
- **Đọc: Abstract (tr.1)** — chỉ 1 đoạn, đọc chậm từng câu.
- Paper giới thiệu **CTC**: huấn luyện RNN gán nhãn chuỗi **không cần pre-segmented data**, gộp alignment vào trong network, mọi thứ trong 1 kiến trúc (Abstract + §1 đoạn cuối).
- Câu hỏi trung tâm: input có T time-steps, output có U labels (U ≤ T), **không ai chỉ cho mình cặp nào khớp cặp nào** — học thế nào?
- Gắn với repo: captcha `2B2847.jpeg` → label `2B2847` chính là temporal classification (§2).

### 2. Motivation
- **Đọc: §1 Introduction (tr.1–2) + Fig 1 (tr.3)** — Fig 1 nên xem trước khi đọc §1 để có hình ảnh trực quan.
- 3 nhược điểm HMM/CRF (§1): (1) cần task-specific knowledge, (2) giả định dependency dễ bịaste để inference tractable, (3) HMM train generative dù bài toán là discriminative.
- RNN mạnh nhưng objective chuẩn chỉ định nghĩa **per-frame** → framewise classification cần pre-segmentation + post-processing.
- Fig 1 là bằng chứng trực quan: framewise network **bị phạt dù đoán đúng phoneme** chỉ vì lệch biên segmentation; CTC chỉ cần "spike" đúng chỗ.
- Hybrid HMM-RNN (Bourlard & Morgan 1994) kế thừa mọi nhược điểm HMM — đây là baseline mà CTC đánh bại ở §5.

### 3. Intuition
- **Đọc: §3.1 đoạn đầu (tr.2, "The crucial step…") + Fig 1 (tr.3)** — paper không có mục "intuition"; phần này tự diễn giải từ Fig 1 + mô tả blank trước eq(2). Bỏ qua công thức ở lần đọc 1.
- Analogy karaoke: biết lời bài hát nhưng không biết từng từ rơi vào nhịp nào.
- CTC **không commit vào 1 alignment** — nó cộng xác suất của MỌI path có thể (eq 3).
- Blank = "nhịp này chưa nhả ký tự mới". Toy: label `aa` trên 4 steps → các path `aa--`, `-aa-`, `a-a--` … đều collapse về `aa` qua map B.
- Trong code: blank cố định ở index 0 (`src/dataset.py:13-15`).

### 4. How it works (phần dài nhất — đi theo pipeline code)
- **Đọc: §2 (tr.2) → §3.1 eq(2)(3) (tr.2–3) → §3.2 eq(4) (tr.3) → §4.1 eq(5)–(11) + rescaling (tr.4–5) → §4.2 eq(12)–(16) (tr.5–6)**; kèm Fig 2, 3, 4. Đây là phần "ăn" gần hết phần kỹ thuật của paper.
1. Softmax mỗi time-step → `y_t^k` = P(label k tại t); output layer có `|L|+1` unit (blank) — §3.1, eq(2) ↔ `log_softmax(2)` tại `src/train.py:53`; `n_class = len(vocab) + 1` tại `src/train.py:142`.
2. Map `B: L'^T → L≤T`: bỏ blank + gộp ký tự lặp — `B(a−ab−) = B(−aa−−abb) = aab` (§3.1).
3. eq(3): `p(l|x) = Σ_{π∈B⁻¹(l)} p(π|x)` — không thể enumerate (số path mũ T) → cần DP.
4. Decoding: best path eq(4) (`decode_greedy`) vs prefix search §3.2 + Fig 2 (`decode_beam_search`).
5. Training — §4.1: extended sequence `l'` chèn blank đầu/cuối/giữa, len `2|l|+1` ↔ `extended_targets` (`ctc_loss.py:273-279`); forward α eq(5)–(7) với 3 transition stay/move/skip — skip chỉ khi `l'_s ≠ blank` và `l'_s ≠ l'_{s−2}` ↔ skip mask `ctc_loss.py:40-41`; `p(l|x) = αT(|l'|) + αT(|l'|−1)` eq(8) ↔ `ctc_loss.py:294`; backward β eq(9)–(11) ↔ `_compute_beta_matrix`; rescaling `C_t` ↔ log-domain trong code.
6. Gradient — §4.2: `α_t(s)β_t(s)` = xác suất mọi path qua symbol s tại t; eq(15)–(16) ra error signal `∂O/∂u_t^k = y_t^k − posterior` ↔ `backward()` `ctc_loss.py:314`; Fig 4: error dạng spike, tự triệt tiêu khi hội tụ.
- **Bài tập bắt buộc trước buổi**: tự vẽ lattice "CAT" như Fig 3, chạy tay eq(6)(7) cho 3 time-step đầu.

### 5. Why it works
- **Đọc: câu "Implicit in (2)…" (tr.3, dưới eq 3) + §4 mở đầu (tr.4) + §6 đoạn đầu (tr.7)**.
- Marginalize latent alignment = biến segmentation thành latent variable; objective differentiable → BPTT chuẩn (§1, §4).
- Giả định then chốt (nói thẳng với khán giả): outputs **conditionally independent given hidden state** (dưới eq 2, đảm bảo bằng việc không có feedback từ output layer) — đây là cái giá phải trả, khai thác ở mục 9.
- Inter-label dependency vẫn được **implicit** qua BiLSTM (§6).

### 6. History & evolution
- **Đọc: §1 (tr.1–2, các ref HMM/hybrid) + References (tr.8)** — skim các ref: Rabiner 1989 (HMM), Bourlard & Morgan 1994 (hybrid), Schuster & Paliwal 1997 (BRNN), Hochreiter & Schmidhuber 1997 (LSTM), Werbos 1990 (BPTT). Timeline sau 2006 là ngoài paper — kiến thức bổ sung cho slide.
- Timeline slide: HMM (Rabiner 1989) → hybrid HMM-NN (Bourlard & Morgan 1994) → framewise RNN (Graves 2005) → **CTC (2006)** → RNN-T (Graves 2012) → seq2seq + attention (2015) → Whisper (2022, không dùng CTC cho decoder).
- Mỗi bước giải quyết hạn chế nào: HMM giải alignment nhưng generative + hand-crafted; hybrid thêm NN nhưng vẫn khung HMM; CTC bỏ hẳn khung ngoài; attention giải conditional independence nhưng đánh đổi autoregressive (chậm, khó streaming).

### 7. Comparison
- **Đọc: §1 (tr.1–2) + §5 intro (tr.6) + §6 đoạn 1–2 (tr.7)** — paper so sánh trực tiếp qua thí nghiệm §5 chứ không có bảng so sánh lý thuyết; mục này tự tổng hợp.
- Bảng so sánh: framewise / HMM / hybrid / CTC / attention seq2seq.
- Khác biệt căn bản của CTC (§6): **không explicit segment**, không model inter-label dependencies, objective chỉ phụ thuộc sequence labels chứ không duration/segmentation.
- Ưu: end-to-end, decode nhanh, monotonic (tự nhiên hợp OCR/streaming). Nhược: conditional independence, không cắm LM trực tiếp vào objective.

### 8. Technical dependencies
- **Đọc: §3.1 định nghĩa `N_w` (tr.2) + §5 intro & §5.2 (tr.6–7)** — các chỗ paper khẳng định "any other architecture could have been used instead".
- Algorithm-level (DP forward-backward — dùng được cho MỌI architecture) ≠ model-level (BLSTM chỉ là backbone thay được — paper khẳng định §5).
- Training (cần α, β, gradient) ≠ inference (decoding không cần α/β — chỉ cần softmax).
- Hệ quả kiến trúc: `blank=0` trong code là convention; `T` của CRNN = W/4 do pooling stride — quyết định xem label có "vừa" input không (mục 9).

### 9. Failure modes & edge cases
- **Đọc: §3.2 đoạn cuối (tr.3, prefix search fail case) + §5.2 (tr.7, noise) + §6 đoạn cuối (tr.7–8, overfitting)** — lưu ý loss = ∞ và `zero_infinity` là kiến thức code-level, paper không nói trực tiếp.
- Target dài hơn input cho phép → không path hợp lệ → loss = ∞. **Trong code chính là lý do `zero_infinity=True`** (`src/train.py:146`) — ví dụ sống: T = W/4 = 25 với imgW=100.
- Ký tự lặp trong text (`"AA"`): bắt buộc cần blank giữa 2 ký tự giống nhau → tìm 1 ảnh captcha có `AA` để demo.
- Overfitting: paper tự nhận ML training của CTC khó generalize (§6) → họ thêm Gaussian noise σ=0.6 vào input (§5.2).
- Decode sai khi prefix search cắt sai section (§3.2 cuối); output peaked làm beam search thừa thải.
- Vocab lẫn lowercase (`c j s u w x y`) trong data — risk nhịn hoa/thường (chưa xử lý trong code, note cho phần thảo luận).

### 10. Experiments
- **Đọc: §2.1 eq(1) (tr.2) → §5.1 (tr.6) → §5.2 (tr.7) → Table 1 + §5.3 (tr.7)** — đọc đúng thứ tự này để hiểu metric trước khi xem kết quả.
- TIMIT, 61 phonemes, metric LER eq(1) = edit distance chuẩn hóa ↔ ý tưởng `full_acc`/`char_acc` trong `src/utils.py:104`.
- Table 1: HMM ctx-indep 38.85% → ctx-dep 35.21% → hybrid 33.84% → weighted hybrid 31.57% → **CTC best path 31.47%** → **CTC prefix search 30.51%**.
- Thảo luận tính fair: cùng BLSTM architecture; hybrid có thêm 183 params HMM + weighted-error heuristic; CTC không cần trick nào — thuyết phục.
- Lưu ý: đây là phoneme labeling (frame-level phoneme error) chứ không phải full ASR word error — đừng nói quá khi trình bày.

### 11. Research intuition
- **Đọc: §6 (tr.7–8, đặc biệt đoạn hierarchical CTC và "very general way of dealing with structured data")**.
- Nguyên lý transferable: **"khi alignment/latent structure không biết trước, sum over alignments thay vì commit một alignment"**.
- Cùng họ tư duy: EM, mixture models, soft attention (heatmap α trong notebook 5c chính là posterior over alignments!), và cả marginal likelihood trong Bayesian.

### 12. Toy example
- **Đọc: Fig 3 lattice "CAT" (tr.5) + Fig 2 prefix search tree (tr.4)** — chạy tay số liệu trên chính hình vẽ của paper.
- Chạy tay sample `"233HP3"` từ `pipeline_deep_dive.ipynb` trên lattice nhỏ: chỉ cho heatmap α, hỏi khán giả đọc soft-alignment.
- Hoặc toy `l = "ab"`, T = 5: liệt kê vài path, chỉ blank ở `a−b`, chạy 1 bước recursion α.
- Kết nối với Fig 3 "CAT" trong paper.

### 13. Key takeaways
- **Đọc: §6 (tr.7–8) + §7 Conclusions (tr.8)** — chốt lại xem có khớp với 5 điều cần nhớ không.
- **5 nhớ**: (1) CTC = marginalize mọi alignment qua blank + map B; (2) forward-backward tính `p(l|x)` trong O(T·|l|); (3) gradient `y − posterior` là error signal dạng spike; (4) decoding = best path (greedy) hoặc prefix/beam search; (5) conditional independence là giới hạn cốt lõi.
- **3 học tiếp**: RNN-T (streaming, không cần blank giữa repeat), seq2seq + attention (bỏ conditional independence), EM/forward-backward tổng quát.
- **1 câu**: "CTC biến bài toán gán nhãn chuỗi không có alignment thành một bài toán maximum likelihood differentiable bằng cách cộng xác suất trên mọi alignment, cho RNN tự học alignment trong lúc train."

---

## PHẦN C — Agenda 90 phút (demo dồn về sau cùng)

> Kể chuyện 2 hồi: **mục 1** mở "vòng lặp" bằng 1 ảnh captcha **tĩnh** trên slide — "đây là bài toán"; **mục 7** đóng vòng lặp bằng demo live — "đây là nó chạy". Mọi phần trước mục 7 chỉ dùng slide / hình chuẩn bị sẵn / walkthrough code, **không chạy lệnh live**.

| # | Nội dung | Thời gian | Asset |
|---|---|---|---|
| 1 | Đặt vấn đề + Big picture + Motivation + Intuition (B1–B3): chiếu **ảnh captcha tĩnh**, hỏi "tách ký tự kiểu cũ chạy được không?" → dẫn vào segmentation-free | 15' | ảnh trong `data/testset/` (chiếu tĩnh, chưa chạy) + Fig 1 + analogy karaoke + toy `aa` |
| 2 | Output representation: blank, map B, eq(2)(3) | 15' | `src/dataset.py`, `src/ctc_loss.py:273` |
| 3 | **Forward-backward**: lattice CAT, eq(5)–(11), log-domain vs rescaling | 20' | walkthrough `src/ctc_loss.py` + heatmap α (notebook 5b–5c) |
| 4 | Gradient eq(12)–(16) + Fig 4 error signal | 10' | notebook 5d + `CTCLossFunction.backward` |
| 5 | Decoding (lý thuyết): best path eq(4) vs prefix search §3.2 | 7' | `src/utils.py` — chỉ đọc code, **chưa chạy** |
| 6 | Experiments, evolution 2006 → nay, failure modes (B6–B10) | 8' | Table 1 + timeline slide |
| 7 | 🎬 **Demo tổng hợp (sau cùng)**: chạy live `inference.py` trên chính **ảnh "neo" của mục 1** → so sánh greedy vs beam trên 100 ảnh test → (còn giờ thì chỉ heatmap α tại 1 ví dụ) | 10' | `src/inference.py` + notebook 6–7 |
| 8 | Quiz + Q&A | 5'+ | bộ quiz Phần D |

Tổng: ~90'. Nếu demo mục 7 hỏng (env/checkpoint lỗi), fallback: play sẵn video terminal đã quay — nên ghi theo checklist Phần D.

## PHẦN D — Checklist chuẩn bị

1. [ ] **Fix bug `squeeze()`** — `src/train.py:45`, `src/train_custom.py:46`: `label_lengths.squeeze()` → `squeeze(-1)` (crash khi batch cuối còn đúng 1 sample — nguy hiểm nếu demo live).
2. [ ] **Tái tạo fuzz-test** `src/verify_ctc.py`: so custom loss vs `nn.CTCLoss` (lần verify trước chưa commit script) → demo "code của tôi = paper = PyTorch" rất thuyết phục. Kiểm tra: 20+ cấu hình random gồm ký tự lặp, `input_lengths` biến thiên, target dài ngắn, `zero_infinity=True`.
3. [ ] Chạy lại `pipeline_deep_dive.ipynb`, export figure: heatmap α, β, error signal, blank spikes, greedy vs beam.
4. [ ] Dựng slide theo khung `blogs/draft.md` (tiếng Việt, thuật ngữ giữ tiếng Anh).
5. [ ] Chuẩn bị quiz: "Vì sao cần blank?", "Tại sao `B(aa−−) = a` mà không phải `aa`?", "Khi nào CTC loss = ∞?", "Decode có cần α/β không?".
6. [ ] Tìm 1 ảnh captcha có ký tự lặp (vd `AA`) trong `data/trainset/` để demo case bắt buộc blank: `bash: ls data/trainset | grep -E '(.)\1'`.
7. [ ] (Tuỳ chọn) Nối `--decode greedy|beam` vào CLI `src/inference.py` để demo tiện hơn.
8. [ ] Quay sẵn **video terminal** chạy inference + so sánh greedy vs beam (fallback nếu demo live mục 7 hỏng).
9. [ ] (Tuỳ chọn) Rehearsal 1 lần bấm giờ, đặc biệt mục 3 (dài nhất, dễ lan man).

## PHẦN E — Lộ trình đi sâu code ("đi theo 1 ảnh captcha")

> 4 nguyên tắc dẫn dắt: (1) **1 ví dụ xuyên suốt** — chọn sẵn 1 ảnh (vd `2B2847`), mọi giá trị trung gian in từ chính nó, không đổi ví dụ giữa chừng; (2) **tensor shape làm xương sống** — mỗi slide code kèm bảng shape trước → sau; (3) **mỗi file = 1 câu hỏi của paper** (bảng dưới); (4) **đọc 2 lượt** — lượt 1 chỉ signature/input/output, lượt 2 mới vào chi tiết recursion. Không đọc từng dòng từ đầu đến cuối.

| Bước | Câu hỏi paper trả lời | File:dòng | Shape trước → sau | Điểm dừng hỏi khán giả |
|---|---|---|---|---|
| 1. Dữ liệu | Label biểu diễn thế nào? blank nằm đâu? | `dataset.py:13-15, 44-61` | ảnh → `[1,32,100]`; label 6 số; `collate_fn` → labels flat `[Σlen]` + lengths `[B]` | "Vì sao phải reserve index 0 cho blank?" |
| 2. Ảnh → chuỗi | `N_w` §3.1 là gì? T đến từ đâu? | `model.py:56-66` (pooling stride `(2,1)` dòng 44,48) | `[B,1,32,100]` → CNN `[B,512,1,25]` → permute `[25,B,512]` → BiLSTM → `[25,B,40]` | "T=25, U=6 — alignment đang ở đâu?" (chưa tồn tại!) |
| 3. Logits → per-step prob | eq(2) | `train.py:53` | `[25,B,40]` → `log_softmax(2)` → `[25,B,40]` | "Đây đã là p(l\|x) chưa?" — chưa, chỉ mới `p(π_t)` từng bước → tạo chỗ hở dẫn sang bước 4 |
| 4a. `l` → `l′` | §4.1 chèn blank (len `2\|l\|+1`) | `ctc_loss.py:273-279` | target 6 số → 13 states | In `−2−B−2−8−4−7−` lên slide, cho khán giả đếm |
| 4b. Forward α | eq(5)–(8), skip rule | `ctc_loss.py:6-61` (skip mask dòng 40–41) | `alpha_history [T,B,S]` qua logaddexp stay/move/skip | "Vì sao skip chỉ khi `l′_s ≠ blank` và `≠ l′_{s−2}`?" — quay lại lattice CAT, chỉ 2 case ký tự lặp |
| 4c. Loss | eq(8), eq(12) | `ctc_loss.py:287-298` | alpha cuối → `logaddexp(α[2L], α[2L−1])` → negate | "Vì sao chỉ cộng đúng 2 ô cuối?" |
| 4d. Gradient | eq(9)–(16) | `ctc_loss.py:63-253` (β) + `314-410` (backward) | `joint = α+β` → posterior → `scatter_add` → `grad = −posterior` (qua softmax thành `y − posterior`) | "Vì sao dùng log-domain thay vì rescaling `C_t` của paper?" |
| 5. Decode | eq(4) + §3.2 | `utils.py:10` (greedy), `:33` (beam) | `[25,B,40]` → strings | "Decode có cần α/β không?" — không, chỉ cần softmax output |
| 6. End-to-end | Đóng vòng | `inference.py` | ảnh → text | → nối sang Demo mục 7 agenda |

**Kỹ thuật trình bày khi walkthrough:**
- Mỗi slide code chỉ **highlight 3–4 dòng** quan trọng, phần còn lại fade/mờ đi.
- Chuyển qua lại **slide code ↔ `pipeline_deep_dive.ipynb`** để đối chiếu "dòng code ↔ con số thật" của sample `"233HP3"` (heatmap α ở 5b–5c, gradient ở 5d).
- Ngay sau bước 4d, chạy `verify_ctc.py` (Phần D item 2) — khép lại: *"code tự viết = paper = PyTorch"*.
- Nếu thiếu thời gian: bỏ chi tiết 4d (chỉ nói ý `α+β → posterior`), giữ trọn 4a–4c — forward mới là phần "ăn tiền" nhất.

---

*Lịch sử: PLAN cũ (07/09/2026) chứa kết quả review + fuzz-test verification — đã gộp toàn bộ thông tin còn giá trị vào plan này.*
