# 📖 PAPER READING NOTES — CTC Loss (Graves et al., ICML 2006)

> File **đọc + ghi chú** riêng rút ra từ Phần B của `PLAN.md`. Đọc theo thứ tự **B1 → B13** (không theo thứ tự trang).
> Cách dùng: tick checkbox `- [ ]` khi đọc xong phần đó → viết thẳng vào ô **✍️ Ghi chú của tôi**.
>
> **Màu theo phase:** 🟨 Context (B1–B3) · 🟥 Core technical (B4–B5) · 🟦 So sánh & bối cảnh (B6–B9) · 🟩 Bằng chứng & tổng hợp (B10–B13)
>
> **Quy ước trong từng mục:**
> - 🔢 **Key point đánh số** — diễn giải của tôi, không có trong paper.
> - 🟣 Khối `📜 PAPER · <vị trí>` — **trích dẫn nguyên văn** từ paper, kèm trang; đặt **ngay dưới key point mà nó chứng minh**.
> - `↳` dòng xám — giải nghĩa quote liên hệ về key point.
>
> ⚠️ *Highlight màu dùng inline HTML — hiện đầy đủ trong VS Code preview / Typora / Obsidian. Trên GitHub, style bị strip nhưng heading vẫn hiện text + emoji màu.*

---

## 📌 Bảng phủ paper (tick ☐ → ☑ khi đọc xong để không bị sót)

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

> eq(14) nằm trong B4 (dẫn tới eq 15); Acknowledgements bỏ qua. Hết bảng = hết paper.

---

<h3>🟨 <span style="background-color:#FFF3CD; color:#856404; padding:3px 12px; border-radius:6px; border:1px solid #FFEEBA">B1 · BIG PICTURE</span> <span style="color:#888; font-size:0.85em">— Đọc: Abstract (tr.1)</span></h3>

**Checklist đọc:**
- [ ] Abstract (tr.1) — chỉ 1 đoạn, đọc chậm từng câu

**Key points:**

**1️⃣ CTC = huấn luyện RNN gán nhãn chuỗi không cần pre-segmented data, mọi thứ trong 1 kiến trúc.**

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · Abstract (tr.1) — câu kết Abstract</span>
>
> *"This paper presents a novel method for training RNNs to label unsegmented sequences directly, thereby solving both problems."*
>
> <span style="color:#777">↳ "both problems" = 2 vấn đề nêu ngay câu trước đó: pre-segmented training data + post-processing output.</span>

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · §1 (tr.2, đoạn cuối cột phải) — câu mở đầu đoạn</span>
>
> *"This paper presents a novel method for labelling sequence data with RNNs that removes the need for pre-segmented training data and post-processed outputs, and models all aspects of the sequence within a single network architecture."*
>
> <span style="color:#777">↳ nguồn của ý "mọi thứ trong 1 kiến trúc".</span>

**2️⃣ Alignment được gộp vào trong network — không phải HMM align hộ từ bên ngoài.**

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · §1 (tr.2, cùng đoạn) — câu "basic idea"</span>
>
> *"The basic idea is to interpret the network outputs as a probability distribution over all possible label sequences, conditioned on a given input sequence."*
>
> <span style="color:#777">↳ alignment được marginalize bên trong mô hình qua phân phối xác suất — đi sâu ở §3.1 (eq 2–3), mục B4.</span>

🧭 **Trước CTC, bài toán này được xử lý thế nào?** (nền để hiểu 4️⃣)
- **Cách cũ 1 — cắt/gắn thủ công:** người gắn label cho **từng frame** rồi train framewise (chính là "Kiểu 1" ở 4️⃣ dưới).
- **Cách cũ 2 — hybrid HMM-RNN:** HMM tự align hộ + post-processing output (§1, tr.1–2 — sẽ đọc kỹ ở B2).
- **CTC:** bỏ cả hai — network **tự học alignment** trong lúc train.

**3️⃣ Câu hỏi trung tâm:** input T time-steps, output U labels (U ≤ T), **không ai chỉ cho mình cặp nào khớp cặp nào** — học thế nào?
*(Chỉ là intuition rút ra từ Abstract, chưa cần đọc gì thêm. Bản formal hóa chính thức của câu hỏi này nằm ở §2 Temporal Classification (tr.2) — sẽ đọc kỹ ở **B4**, không phải bây giờ.)*

**4️⃣ Gắn với repo — so sánh 2 kiểu data khi train:**

> *Thuật ngữ: **cột = time-step**. ⚠️ Lưu ý phạm vi: **paper 2006 chỉ nói RNN**, với input **đã là chuỗi 1D** (speech frames — mỗi frame là 1 time-step). Còn repo mình dùng **CRNN (Shi et al. 2015)** — bản mở rộng **sau** paper: **CNN** phụ trách biến ảnh 2D thành chuỗi 1D (các dải dọc, imgW=100 → T=25 time-steps), phần **RNN + CTC** phía sau chạy đúng như paper. Khi giảng: nói rõ "paper = RNN + CTC; CNN chỉ là phần chuẩn bị input của mình".*

**Kiểu 1 — Có label từng cột (segmented) — cách truyền thống TRƯỚC CTC:**

```
Ảnh "AB" (7 cột minh họa):
 ┌───┬───┬───┬───┬───┬───┬───┐
 │ A │ A │ A │ · │ B │ B │ B │  ← nội dung ảnh từng dải dọc (· = nền trống)
 ├───┼───┼───┼───┼───┼───┼───┤
 │ A │ A │ A │ – │ B │ B │ B │  ← label ĐÃ CÓ SẴN cho từng cột
 └───┴───┴───┴───┴───┴───┴───┘
Mỗi cột có đáp án riêng ⇒ train framewise: softmax + CE từng cột.
```

**Kiểu 2 — Chỉ có chuỗi cuối (unsegmented) → PHẢI dùng CTC:**

```
Captcha "2B2847" (25 cột, rút gọn):
 ┌───┬───┬───┬───┬───┬───┬─ … ─┐
 │ ? │ 2 │ ? │ ? │ B │ ? │  7  │  ← model có thể đoán, nhưng KHÔNG có đáp án đúng/sai từng cột
 └───┴───┴───┴───┴───┴───┴─ … ─┘
             ↓ thứ duy nhất biết được ↓
      label = "2B2847"   (6 ký tự, KHÔNG kèm vị trí)
Ngoài ra chữ/số nét đậm nhạt, dính nhau… ⇒ càng không thể gắn cột thủ công.
⇒ CTC: thử MỌI cách ghép 25 cột → "2B2847", cộng xác suất tất cả (mục B4).
```

→ Captcha trong repo thuộc kiểu 2 ⇒ đây chính là temporal classification (§2), lý do tồn tại của CTC loss.

> [!TIP]
> Sau khi đọc xong Abstract, thử **tự trả lời trước khi đọc tiếp**: "Nếu tôi phải train mạng mà không biết mỗi frame thuộc chữ nào, tôi làm sao?"

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟨 <span style="background-color:#FFF3CD; color:#856404; padding:3px 12px; border-radius:6px; border:1px solid #FFEEBA">B2 · MOTIVATION</span> <span style="color:#888; font-size:0.85em">— Đọc: §1 Introduction (tr.1–2) + Fig 1 (tr.3)</span></h3>

**Checklist đọc:**
- [ ] Xem **Fig 1** trước (tr.3) — có hình ảnh trực quan rồi mới đọc §1 dễ hơn
- [ ] §1 (tr.1–2): 3 nhược điểm HMM/CRF, giới hạn framewise RNN, hybrid HMM-RNN

**Key points:**

🧭 **Mạch của §1** (giữ mạch này khi giảng): HMM/CRF chủ đạo nhưng có 3 nhược điểm (1️⃣) → RNN khắc phục được cả 3 (2️⃣) → mà RNN lại không dùng trực tiếp được (3️⃣) → workaround hybrid cũng chưa ổn (4️⃣) → Fig 1 trực quan hóa vấn đề (5️⃣). **CTC = lời giải cho đúng mạch này.**

**1️⃣ Trước CTC: HMM/CRF là framework chủ đạo cho sequence labelling — nhưng có 3 nhược điểm.**

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · §1 (tr.1, cột phải)</span>
>
> *"While these approaches have proved successful for many problems, they have several drawbacks: (1) they usually require a significant amount of task specific knowledge, e.g. to design the state models for HMMs, or choose the input features for CRFs; (2) they require explicit (and often questionable) dependency assumptions to make inference tractable, e.g. the assumption that observations are independent for HMMs; (3) for standard HMMs, training is generative, even though sequence labelling is discriminative."*
>
> <span style="color:#777">↳ nhược điểm (3) buồn cười nhất: bài toán là discriminative mà HMM lại train generative — "xài sai công cụ".</span>

**2️⃣ RNN khắc phục được cả 3 nhược điểm đó.**

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · §1 (tr.1–2, chuyển cột)</span>
>
> *"Recurrent neural networks (RNNs), on the other hand, require no prior knowledge of the data, beyond the choice of input and output representation. They can be trained discriminatively, and their internal state provides a powerful, general mechanism for modelling time series."*
>
> <span style="color:#777">↳ "on the other hand" = đối chiếu thẳng từng điểm với 1️⃣: không cần task-specific knowledge, train discriminative được.</span>

**3️⃣ Nhưng có 1 chướng ngại: objective chuẩn của NN định nghĩa per-frame → phải pre-segment + post-process.**

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · §1 (tr.2, đầu cột trái)</span>
>
> *"The problem is that the standard neural network objective functions are defined separately for each point in the training sequence; in other words, RNNs can only be trained to make a series of independent label classifications. This means that the training data must be pre-segmented, and that the network outputs must be post-processed to give the final label sequence."*
>
> <span style="color:#777">↳ đây chính là "both problems" trong Abstract đã thấy ở B1 — không phải ngẫu nhiên, §1 giải thích chi tiết 2 vấn đề đó.</span>

**4️⃣ Workaround thời điểm đó: hybrid HMM-RNN — nhưng kế thừa nhược điểm HMM.**

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · §1 (tr.1–2, cuối phần hybrid)</span>
>
> *"However, as well as inheriting the aforementioned drawbacks of HMMs, hybrid systems do not exploit the full potential of RNNs for sequence modelling."*
>
> <span style="color:#777">↳ hybrid = HMM align hộ + NN phân loại cục bộ; "aforementioned drawbacks" = đúng 3 nhược điểm ở 1️⃣. Đây là baseline mà CTC đánh bại ở §5 (B10).</span>

**5️⃣ Bằng chứng trực quan (Fig 1): framewise network bị phạt oan.**

> 📜 <span style="background-color:#EDE7F6; color:#5E35B1; padding:2px 8px; border-radius:4px; font-weight:bold">PAPER · Fig 1 caption (tr.3)</span>
>
> *"The framewise network receives an error for misaligning the segment boundaries, even if it predicts the correct phoneme (e.g. 'dh')."*
>
> <span style="color:#777">↳ đoán ĐÚNG phoneme vẫn bị phạt chỉ vì lệch biên segmentation — lệch biên là noise của label, không phải lỗi của model. CTC chỉ cần "spike" đúng chỗ, không cần đúng biên.</span>

![Fig 1 — Framewise vs CTC networks classifying a speech signal](images/Fig1.png)

**📖 Cách đọc Fig 1 — 3 panel, chung 1 trục thời gian (trái → phải = câu "the sound of"):**

```
Panel 1 · Waveform  : âm thanh thô. Các vạch dọc đứt = biên segmentation THỦ CÔNG
                      (người gắn: dh | ax | s | aw | n | dcl | d | ix | v).
Panel 2 · Framewise : mỗi đường màu = xác suất 1 phoneme theo thời gian.
                      Các "gò" RỘNG, cố khớp vào các vạch dọc.
Panel 3 · CTC       : các "spike" NHỌN + khoảng giữa là blank
                      (đường đứt nét gần 1 = "đang không nhả ký tự nào").
```

**Ý đồ của figure — authors muốn chứng minh 4 điều:**

1. **Đối chiếu 2 kiểu output trên cùng 1 audio**: framewise cho "gò" rộng mềm (mỗi phoneme chiếm 1 khoảng), CTC cho spike sắc ném tại 1 điểm. Khác biệt này đến từ **cách train**, không phải kiến trúc.
2. **Framewise bị phạt oan** (ý chính): vì label train gắn theo biên thủ công, model phải khớp CẢ BIÊN — mà biên giữa 2 phoneme vốn mơ hồ (âm chuyển tiếp dần). Đoán đúng 'dh' nhưng hump lệch vạch 1 chút → vẫn ăn error. CTC không có label per-frame nên không bị ràng buộc này.
3. **Blank hoạt động đúng thiết kế**: nhìn panel CTC — giữa các spike, blank ≈ 1. Blank làm 2 việc: (a) tách các label cạnh nhau, (b) phân biệt ký tự lặp (`aa` ≠ `a`) — nền cho map B ở B3/B4.
4. **"Follow the spikes" — đọc label trực tiếp**: theo thứ tự spike trong panel CTC là ra `dh ax s aw n dcl d ix v`, không cần post-processing. Ngược lại panel framewise các gò chồng lấn, phải xử lý thêm mới ra chuỗi.

> [!TIP]
> Chi tiết dễ bỏ sót: quanh "of" thấy **spike kép `dcl d`** — 2 phoneme luôn đi cùng nhau nên CTC học dự đoán chúng thành 1 cú spike đôi. Đây là bằng chứng CTC **implicit** học inter-label dependency (paper §6, sẽ thấy ở B5).

**🔗 Nối về repo:** cùng ý tưởng cho ảnh captcha — notebook `pipeline_deep_dive.ipynb` (phần 4) vẽ logits `[T,B,C]` theo trục width: spike tại vị trí ký tự, blank giữa các ký tự — đúng dạng panel CTC, chỉ khác speech → image.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟨 <span style="background-color:#FFF3CD; color:#856404; padding:3px 12px; border-radius:6px; border:1px solid #FFEEBA">B3 · INTUITION</span> <span style="color:#888; font-size:0.85em">— Đọc: §3.1 đoạn đầu (tr.2) + Fig 1 (tr.3)</span></h3>

**Checklist đọc:**
- [ ] §3.1 đoạn đầu (tr.2, từ "The crucial step…") — mô tả blank trước eq(2)
- [ ] ❗ Bỏ qua công thức ở lần đọc 1 — chỉ lấy trực giác

**Key points:**
- Paper không có mục "intuition" — phần này tự diễn giải.
- Analogy karaoke: biết lời bài hát nhưng không biết từng từ rơi vào nhịp nào.
- CTC **không commit vào 1 alignment** — nó cộng xác suất của MỌI path có thể (eq 3).
- Blank = "nhịp này chưa nhả ký tự mới". Toy: label `aa` trên 4 steps → các path `aa--`, `-aa-`, `a-a--`… đều collapse về `aa` qua map B.
- Trong code: blank cố định ở index 0 (`src/dataset.py:13-15`).

> [!TIP]
> Tự vẽ toy `aa` lên giấy trước khi sang B4 — đây là nền cho toàn bộ phần technical.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟥 <span style="background-color:#F8D7DA; color:#721C24; padding:3px 12px; border-radius:6px; border:1px solid #F5C6CB">B4 · HOW IT WORKS</span> <span style="color:#888; font-size:0.85em">— Đọc: §2 → §3.1 → §3.2 → §4.1 → §4.2 (tr.2–6)</span></h3>

**Checklist đọc (ăn gần hết phần kỹ thuật của paper):**
- [ ] §2 (tr.2): formalism X, Z, U ≤ T
- [ ] §3.1 + eq(2)(3) (tr.2–3): softmax, path π, map B
- [ ] §3.2 + eq(4) (tr.3): best path decoding, prefix search + **Fig 2**
- [ ] §4.1 (tr.4–5): extended `l′`, forward α eq(5)–(7), `p(l|x)` eq(8), backward β eq(9)–(11) + **Fig 3**
- [ ] §4.1 rescaling `C_t`, `D_t` (tr.5): chống underflow
- [ ] §4.2 (tr.5–6): objective eq(12), `α·β` eq(14), gradient eq(15)–(16) + **Fig 4**

**Key points (theo pipeline code):**
1. Softmax mỗi time-step → `y_t^k` = P(label k tại t); output `|L|+1` units — eq(2) ↔ `log_softmax(2)` tại `src/train.py:53`; `n_class = len(vocab) + 1` tại `src/train.py:142`.
2. Map `B: L′^T → L≤T`: bỏ blank + gộp ký tự lặp — `B(a−ab−) = B(−aa−−abb) = aab` (§3.1).
3. eq(3): `p(l|x) = Σ paths` — không thể enumerate (số path mũ T) → cần DP.
4. Decoding: best path eq(4) (`decode_greedy`) vs prefix search §3.2 (`decode_beam_search`).
5. Training §4.1: `l′` chèn blank đầu/cuối/giữa, len `2|l|+1` ↔ `extended_targets` (`ctc_loss.py:273-279`); forward α eq(6)–(7) với 3 transition stay/move/skip — skip chỉ khi `l′_s ≠ blank` và `l′_s ≠ l′_{s−2}` ↔ skip mask `ctc_loss.py:40-41`; eq(8) ↔ `ctc_loss.py:294`; backward β ↔ `_compute_beta_matrix`; rescaling `C_t` ↔ log-domain + `logaddexp` trong code.
6. Gradient §4.2: `α_t(s)β_t(s)` = xác suất mọi path qua symbol s tại t; eq(15)–(16) → error signal `∂O/∂u_t^k = y_t^k − posterior` ↔ `backward()` `ctc_loss.py:314`; Fig 4: error dạng spike, tự triệt tiêu khi hội tụ.

> [!IMPORTANT]
> **Bài tập bắt buộc trước buổi seminar:** tự vẽ lattice "CAT" như Fig 3, chạy tay eq(6)(7) cho 3 time-step đầu. Nếu chạy tay ra được = đã hiểu 80% paper.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟥 <span style="background-color:#F8D7DA; color:#721C24; padding:3px 12px; border-radius:6px; border:1px solid #F5C6CB">B5 · WHY IT WORKS</span> <span style="color:#888; font-size:0.85em">— Đọc: câu "Implicit in (2)…" (tr.3) + §4 mở đầu (tr.4) + §6 đoạn đầu (tr.7)</span></h3>

**Checklist đọc:**
- [ ] Câu "Implicit in (2) is the assumption…" ngay dưới eq(3) (tr.3)
- [ ] §4 đoạn mở đầu (tr.4): maximum likelihood + BPTT
- [ ] §6 đoạn đầu (tr.7): implicit inter-label dependencies

**Key points:**
- Marginalize latent alignment = biến segmentation thành latent variable; objective differentiable → BPTT chuẩn.
- Giả định then chốt: outputs **conditionally independent given hidden state** (đảm bảo bằng việc không có feedback từ output layer) — cái giá phải trả, khai thác ở B9.
- Inter-label dependency vẫn được **implicit** qua BiLSTM (§6).

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟦 <span style="background-color:#D1ECF1; color:#0C5460; padding:3px 12px; border-radius:6px; border:1px solid #BEE5EB">B6 · HISTORY & EVOLUTION</span> <span style="color:#888; font-size:0.85em">— Đọc: §1 (tr.1–2) + References (tr.8)</span></h3>

**Checklist đọc:**
- [ ] §1 (tr.1–2): các ref HMM/hybrid trong thân bài
- [ ] References (tr.8): skim Rabiner 1989 (HMM), Bourlard & Morgan 1994 (hybrid), Schuster & Paliwal 1997 (BRNN), Hochreiter & Schmidhuber 1997 (LSTM), Werbos 1990 (BPTT)

**Key points:**
- Timeline: HMM (1989) → hybrid HMM-NN (1994) → framewise RNN (2005) → **CTC (2006)** → RNN-T (2012) → seq2seq + attention (2015) → Whisper (2022, không dùng CTC cho decoder).
- Mỗi bước giải quyết hạn chế nào: HMM giải alignment nhưng generative + hand-crafted; hybrid thêm NN nhưng vẫn khung HMM; CTC bỏ hẳn khung ngoài; attention giải conditional independence nhưng đánh đổi autoregressive (chậm, khó streaming).
- Timeline sau 2006 là ngoài paper — kiến thức bổ sung cho slide.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟦 <span style="background-color:#D1ECF1; color:#0C5460; padding:3px 12px; border-radius:6px; border:1px solid #BEE5EB">B7 · COMPARISON</span> <span style="color:#888; font-size:0.85em">— Đọc: §1 (tr.1–2) + §5 intro (tr.6) + §6 đoạn 1–2 (tr.7)</span></h3>

**Checklist đọc:**
- [ ] §1 (tr.1–2): HMM vs RNN vs hybrid
- [ ] §5 intro (tr.6): thiết kế so sánh trong thí nghiệm
- [ ] §6 đoạn 1–2 (tr.7): điểm khác căn bản của CTC

**Key points:**
- Bảng tự tổng hợp: framewise / HMM / hybrid / CTC / attention seq2seq (paper không có bảng so sánh lý thuyết — so qua thí nghiệm).
- Khác biệt căn bản (§6): **không explicit segment**, không model inter-label dependencies, objective chỉ phụ thuộc sequence labels chứ không duration/segmentation.
- Ưu: end-to-end, decode nhanh, monotonic (hợp OCR/streaming). Nhược: conditional independence, không cắm LM trực tiếp vào objective.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟦 <span style="background-color:#D1ECF1; color:#0C5460; padding:3px 12px; border-radius:6px; border:1px solid #BEE5EB">B8 · TECHNICAL DEPENDENCIES</span> <span style="color:#888; font-size:0.85em">— Đọc: §3.1 định nghĩa `N_w` (tr.2) + §5 intro & §5.2 (tr.6–7)</span></h3>

**Checklist đọc:**
- [ ] §3.1 (tr.2): định nghĩa network `N_w: (R^m)^T → (R^n)^T`
- [ ] §5 intro + §5.2 (tr.6–7): chỗ paper khẳng định "any other architecture could have been used instead"

**Key points:**
- Algorithm-level (DP forward-backward — dùng cho MỌI architecture) ≠ model-level (BLSTM chỉ là backbone thay được).
- Training (cần α, β, gradient) ≠ inference (decoding không cần α/β — chỉ cần softmax).
- `blank=0` trong code là convention; `T` của CRNN = W/4 do pooling stride — quyết định xem label có "vừa" input không.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟦 <span style="background-color:#D1ECF1; color:#0C5460; padding:3px 12px; border-radius:6px; border:1px solid #BEE5EB">B9 · FAILURE MODES & EDGE CASES</span> <span style="color:#888; font-size:0.85em">— Đọc: §3.2 đoạn cuối (tr.3) + §5.2 (tr.7) + §6 đoạn cuối (tr.7–8)</span></h3>

**Checklist đọc:**
- [ ] §3.2 đoạn cuối (tr.3): prefix search fail khi nào
- [ ] §5.2 (tr.7): Gaussian noise σ=0.6
- [ ] §6 đoạn cuối (tr.7–8): overfitting, hướng future work

**Key points:**
- Target dài hơn input cho phép → không path hợp lệ → loss = ∞. **Trong code chính là lý do `zero_infinity=True`** (`src/train.py:146`) — ví dụ sống: T = W/4 = 25 với imgW=100. *(Kiến thức code-level, paper không nói trực tiếp.)*
- Ký tự lặp trong text (`"AA"`): bắt buộc cần blank giữa 2 ký tự giống nhau.
- Overfitting: paper tự nhận ML training của CTC khó generalize (§6).
- Prefix search decode sai khi cắt sai section (§3.2 cuối); output peaked làm beam search thừa thải.
- Vocab lẫn lowercase (`c j s u w x y`) trong data — risk hoa/thường, chưa xử lý trong code.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟩 <span style="background-color:#D4EDDA; color:#155724; padding:3px 12px; border-radius:6px; border:1px solid #C3E6CB">B10 · EXPERIMENTS</span> <span style="color:#888; font-size:0.85em">— Đọc: §2.1 eq(1) (tr.2) → §5.1 (tr.6) → §5.2 (tr.7) → Table 1 + §5.3 (tr.7)</span></h3>

**Checklist đọc (đúng thứ tự — hiểu metric trước khi xem kết quả):**
- [ ] §2.1 + eq(1) (tr.2): LER
- [ ] §5.1 (tr.6): TIMIT, MFCC
- [ ] §5.2 (tr.7): setup chi tiết
- [ ] Table 1 + §5.3 (tr.7): kết quả

**Key points:**
- TIMIT, 61 phonemes, metric LER eq(1) = edit distance chuẩn hóa ↔ `full_acc`/`char_acc` trong `src/utils.py:104`.
- Table 1: HMM ctx-indep 38.85% → ctx-dep 35.21% → hybrid 33.84% → weighted hybrid 31.57% → **CTC best path 31.47%** → **CTC prefix search 30.51%**.
- Tính fair: cùng BLSTM architecture; hybrid có thêm 183 params HMM + weighted-error heuristic; CTC không cần trick nào.
- ⚠️ Đây là phoneme labeling (frame-level) chứ không phải full ASR word error — đừng nói quá khi trình bày.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟩 <span style="background-color:#D4EDDA; color:#155724; padding:3px 12px; border-radius:6px; border:1px solid #C3E6CB">B11 · RESEARCH INTUITION</span> <span style="color:#888; font-size:0.85em">— Đọc: §6 (tr.7–8, đoạn hierarchical CTC)</span></h3>

**Checklist đọc:**
- [ ] §6 (tr.7–8): "One very general way of dealing with structured data…" + hierarchical CTC

**Key points:**
- Nguyên lý transferable: **"khi alignment/latent structure không biết trước, sum over alignments thay vì commit một alignment"**.
- Cùng họ tư duy: EM, mixture models, soft attention (heatmap α trong notebook 5c chính là posterior over alignments!), marginal likelihood trong Bayesian.

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟩 <span style="background-color:#D4EDDA; color:#155724; padding:3px 12px; border-radius:6px; border:1px solid #C3E6CB">B12 · TOY EXAMPLE</span> <span style="color:#888; font-size:0.85em">— Đọc: Fig 3 lattice "CAT" (tr.5) + Fig 2 prefix tree (tr.4)</span></h3>

**Checklist đọc:**
- [ ] Fig 3 (tr.5): chạy tay số liệu trên chính hình lattice "CAT"
- [ ] Fig 2 (tr.4): cấu trúc cây prefix search

**Key points:**
- Chạy tay sample `"233HP3"` từ `pipeline_deep_dive.ipynb` trên lattice nhỏ: chỉ heatmap α, hỏi khán giả đọc soft-alignment.
- Hoặc toy `l = "ab"`, T = 5: liệt kê vài path, chỉ blank ở `a−b`, chạy 1 bước recursion α.

**Bài tập của tôi (vẽ lattice vào đây hoặc giấy):**

&nbsp;

**✍️ Ghi chú của tôi:**

&nbsp;

---

<h3>🟩 <span style="background-color:#D4EDDA; color:#155724; padding:3px 12px; border-radius:6px; border:1px solid #C3E6CB">B13 · KEY TAKEAWAYS</span> <span style="color:#888; font-size:0.85em">— Đọc: §6 (tr.7–8) + §7 Conclusions (tr.8)</span></h3>

**Checklist đọc:**
- [ ] §6 (tr.7–8)
- [ ] §7 Conclusions (tr.8) — chốt xem có khớp 5 điều dưới đây không

**Key points:**
- **5 nhớ**: (1) CTC = marginalize mọi alignment qua blank + map B; (2) forward-backward tính `p(l|x)` trong O(T·|l|); (3) gradient `y − posterior` là error signal dạng spike; (4) decoding = best path (greedy) hoặc prefix/beam search; (5) conditional independence là giới hạn cốt lõi.
- **3 học tiếp**: RNN-T (streaming, không cần blank giữa repeat), seq2seq + attention (bỏ conditional independence), EM/forward-backward tổng quát.
- **1 câu**: "CTC biến bài toán gán nhãn chuỗi không có alignment thành một bài toán maximum likelihood differentiable bằng cách cộng xác suất trên mọi alignment, cho RNN tự học alignment trong lúc train."

**Tự kiểm tra cuối:** đóng paper, tự nói lại trong 1 phút: *what / why / how / khác gì trước / khi nào dùng*. Nói trôi = đạt.

**✍️ Ghi chú của tôi:**

&nbsp;
