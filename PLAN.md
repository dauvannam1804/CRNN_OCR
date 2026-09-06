# PLAN — CRNN_OCR Review + CTC Seminar Prep

> Ghi chú ngày 07/09/2026. Review toàn bộ repo + kiểm chứng `src/ctc_loss.py`
> bằng fuzz-test so với `nn.CTCLoss` của PyTorch.

## A. Kết quả review project (đã verify bằng fuzz-test)

**Đã kiểm chứng:** `src/ctc_loss.py` khớp 100% với `nn.CTCLoss` của PyTorch
(loss + gradient, sai số ~1e-6, 20 cấu hình fuzz gồm ký tự lặp,
`input_lengths` biến thiên, target dài ngắn) khi `zero_infinity=True`.

**Mapping paper ↔ code (tài liệu seminar):**

| Paper | Code |
|-------|------|
| eq(2) — path probability | `log_softmax` (src/train.py:53, src/train_custom.py:54) |
| eq(5)–(7) — forward variables | `_compute_alpha_matrix` (src/ctc_loss.py:6) |
| eq(9)–(11) — backward variables | `_compute_beta_matrix` (src/ctc_loss.py:63) |
| eq(15)–(16) — gradient | `CTCLossFunction.backward` (src/ctc_loss.py:314) |
| Best path decoding | `decode_greedy` (src/utils.py:10) |
| Prefix search decoding | `decode_beam_search` (src/utils.py:33) |

**Các việc cần làm (theo thứ tự ưu tiên):**

1. [ ] **Fix bug squeeze** — src/train.py:45, src/train_custom.py:46:
       `label_lengths.squeeze()` → `squeeze(-1)`
       (crash khi batch cuối chỉ còn 1 sample)
2. [ ] **Warn khi drop ký tự ngoài vocab** — src/dataset.py:47:
       thêm log warning thay vì bỏ im lặng
3. [ ] **Quyết định case-sensitive**: data có lẫn lowercase
       (`X86jC4.jpeg`, `x998Y9.jpeg`) → nếu captcha không phân biệt hoa/thường
       thì normalize `.upper()` ở src/dataset.py:44; nếu có thì giữ + ghi chú README
4. [ ] **inference.py thêm `--decode greedy|beam`** — tái sử dụng
       `decode_beam_search` (đang có sẵn nhưng chưa nối vào CLI)
5. [ ] **Thêm requirements.txt hoặc pyproject.toml**
       (torch, torchvision, numpy, Pillow, matplotlib, pandas)
6. [ ] **Dọn cosmetic**: duplicate import src/utils.py:2-5,
       duplicate dòng README.md:67-68
7. [ ] **README ghi chú**: testset đang dùng làm validation
       (không có test set độc lập) — quyết định thiết kế, không bắt buộc sửa

## B. Lộ trình đọc lại paper CTC (5 lượt, ~30–45 phút/lượt)

1. **Vấn đề** (Sec 1–2): framewise thất bại thế nào, hạn chế HMM/hybrid,
   LER, vì sao không align trước được
2. **Biểu diễn đầu ra** (Sec 3, Fig 1–2): blank, eq(2), map B,
   decoding NP-hard, best-path vs prefix search
3. **Forward-backward** (Sec 4.1, Fig 3) — *quan trọng nhất*:
   tự vẽ lattice "CAT", chạy tay eq(6)(7), rule skip
   (l'_s ≠ blank và l'_s ≠ l'_{s-2}), rescaling chống underflow
4. **Gradient & training** (Sec 4.2, Fig 4): eq(12)→(16),
   ý nghĩa α·β/y, error signal dạng spike
5. **Thí nghiệm & tổng hợp** (Sec 5–7): tính công bằng của so sánh,
   hạn chế CTC, tiến hóa HMM → hybrid → CTC → RNN-T → attention

## C. Prompt đọc paper (3-pass, thay thế prompt 15-phút của mentor)

- **Pass 1 — Intuition**: giữ sec 1–3, 5–7, 11–13 của prompt gốc;
  thêm ràng buộc "trích section/equation number, không bịa ngoài paper text";
  ghi rõ "ICML 2006, pre-transformer era"
- **Pass 2 — Math deep-dive** (mới): derive eq(5)–(16) từng bước +
  toy example "CAT" với bảng α/β đầy đủ +
  map từng equation sang `src/ctc_loss.py` của tôi
- **Pass 3 — Seminar-prep** (mới): câu hỏi dự kiến từ khán giả + quiz +
  gợi ý demo (visualize blank-spike như Fig 1/Fig 4 bằng model đã train) +
  mục "CTC 2006 vs hiện đại" (RNN-T, seq2seq+attention,
  vì sao Whisper không dùng CTC, vì sao CTC vẫn sống ở OCR/streaming ASR)
