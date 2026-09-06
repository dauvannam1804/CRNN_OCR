# Deep Dive into CRNN & CTC Loss for OCR

In this post, we'll explore the inner workings of a CRNN (Convolutional Recurrent Neural Network) for Optical Character Recognition (OCR), specifically focusing on **Connectionist Temporal Classification (CTC) Loss**.

We will walk through the implementation in this project (`src/`) and analyze a specific sample ("233HP3") from our `pipeline_deep_dive.ipynb` to demystify how gradients are calculated.

## 1. The Data Pipeline (`src/dataset.py`)

Before training, we need to prepare our images. The `OCRDataset` class handles this:

1.  **Loading**: Images are loaded and converted to grayscale (`L` mode).
2.  **Resizing**: All images are resized to a fixed height of **32px** and width of **128px**. This fixed size is crucial for batching.
3.  **Transforms**: The images are converted to Tensors and normalized to range `[-1, 1]`.
4.  **Label Encoding**: Combining a vocab list, text labels are converted to integer sequences (e.g., 'A' -> 1, 'B' -> 2). **Index 0 is reserved for the CTC Blank token.**

## 2. The Model Architecture: CRNN (`src/model.py`)

The model consists of three main components:

### A. Convolutional Feature Extractor (CNN)
The backbone is a 7-layer CNN (VGG-style).
*   **Input**: `[Batch, 1, 32, 128]`
*   It uses **Max Pooling** with a stride of `(2, 2)` initially, but later uses `(2, 1)`.
*   **Why `(2, 1)`?** We want to compress the specific Height (to 1) but preserve the **Width** (Time).
*   **Output**: Feature map of shape `[Batch, 512, 1, 26]`. The width `26` becomes our sequence length $T$.

### B. Map-to-Sequence
The tensor is reshaped to be compatible with RNNs: `[Width, Batch, Channels]` = `[26, Batch, 512]`. Now, each "pixel" column in the feature map is treated as a timestep.

### C. Recurrent Layers (RNN)
Two `BidirectionalLSTM` layers capture sequence context.
*   **Input**: `[26, Batch, 512]`
*   **Output**: `[26, Batch, n_class]`. These are the "Logits".

## 3. Explaining CTC Loss with a Real Sample

Let's look at the sample processed in `pipeline_deep_dive.ipynb`.
*   **Ground Truth**: `"233HP3"`
*   **Model Input Width ($T$)**: 26 timesteps.

 The challenge is **Alignment**. We don't know *where* the "H" is in those 26 steps. It could be at step 10, 11, or 12. CTC solves this by considering **all valid alignments**.

<image>Insert "Probability Heatmap (Model Output)" from notebook to show raw predictions</image>

### Step 3a: The Extended Target
CTC introduces a **Blank (`-`)** token to handle repeated characters and separation.
*   Original: `2 3 3 H P 3`
*   Extended ($L'$): `- 2 - 3 - 3 - H - P - 3 -`
We calculate probability paths through this extended sequence.

### Step 3b: Forward Algorithm ($\alpha$)
The **Alpha Matrix** ($\alpha_{t,s}$) stores the probability of aligning the first $s$ symbols of the target by timestep $t$.
At each step, we can:
1.  **Stay**: Keep outputting the same character (e.g., `2` -> `2`).
2.  **Next**: Move to the next token (e.g., `2` -> `-`).
3.  **Skip**: Jump over a blank (only if the next char is different, e.g., `2` -> `3`).

<image>Insert "Alpha Matrix" visualization from notebook showing the forward probability accumulation</image>

### Step 3c: The Gradient Calculation (The "Why")
How does the model learn? It effectively compares what it **predicted** vs. what the alignment **required**.

1.  **Soft Alignment ($\gamma$)**: calculated using Forward ($\alpha$) and Backward ($\beta$) variables.
    $$ \gamma_{t,s} = \frac{\alpha_{t,s} \cdot \beta_{t,s}}{P(z|x)} $$
    This tells us: "What is the probability that we *must* be at state $s$ at time $t$ given the full label?"

    <image>Insert "Soft Alignment (Gamma)" plot from notebook showing the bright valid path</image>

2.  **The Gradient Formula**:
    The derivative of the Loss with respect to the network output $y_k^t$ is:
    $$ \frac{\partial L}{\partial y_k^t} = y_k^t - \sum_{s \in \text{states}(k)} \gamma_{t,s} $$
    Or simply: **Gradient = Prediction - Target Alignment**.

    *   If the alignment says we *should* be outputting '3' (high $\gamma$) but the model predicts low probability ($y$), the gradient is negative, pushing the probability **up**.
    *   If the model predicts '3' where no valid path exists (low $\gamma$), the gradient is positive, pushing the probability **down**.

This mechanism allows the CRNN to clarify its predictions over time without ever needing pixel-level annotations!

### References
*   [Sequence Modeling With CTC (Distill.pub)](https://distill.pub/2017/ctc/)
*   [Breaking down the CTC Loss (Ogunlao)](https://ogunlao.github.io/blog/2020/07/17/breaking-down-ctc-loss.html)
*   Our `pipeline_deep_dive.ipynb` notebook.
