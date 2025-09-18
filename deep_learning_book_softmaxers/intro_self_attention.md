# Self-Attention: Beginner-Friendly Notes

Since the 2017 paper *Attention Is All You Need*, attention-based models have spread across many fields, especially NLP and computer vision, because they work so well. But what does “attention” actually mean?

Think of a clue like “Houston, we have a problem.” To guess the movie, you naturally focus on the most informative parts, such as “Houston” and “problem.” With a bit of world knowledge (often useful in pub quizzes), you recall the Apollo 13 mission and the 1995 film *Apollo 13*. That selective focus is the intuition behind attention.

In transformers, the attention mechanism lets the model attend to the most relevant pieces of information when processing many tokens. This helps the model capture long-range relationships within its context window, linking words that are far apart. In practice, a transformer can look back over the tokens currently in context and weight them by relevance, so it focuses computation where it matters most rather than treating everything equally.

---

## But don’t RNNs already do that?

Recurrent networks (RNN/GRU/LSTM) also consume sequences, but they pass information step by step. This makes learning long dependencies hard (gradients fade or explode), even with gates. Transformers replace recurrence with attention over the entire visible context in parallel, which is why they scale better. Note: the window is not infinite; it is limited by the model’s configured context length.

---

## What is self-attention?

Self-attention computes how much each token in a sequence should attend to every other token in the same sequence. The output for a token is a weighted mixture of transformed representations of all tokens, with weights telling us who matters for this token.

Three learned projections per token embedding:
- **Query (Q)** – what am I looking for?
- **Key (K)** – what do I contain?
- **Value (V)** – what information do I provide?

---

## Scaled dot-product attention (one head)

Given matrices \(Q, K, V\):

\[
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
\]

- \(QK^\top\) are relevance scores (query-key matches).
- Divide by \(\sqrt{d_k}\) to keep scores numerically stable as dimensionality grows.
- Softmax turns scores into weights (each row sums to 1).
- Multiply by \(V\) to mix values according to those weights.

---

## Multi-head attention (MHA)

One set of Q/K/V might miss different kinds of relations (syntax vs semantics, local vs global). Multi-head attention runs several attention heads in parallel, each with its own \(W_Q, W_K, W_V\), then concatenates and linearly projects:

\[
\text{MHA}(X) = \operatorname{Concat}(\text{head}_1,\ldots,\text{head}_h)\, W_O
\]

Each head captures a different view of the sequence.

---

## What do we pass to Q, K, and V?

Start from token embeddings \(X \in \mathbb{R}^{T \times d_{\text{model}}}\). Learn parameter matrices \(W_Q, W_K, W_V\), then compute:
\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V.
\]

- **Self-attention**: \(Q, K, V\) all come from the same sequence \(X\).
- **Cross-attention** (decoder → encoder): \(Q\) comes from the decoder state, \(K, V\) come from encoder outputs.

---

## Why do we need positional encoding?

Attention by itself is permutation-invariant. Language is not. Order matters. Transformers inject order by adding positional encodings to token embeddings:

\[
E_{\text{input}} = E_{\text{token}} + E_{\text{pos}}
\]

Two common choices:
- **Sinusoidal**: fixed, deterministic functions of position that allow learning relative positions via linear operations.
- **Learned positional embeddings**: parameters trained like word embeddings.

Intuition: positional signals let attention distinguish “dog bites man” from “man bites dog.”

---

## Encoder (quick tour from the original paper)

Each encoder layer contains:
1. Multi-head self-attention over the input sequence.  
2. Add & LayerNorm (residual connection plus normalization).  
3. Position-wise Feed-Forward Network (two linear layers with a nonlinearity like GELU or ReLU).  
4. Add & LayerNorm again.

Stack \(N\) such layers to deepen representation. The encoder output is a sequence of enriched embeddings that a decoder (for seq-to-seq tasks) can attend to.

---

## A tiny toy walk-through

Suppose our 4-token sentence is embedded into \(X \in \mathbb{R}^{4 \times d}\). For a single head:

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]
\[
S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{4 \times 4},\quad A = \operatorname{softmax}(S),\quad Z = AV
\]

- \(S\) are attention scores, \(A\) are row-wise weights, \(Z\) mixes values across tokens.
- With multiple heads, concatenate the per-head \(Z\)s and project with \(W_O\).

No recurrence; all tokens attend in parallel.

---

## Causal masks and padding masks

- **Causal (look-ahead) mask**: used in decoders and language models to prevent a token from attending to future tokens.
- **Padding mask**: prevents attention to padded positions in minibatches of varying lengths.

Masks are typically added as \(-\infty\) to disallowed positions before softmax.

---

## Pseudocode (one head)

```python
# X: [T, d_model]
Q = X @ W_Q         # [T, d_k]
K = X @ W_K         # [T, d_k]
V = X @ W_V         # [T, d_v]

scores = (Q @ K.T) / sqrt(d_k)  # [T, T]
scores += mask                  # optional: causal/padding (use -inf where blocked)

A = softmax(scores, dim=-1)     # [T, T]
Z = A @ V                       # [T, d_v]
out = Z @ W_O                   # [T, d_model]  # if single head, W_O may be identity

