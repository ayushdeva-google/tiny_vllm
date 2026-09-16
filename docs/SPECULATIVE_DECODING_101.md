# Speculative Decoding 101: Theory, Mechanics & Mathematics

Speculative Decoding (also called Assisted Generation or Speculative Sampling) is an algorithmic optimization technique that accelerates Large Language Model (LLM) inference without modifying model weights or sacrificing output quality.

---

## 1. The Core Motivation: The Memory Wall

During standard autoregressive generation (the decoding phase), an LLM generates text **one token at a time**:
1. Read prompt / previous tokens.
2. Load all model weights from High-Bandwidth Memory (HBM/VRAM) into processor registers/SRAM.
3. Compute forward pass for **1 token** ($X \in \mathbb{R}^{1 \times d_{\text{model}}}$).
4. Emit 1 token.
5. Repeat.

### Why Is This Inefficient?
At batch size 1, the arithmetic intensity (FLOPs per byte transferred) of standard decoding is near zero. A 7-billion parameter FP16 model has ~14 GB of weights. Generating a single token requires streaming all 14 GB of weights across the memory bus just to perform a few matrix-vector multiplications ($Y = XW$). The GPU compute cores (Tensor Cores) spend over 95% of their execution cycles stalled, waiting for memory to arrive.

### The Speculative Decoding Solution
Instead of loading the large model's weights once per token, what if we loaded them **once for every $K$ tokens**?
Speculative decoding achieves this by pairing two models:
* **Draft Model ($M_q$):** A small, low-latency model (e.g., 100M–1B parameters).
* **Target / Expert Model ($M_p$):** The large, high-capacity model (e.g., 8B–70B parameters).

The Draft model rapidly guesses $K$ candidate tokens sequentially. Then, the Target model evaluates all $K$ candidates **simultaneously in a single parallel forward pass** (acting like a mini-prefill). 

Because the Target model streams its heavy weight matrices over the memory bus **only once** to verify $K$ tokens, we drastically reduce memory bandwidth pressure and achieve a wall-clock speedup of **1.5x – 3x** with zero degradation in mathematical output quality.

> **Key Architectural Insight:** Speculative decoding is an **algorithmic** optimization, not a hardware kernel trick. It yields speedups whether executed in native PyTorch code or custom fused CUDA kernels.

---

## 2. Core Mechanics: Draft, Verify, Rollback

Every iteration of speculative decoding consists of four distinct phases:

```
[Context / Prompt] 
       │
       ▼
1. DRAFT PHASE ──► Draft Model generates K candidate tokens autoregressively
       │           Candidate sequence: [x_1, x_2, ..., x_K]
       ▼
2. VERIFY PHASE ─► Target Model runs ONE parallel forward pass over all K tokens
       │           Produces K+1 next-token distributions: [p_1, p_2, ..., p_{K+1}]
       ▼
3. MATCH / REJECT ► Compare Draft proposals against Target distributions
       │           Accept first i tokens (0 <= i <= K)
       │           Emit 1 bonus/corrected token from Target: y_{i+1}
       ▼
4. ROLLBACK ─────► Truncate KV Cache back to position (start_pos + i + 1)
                   Repeat cycle with new context
```

### The Causal Verification Pass
When the Target model evaluates the draft sequence $[x_1, x_2, \dots, x_K]$ appended to context $C$, causal self-attention computes logits for every position in parallel:
* Given $C$, the Target predicts distribution $p_1$ for position 1.
* Given $[C, x_1]$, the Target predicts distribution $p_2$ for position 2.
* Given $[C, x_1, x_2]$, the Target predicts distribution $p_3$ for position 3.
* ...
* Given $[C, x_1, \dots, x_K]$, the Target predicts distribution $p_{K+1}$ for position $K+1$.

Notice that for $K$ draft inputs, the Target produces **$K + 1$ output predictions**.

### The "+1 Bonus Token" Rule
Suppose the first $i$ draft tokens match the Target model's criteria ($0 \le i \le K$):
1. **If $i < K$ (Mismatch at index $i+1$):**
   * The draft proposal $x_{i+1}$ is rejected.
   * However, the Target model already computed the ground-truth prediction $y_{i+1}$ from the prefix $[C, x_1, \dots, x_i]$.
   * We immediately accept $y_{i+1}$ from the Target!
   * **Total emitted tokens:** $i$ (from draft) $+ 1$ (corrected by target) $= \mathbf{i + 1}$.

2. **If $i = K$ (All $K$ tokens match):**
   * All $K$ draft proposals are accepted.
   * The Target's prediction $y_{K+1}$ (computed from the full sequence $[C, x_1, \dots, x_K]$) is obtained **for free**.
   * **Total emitted tokens:** $K$ (from draft) $+ 1$ (bonus from target) $= \mathbf{K + 1}$.

3. **Worst Case ($i = 0$, first token mismatches):**
   * The Target rejects $x_1$, but emits $y_1$.
   * **Total emitted tokens:** $0 + 1 = \mathbf{1}$.
   * Even when the draft model is completely wrong, forward progress is guaranteed at standard autoregressive speed.

### Causal Branching & KV Cache Rollback
Why can't we keep downstream predictions if token $x_2$ mismatches?
Because autoregressive attention is strictly causal:
$$\text{Logits at position } 3 = f([C, x_1, x_2])$$
If $x_2$ is replaced by the Target's corrected token $y_2$, the computation at position 3 was evaluated on an alternate, invalid history. Keeping tokens after a mismatch causes severe hallucinations and grammar collapse.

Therefore, the KV cache entries for all positions after the accepted index must be **rolled back**. In a pre-allocated static KV cache buffer, this requires no memory re-allocations—we simply adjust the `start_pos` pointer backwards, allowing future steps to overwrite the discarded slots.

---

## 3. Verification Strategies: Greedy vs. Stochastic Sampling

### Strategy A: Greedy Decoding ($\text{Temperature} = 0.0$)
In greedy decoding, verification is a deterministic integer comparison:
```python
accepted_tokens = []
for j in range(K):
    target_token = target_logits[j].argmax(dim=-1).item()
    if draft_tokens[j] == target_token:
        accepted_tokens.append(draft_tokens[j])
    else:
        # Mismatch: append target's correction and break
        accepted_tokens.append(target_token)
        break
else:
    # All K matched: append bonus token
    bonus_token = target_logits[K].argmax(dim=-1).item()
    accepted_tokens.append(bonus_token)
```

---

### Strategy B: Speculative Sampling ($\text{Temperature} > 0$)
When sampling probabilistically, both models produce probability distributions rather than fixed token IDs. 
* Let $q(x)$ be the Draft model's probability distribution.
* Let $p(x)$ be the Target model's probability distribution.

If both models independently sample tokens from their respective distributions, they will frequently disagree due to random noise, even when $p(x) \approx q(x)$. 

To preserve the exact target distribution while maximizing agreement, Leviathan et al. (2022) developed **Speculative Rejection Sampling**:

#### 1. The Acceptance Criterion
When the draft model samples candidate token $x \sim q$:
$$\text{Acceptance Probability} = \min\left(1, \frac{p(x)}{q(x)}\right)$$
* **If $p(x) \ge q(x)$:** The Target model considers $x$ at least as likely as the Draft model did. **Accept with 100% probability.**
* **If $p(x) < q(x)$:** Accept with probability $\frac{p(x)}{q(x)}$ by drawing $r \sim \text{Uniform}(0, 1)$ and checking $r \le \frac{p(x)}{q(x)}$.

#### 2. The Residual Recovery Distribution
If candidate token $x$ is rejected, we cannot simply draw a new sample from $p(x)$. We must sample from the **residual distribution** $p'(x)$:
$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y \in V} \max(0, p(y) - q(y))}$$

This formula zeroes out any tokens the Draft model over-sampled and renormalizes the remaining probabilities, steering the replacement token strictly toward tokens the Target favored.

---

### Concrete 4-Token Worked Example

Let vocabulary $V = \{\text{"apple"}, \text{"banana"}, \text{"cherry"}, \text{"date"}\}$.

| Token $x$ | Draft $q(x)$ | Target $p(x)$ | Difference $p(x) - q(x)$ | Residual $\max(0, p - q)$ | Normalized $p'(x)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **"apple"** | 0.60 | 0.30 | -0.30 | 0.00 | **0.0%** |
| **"banana"** | 0.20 | 0.50 | +0.30 | 0.30 | **100.0%** |
| **"cherry"** | 0.10 | 0.10 | 0.00 | 0.00 | **0.0%** |
| **"date"** | 0.10 | 0.10 | 0.00 | 0.00 | **0.0%** |
| **Total** | **1.00** | **1.00** | | **0.30** | **1.00** |

#### Scenario 1: Draft proposes `"banana"`
$$\text{Acceptance Prob} = \min\left(1, \frac{0.50}{0.20}\right) = 1.0 \implies \mathbf{Accepted\ (100\%)}$$

#### Scenario 2: Draft proposes `"apple"`
$$\text{Acceptance Prob} = \min\left(1, \frac{0.30}{0.60}\right) = 0.50 \implies \mathbf{50\%\ chance\ of\ acceptance}$$
* If rejected, sample replacement from $p'(x)$.
* Since $p'(\text{"banana"}) = 1.0$, the Target emits `"banana"` as the replacement.

#### Mathematical Losslessness Proof
The marginal probability of emitting `"apple"` across the system:
$$P(\text{output} = \text{"apple"}) = q(\text{"apple"}) \times \min\left(1, \frac{p(\text{"apple"})}{q(\text{"apple"})}\right) = 0.60 \times 0.50 = \mathbf{0.30} = p(\text{"apple"})$$

The marginal probability of emitting `"banana"`:
$$\begin{aligned}
P(\text{output} = \text{"banana"}) &= \left[q(\text{"banana"}) \times 1.0\right] + \left[q(\text{"apple"}) \times (1 - 0.50) \times p'(\text{"banana"})\right] \\
&= (0.20 \times 1.0) + (0.60 \times 0.50 \times 1.0) \\
&= 0.20 + 0.30 = \mathbf{0.50} = p(\text{"banana"})
\end{aligned}$$

The final output is **statistically indistinguishable** from running the Target model alone.

---

## 4. Hyperparameter Optimization: The Lookahead $K$

The choice of draft length $K$ governs efficiency:
* Setting $K$ too high wastes compute running the draft model on tokens destined for rejection.
* Setting $K$ too low leaves memory bandwidth unexploited.

### Mathematical Formulation
Let:
* $\alpha \in [0, 1]$: Empirical acceptance rate per token.
* $c$: Cost ratio of Draft forward pass relative to Target forward pass ($c = \frac{T_{\text{draft}}}{T_{\text{target}}} \ll 1$).

The expected number of accepted tokens per verification round is:
$$\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

The normalized time cost per round is:
$$\text{Cost}(K) = c \cdot K + 1$$

The expected speedup factor is:
$$\text{Speedup}(K) = \frac{\mathbb{E}[\text{tokens}]}{\text{Cost}(K)} = \frac{1 - \alpha^{K+1}}{(1 - \alpha)(c \cdot K + 1)}$$

### Empirical Nature of $\alpha$ & Dynamic $K$
Because $\alpha$ varies significantly depending on context (e.g., $\alpha \approx 0.85$ for structured code generation vs. $\alpha \approx 0.35$ for creative prose), production engines (like vLLM) maintain an exponential moving average of $\alpha$ during generation and dynamically scale $K \in [2, 8]$ in real time.

---

## 5. Draft Models: Acquisition & Training

1. **Architecture Families:** The Draft and Target models **must share the identical tokenizer and vocabulary**.
2. **Off-the-shelf Pairings:**
   * Target: Llama-3.1-70B $\rightarrow$ Draft: Llama-3.1-8B.
   * Target: Qwen-2.5-72B $\rightarrow$ Draft: Qwen-2.5-7B or 1.5B.
3. **Knowledge Distillation for Custom Draft Models:**
   When no smaller public model exists (e.g., for Llama-3.2-1B):
   * Build a truncated model (e.g., 4 layers, reduced hidden dimension).
   * Train using Generalized Knowledge Distillation (GKD) or KL-divergence loss against the Target model's output logits over domain corpora.
   * Aligning the Draft's predictive distribution directly to the Target maximizes empirical $\alpha$.

---

## 6. Comparison with Other Inference Optimizations

| Technique | Optimization Axis | Quality Trade-off | Implementation Layer |
| :--- | :--- | :--- | :--- |
| **Quantization (W4A16 / FP8)** | Memory Bandwidth (smaller weights) | Slight perplexity change | Kernel / Matrix multiplication |
| **PagedAttention** | Memory Fragmentation (non-contiguous KV) | 100% Lossless | Memory Manager + Attention Kernel |
| **Continuous Batching** | GPU Compute Utilization (in-flight scheduling)| 100% Lossless | Inference Serving Scheduler |
| **Speculative Decoding** | Algorithmic Memory Bandwidth (fewer reads) | **100% Lossless** | Decoding Loop / Model Orchestration |

---

## 7. Verified Primary References

1. **Leviathan, Kalman, and Matias (Google Research / DeepMind, 2022):**  
   [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192)  
   *Introduces the formal algorithm and mathematical proof of lossless rejection sampling.*
2. **Chen et al. (DeepMind, 2023):**  
   [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318)  
   *Independent derivation of speculative sampling with benchmark speedup bounds.*
3. **Hugging Face Blog (João Gante, 2023):**  
   [*Assisted Generation: a new direction for low-latency text generation using Transformers*](https://huggingface.co/blog/assisted-generation)  
   *Detailed visual breakdowns of assisted decoding loops and KV cache handling.*
4. **PyTorch Engineering (2023):**  
   [*Accelerating Generative AI with PyTorch II: GPT, Fast*](https://pytorch.org/blog/accelerating-generative-ai-2/)  
   *Reference implementation of speculative decoding using pure PyTorch with static KV caches.*

---

## Appendix: Mathematical Proof of Losslessness (Exact Distribution Recovery)

A critical claim of speculative sampling is that the final output tokens are **strictly distributed according to the target model $p(x)$**, despite being proposed by a completely different draft distribution $q(x)$.

Here is the formal proof (Leviathan et al., 2022).

### 1. Definitions & Setup
Let:
* $V$ be the discrete vocabulary set.
* $q(x)$ be the probability distribution over $V$ output by the Draft model, where $\sum_{x \in V} q(x) = 1$ and $q(x) \ge 0$.
* $p(x)$ be the probability distribution over $V$ output by the Target model, where $\sum_{x \in V} p(x) = 1$ and $p(x) \ge 0$.

The sampling protocol:
1. Sample a candidate token $X \sim q$.
2. Accept $X = x$ with probability:
   $$\alpha(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$
3. If $X = x$ is rejected, sample a replacement token $Y$ from the residual distribution $p'(x)$:
   $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y \in V} \max(0, p(y) - q(y))}$$

Let $Y \in V$ be the final emitted token (either accepted from the draft or drawn from the residual distribution). We want to prove that:
$$\forall x \in V, \quad P(Y = x) = p(x)$$

---

### 2. Lemma: Total Rejection Probability
First, let us calculate the total probability $\beta$ that a proposed draft token is accepted:
$$\beta = \sum_{x \in V} P(X = x) \cdot \alpha(x) = \sum_{x \in V} q(x) \min\left(1, \frac{p(x)}{q(x)}\right) = \sum_{x \in V} \min(q(x), p(x))$$

Consequently, the total probability of **rejection** is:
$$1 - \beta = 1 - \sum_{x \in V} \min(q(x), p(x))$$

Since $\sum_{x \in V} p(x) = 1$, we can substitute $1$:
$$1 - \beta = \sum_{x \in V} p(x) - \sum_{x \in V} \min(q(x), p(x)) = \sum_{x \in V} \Big(p(x) - \min(q(x), p(x))\Big)$$

Notice the algebraic identity for any two real numbers $a, b$:
$$b - \min(a, b) = \max(0, b - a)$$

Therefore:
$$1 - \beta = \sum_{x \in V} \max(0, p(x) - q(x))$$

This proves that the denominator of the normalized residual distribution $p'(x)$ is **identically equal to the total rejection probability $(1 - \beta)$**:
$$p'(x) = \frac{\max(0, p(x) - q(x))}{1 - \beta}$$

---

### 3. Proof of Marginal Distribution
A token $x \in V$ can be emitted in exactly one of two mutually exclusive events:
1. **Event 1 (Draft Acceptance):** The Draft proposed $x$, and it was accepted.
2. **Event 2 (Rejection & Resampling):** The Draft proposed some candidate that was rejected, and the replacement token drawn from $p'$ happened to be $x$.

Using the Law of Total Probability:
$$P(Y = x) = P(\text{Accepted } x) + P(\text{Rejected and Resampled } x)$$

#### Evaluating Event 1:
$$P(\text{Accepted } x) = q(x) \cdot \alpha(x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$

#### Evaluating Event 2:
$$\begin{aligned}
P(\text{Rejected and Resampled } x) &= P(\text{Rejection}) \cdot P(\text{Sample } x \text{ from } p' \mid \text{Rejection}) \\
&= (1 - \beta) \cdot p'(x) \\
&= (1 - \beta) \cdot \frac{\max(0, p(x) - q(x))}{1 - \beta} \\
&= \max(0, p(x) - q(x))
\end{aligned}$$

#### Combining Both Events:
$$P(Y = x) = \min(q(x), p(x)) + \max(0, p(x) - q(x))$$

Applying the fundamental identity $\min(a, b) + \max(0, b - a) = b$:
* **Case 1 ($p(x) \ge q(x)$):**  
  $\min(q(x), p(x)) = q(x)$ and $\max(0, p(x) - q(x)) = p(x) - q(x)$.  
  Sum $= q(x) + p(x) - q(x) = \mathbf{p(x)}$.
* **Case 2 ($p(x) < q(x)$):**  
  $\min(q(x), p(x)) = p(x)$ and $\max(0, p(x) - q(x)) = 0$.  
  Sum $= p(x) + 0 = \mathbf{p(x)}$.

In all cases:
$$\forall x \in V, \quad P(Y = x) = p(x) \quad \blacksquare$$

---

### 4. Significance of the Result
This confirms that Speculative Sampling is **completely unbiased**:
1. **Zero Approximation Error:** Unlike quantization or pruning which introduce approximations, speculative sampling draws from the exact same mathematical probability distribution as the target model.
2. **Lossless Sampling:** Perplexity, temperature scaling, top-$p$, and diversity metrics are preserved identically to running the target model standalone.


