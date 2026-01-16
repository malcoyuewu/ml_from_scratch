Since you have a background in large-scale generative AI, you'll recognize that Multi-Head Attention (MHA) is the engine of the Transformer. The core idea is to allow the model to jointly attend to information from different representation subspaces at different positions.

Below is a clean, modular implementation using **PyTorch**. This version avoids using `nn.MultiheadAttention` to show you exactly how the tensor reshaping and "head splitting" works.

### 🧠 Multi-Head Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers to project Query, Key, and Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch, num_heads, seq_len, d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        # Reshape to (batch, seq_len, num_heads, d_k) then transpose to (batch, num_heads, seq_len, d_k)
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        # Transpose back to (batch, seq_len, num_heads, d_k) then flatten last two dims
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. Linear projections
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # 2. Scaled Dot-Product Attention
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. Combine heads and final projection
        output = self.combine_heads(attn_output)
        return self.W_o(output)

# --- Demo Code ---
if __name__ == "__main__":
    # Parameters
    batch_size = 8
    seq_len = 24
    d_model = 512
    num_heads = 8
    
    # Random input (Batch, Seq_Len, D_Model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(x, x, x) # Self-attention
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")

```

---

### 🔍 Key Logic Explanations

#### 1. The Scaling Factor ()

We divide by the square root of the head dimension. This is crucial for stability. As  grows, the magnitude of the dot product grows, pushing the softmax function into regions where gradients are extremely small.

#### 2. Transposition for Parallelism

The most important part of the code is the `transpose(1, 2)` in `split_heads`.

* **Input:** `(Batch, Seq, Model_Dim)`
* **After Split:** `(Batch, Seq, Heads, Head_Dim)`
* **After Transpose:** `(Batch, Heads, Seq, Head_Dim)`
This allows PyTorch to perform the matrix multiplication for all heads simultaneously using efficient batch processing.

#### 3. Linear Projections

Instead of creating  separate linear layers for  heads, we create one large linear layer (`d_model` to `d_model`) and split the resulting tensor. This is mathematically equivalent and much more computationally efficient.

---

### 💡 Potential Optimizations for Production

Given your work with large-scale generative systems, you might consider these follow-up adjustments for a more robust version:

* **FlashAttention:** In practice, standard `matmul` for attention is . Using `torch.nn.functional.scaled_dot_product_attention` (available in PyTorch 2.0+) leverages FlashAttention kernels.
* **Rotary Positional Embeddings (RoPE):** Most modern LLMs (like Llama) apply RoPE to the  and  tensors *after* the split but *before* the dot product.
* **KV Caching:** For inference, you wouldn't recompute the entire Key and Value tensors for every new token.

**Would you like to see how to integrate Rotary Positional Embeddings (RoPE) into this MHA class?**
