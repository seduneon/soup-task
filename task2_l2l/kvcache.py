"""KV caches with the duck-typed interface HF attention calls: update(k, v, layer_idx, kwargs) -> (K, V).

The engine sets `rows` (micro-batch slice) and `col` (first column written) before each layer call.
Columns are sequence slots: prompt columns 0..P-1 (left padded), generated token t at column P+t.

StaticKV  regular arm: every layer's cache allocated at full length up front (fixed size, reserved ~ allocated).
PagedKV   dynamic arm: pages of `page` columns allocated as decode reaches them, so memory the cache has not
          claimed yet can hold extra resident layers. Returns the concatenation of the pages (a transient copy).
"""
import torch


class StaticKV:
    def __init__(self, n_layers, B, S, kv_heads, head_dim, dtype, device):
        shape = (B, kv_heads, S, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.rows, self.col, self.S = slice(None), 0, S

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def update(self, k, v, layer_idx, cache_kwargs=None):
        c, q = self.col, k.shape[2]
        assert c + q <= self.S, "KV cache overflow"
        self.k[layer_idx][self.rows, :, c:c + q] = k
        self.v[layer_idx][self.rows, :, c:c + q] = v
        return self.k[layer_idx][self.rows, :, :c + q], self.v[layer_idx][self.rows, :, :c + q]


class PagedKV:
    def __init__(self, n_layers, B, S, kv_heads, head_dim, dtype, device, page=64):
        self.k = [[] for _ in range(n_layers)]
        self.v = [[] for _ in range(n_layers)]
        self.page_shape = (B, kv_heads, page, head_dim)
        self.page, self.dtype, self.device = page, dtype, device
        self.rows, self.col, self.S = slice(None), 0, S

    @property
    def bytes(self):
        return sum(p.numel() * p.element_size() for pages in self.k + self.v for p in pages)

    def bytes_at(self, cols):
        """Bytes once `cols` columns are filled in every layer (for the residency plan)."""
        per_page = 2 * len(self.k) * torch.Size(self.page_shape).numel() * torch.empty(0, dtype=self.dtype).element_size()
        return -(-cols // self.page) * per_page

    def _write(self, pages, x, c):
        q, done = x.shape[2], 0
        while done < q:
            p, o = divmod(c + done, self.page)
            while len(pages) <= p:
                pages.append(torch.zeros(self.page_shape, dtype=self.dtype, device=self.device))
            n = min(self.page - o, q - done)
            pages[p][self.rows, :, o:o + n] = x[:, :, done:done + n]
            done += n

    def update(self, k, v, layer_idx, cache_kwargs=None):
        c, q = self.col, k.shape[2]
        assert c + q <= self.S, "KV cache overflow"
        self._write(self.k[layer_idx], k, c)
        self._write(self.v[layer_idx], v, c)
        K = torch.cat(self.k[layer_idx], dim=2)[self.rows, :, :c + q]
        V = torch.cat(self.v[layer_idx], dim=2)[self.rows, :, :c + q]
        return K, V
