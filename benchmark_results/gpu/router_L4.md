Semantic router similarity scoring on NVIDIA L4 vs the container CPU (17 cores), 384-dim embeddings x 6 categories, median of 7 runs.

| Intents | CPU NumPy (ms) | GPU compute (ms) | GPU incl. transfer (ms) | Speedup (compute) | Speedup (incl. transfer) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.003 | 0.122 | 0.258 | 0.02x | 0.01x |
| 100 | 0.011 | 0.095 | 0.271 | 0.11x | 0.04x |
| 1,000 | 0.272 | 0.095 | 0.572 | 2.86x | 0.48x |
| 10,000 | 2.843 | 0.099 | 4.171 | 28.87x | 0.68x |
| 100,000 | 35.266 | 0.896 | 42.035 | 39.36x | 0.84x |
| 1,000,000 | 431.560 | 8.835 | 354.940 | 48.85x | 1.22x |
