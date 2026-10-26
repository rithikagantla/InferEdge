Held-out accuracy (60 labeled support messages):

| Embeddings | Category accuracy | Model-tier accuracy |
|---|---:|---:|
| Hashed words (default) | 42% | 67% |
| all-MiniLM-L6-v2 | 93% | 100% |

End-to-end routing latency, text in to category out, on NVIDIA L4 vs 8 CPU cores (median of several runs):

| Messages | CPU (ms) | GPU, all on device (ms) | GPU embed, CPU score (ms) | Speedup | GPU messages/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.6 | 7.1 | 7.1 | 1.5x | 140 |
| 10 | 17.7 | 8.9 | 8.6 | 2.0x | 1,122 |
| 100 | 75.6 | 13.8 | 13.8 | 5.5x | 7,262 |
| 1,000 | 703.0 | 72.7 | 74.1 | 9.7x | 13,757 |
| 10,000 | 5850.9 | 647.9 | 654.7 | 9.0x | 15,435 |
