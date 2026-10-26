| Strategy | Weak-model share | Quality (1-5) | Quality, hard requests | Good or better | TTFT p50 (ms) | TTFT p95 (ms) | Cost incl. judge ($) | vs always-Ultra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always-ultra | 0% | 3.78 | 3.8 | 66% | 660 | 2707 | 0.00739 | +0% |
| always-lightning | 100% | 3.66 | 3.8 | 70% | 233 | 308 | 0.00086 | -88% |
| inferedge-semantic | 74% | 3.68 | 4.2 | 70% | 247 | 1636 | 0.00257 | -65% |
| sy-classifier | 16% | 3.92 | 4.3 | 74% | 3874 | 7577 | 0.01115 | +51% |
| sy-random | 48% | 3.88 | 3.7 | 72% | 356 | 2079 | 0.00483 | -35% |
| sy-random-matched | 18% | 3.72 | 3.7 | 64% | 396 | 10529 | 0.00649 | -12% |

Quality vs always-ultra, paired by prompt (95% bootstrap interval):

- always-lightning: -0.12 [-0.54, +0.30] (not significant)
- inferedge-semantic: -0.10 [-0.44, +0.22] (not significant)
- sy-classifier: +0.14 [-0.12, +0.40] (not significant)
- sy-random: +0.10 [-0.26, +0.46] (not significant)
- sy-random-matched: -0.06 [-0.34, +0.20] (not significant)
