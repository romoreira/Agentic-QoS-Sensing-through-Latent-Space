### Pairwise Comparison — paired-by-seed effect sizes on mean Gold shortage

| treatment | control   | n_seeds | mean_delta_mbps | std_delta_mbps | cohen_d_paired | seeds_treatment_wins | seeds_tied | pct_improvement |
| --------- | --------- | ------- | --------------- | -------------- | -------------- | -------------------- | ---------- | --------------- |
| DQN-LS    | DQN       | 3       | -2.43           | 2.52           | -0.964         | 2/3                  | 0          | 6.53            |
| DQN-LS    | Reactive  | 3       | -8.17           | 0.48           | -17.2          | 3/3                  | 0          | 19.05           |
| DQN-LS    | Fixed MBR | 3       | -98.33          | 0.98           | -100.525       | 3/3                  | 0          | 73.9            |
| DQN       | Reactive  | 3       | -5.75           | 2.53           | -2.269         | 3/3                  | 0          | 13.4            |
| DQN       | Fixed MBR | 3       | -95.91          | 1.58           | -60.687        | 3/3                  | 0          | 72.08           |
| Reactive  | Fixed MBR | 3       | -90.16          | 1.16           | -77.859        | 3/3                  | 0          | 67.76           |

_delta = mean_shortage(treatment) - mean_shortage(control). Negative delta -> treatment protects Gold better. `seeds_treatment_wins = k/n` is the count of seeds where the treatment had lower mean shortage. Cohen's d is paired and *descriptive only* given n=3: with such a small sample, inferential tests (Wilcoxon paired, paired t) are underpowered and are deliberately not reported._
