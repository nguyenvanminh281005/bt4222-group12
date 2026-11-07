## 16. Strategic Business Recommendations for Homepage Slate Rotation

### 16.1 Addressing Advisor Feedback Regarding Dynamic Slate Rotation
Our business objective targets **periodic homepage slate rotation every few hours**, whereas MIND logs represent static event snapshots showing only what MSN displayed at that specific timestamp.

To deploy NRMS effectively within a dynamic homepage rotation pipeline without logged feedback for newly rotated articles:
1. **Dense Semantic Embedding Space for Instant Article Rotation**:
   - The NRMS News Encoder projects every published article into a compact, semantically calibrated 256-dimensional latent space.
   - For brand-new articles without exposure logs, their 256-dimensional vectors are immediately available as soon as their headlines are tokenized.
2. **Nearest-Neighbor Proxy Feedback for Cold-Start Articles**:
   - Newly introduced candidate articles can be matched with their nearest neighbors in the 256-dimensional news embedding space using cosine similarity.
   - Historical engagement conversion rates and topical click profiles from nearest neighbors provide a reliable warm-up prior for offline slate replacement simulations.
3. **Diurnal Invariance & Safe Rotation Windows**:
   - Our temporal analysis (Experiment 4) demonstrates that NRMS ranking capability remains exceptionally stable across all 24 hours of Day 7 (AUC: 0.63 ± 0.01, NDCG@5: 0.32 ± 0.01).
   - This temporal stability indicates that scheduled rotation frequencies (e.g., every 3 to 4 hours) will not suffer from model degradation during morning or evening peak consumption hours.
4. **Seamless Integration of PG-NRMS Cold-Start Mitigation**:
   - For zero-history visitors, our **Popularity-Gated NRMS (PG-NRMS)** dynamically activates the global consensus prior $\mathbf{u}_{pop}$, boosting Cold-Start AUC by **+15.51%** (0.4875 $\to$ 0.5631) and NDCG@5 by **+14.50%** (0.2384 $\to$ 0.2730).
   - As users engage, the system naturally transitions into full personalized ranking (AUC > 0.61).

---

### Conclusion & Deliverable Summary
- **Original NRMS** successfully reproduced and benchmarked on MIND-small, outperforming Microsoft's official MIND-small baseline (AUC 0.6369 vs 0.6127).
- **Data pipeline cleaned**: `slate_length` and flawed `article_recency` proxies eliminated; 5-1-1 temporal split preserved.
- **Metrics reported**: AUC, MRR, NDCG@5, and NDCG@10 across single seeds and multi-seed configurations (Mean ± Std).
- **In-depth analyses completed**: Cold-start stratification, hourly diurnal temporal analysis, attention visualization, and architecture ablation.
- **Architectural innovation evaluated**: Popularity-Gated NRMS (PG-NRMS) successfully mitigates the cold-start gap without retraining or latency overhead.
- **Checkpoints, figures, and artifacts saved** in [`models/checkpoints/`](models/checkpoints/), [`models/figures/`](models/figures/), and [`.github/nrms.md`](.github/nrms.md).
