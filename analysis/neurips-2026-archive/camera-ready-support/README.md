# Camera-ready working draft

Edit `main.tex` for the main narrative and `appendix.tex` for the supporting material. The folder includes its own NeurIPS style, bibliography, checklist, and figure assets.

Build from this folder with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The main text presents response polarization through the histogram, KL distribution shifts, and a compact susceptibility summary, followed by within-persona robustness loss. A separate section covers harmful-dataset extensions, cross-instrument replication, and the MMLU control, before the mechanistic analysis of persona representations. Table 1 reports alignment; Appendix B.1 reports coherence. Section 6 includes the mechanistic implementation and persona-resampling procedure. The appendices collect measurement and training details, supporting plots, and toxic-persona prompts and profiles. Dataset recipes are consolidated in the fine-tuning appendix; questionnaire replication and the MMLU control are described in the main paper. Page-limit trimming is intentionally deferred.

## Figure and evidence notes

- The primary persona study covers GPT-4.1, GPT-4o, Qwen3-235B, and Qwen3.5-397B. The figures use simple model names; recipe details are in Methods.
- `regenerate_five_model_figures.py` reproduces the primary metric bars, foundation comparisons, self profiles, available toxic profiles, coherence comparison, and MMLU plot. `regenerate_kl_figures.py` reproduces the histogram and the paired KL heatmaps in Figure 3. Both write only into this folder and require NumPy and Matplotlib.
- `figure_style.py` provides the shared palette, typography, axes, and condition legend. Axes, ticks, and plot text are black; grid lines are light gray. `regenerate_replication_figures.py` reproduces the dataset extension, questionnaire replication, and activation-analysis figures; `regenerate_five_model_figures.py` reproduces the MMLU control. Figure labels use Qwen3 and Qwen3.5; full model identifiers are given once in Methods. Figure 4 pairs susceptibility with its percentage change from base; Figure 5 does the same for robustness. Figure 6 shows the Qwen3.5 dataset extension in two panels. Figure 7 shows questionnaire susceptibility, its percentage change, robustness, and its percentage change in four panels. Figure 8 pairs MMLU accuracy with its change from base in percentage points, using standard errors for accuracy and paired-bootstrap 95% confidence intervals for the changes. Figure 9 shows the mechanistic analysis. The histogram and metric bars use consistent model hues, with shades and white hatching for training conditions.
- KL retains all six valid MFQ ratings, including zero, and excludes failed parses. Each block mean includes all 16 comparisons, including the four zero self-comparisons in diagonal blocks. Mean base-to-insecure divergence is 2.030 bits, versus 0.569 for base-to-secure and 0.189 within the base-to-base block. Within-family insecure/secure divergence ratios span 3.03–5.15.
- Average susceptibility change is +69.87%; average robustness change is −75.01%. `data/figure_coverage.json` records the model coverage of the current plots.
- MMLU covers the four families with matching model variants. The selected MMLU sample has no statistically detectable accuracy decrease; some point estimates decrease and repeated-answer disagreement increases in two families. The main MMLU subsection describes the protocol, uncertainty estimates, and these results.
- Section 5.3 also reports the simple instruction-following control in `instruction-following-control/`: GPT-4o base, secure, and insecure each return 300/300 correct, valid lookup answers across 30 tables repeated 10 times, with zero within-table variation. The main paper gives the protocol; the repository folder retains the sampler, analysis, raw responses, and summary results.
- Appendix D, Moral Foundations Profiles, collects both the four-family Assistant profiles and the three-family toxic-persona comparison. It discusses the tendency toward saturation as supporting evidence of assistant-persona collapse and the toxic-profile mismatch as evidence against selection of the tested archetypes alone. Ceiling saturation is not universal. The BFI replication covers GPT-4o; the harmful-dataset extension covers Qwen3.5.
- Individual toxic-persona means are retained in `data/toxic_persona_individual_profiles.csv`, with the same persona IDs as the prompts in Appendix D.2. Response-level toxic logs are unavailable.
- The mechanistic analysis covers one family with one model per condition. Persona-resampling bands do not measure training-run uncertainty.
