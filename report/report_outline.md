# Report outline (hand-in Sun 08.11)

Each section says what goes in and where the material already is.

1. **Introduction**: the paper, the task (one value 24 steps ahead), our question: does the deep
   part earn its complexity? Who did what (Electricity: Chandana; Solar: Dinesh; shared code).
2. **Data**: both datasets, licences (Electricity: UCI, CC BY 4.0; Solar: NREL), split by time
   60/20/20, scaling by the series maximum (and its small leak: 53 Electricity customers peak in
   the test period, median ratio 1.00). Sources: notebooks 01.
3. **Method**
   - LSTNet-Skip: the five parts, figure of the architecture, settings from the authors'
     `ele.sh` / `solar.sh`. Differences between paper and code (GRU activation, skip length
     rounding, summed L1 loss in original units, Solar run script uses horizon 12).
   - Baselines: copy-the-past rules, one neuron, AR per series (L chosen on validation).
   - Improvement: RevIN, with the reason (paper Sec. 4.6, level shift found in the data).
   - Comparisons: DLinear, LSTNet without AR.
4. **Evaluation setup**: RSE and CORR as in the paper (constant series left out of CORR), model
   selection on validation, test used once, 10 seeds, paired tests over seeds, Diebold–Mariano
   with Newey–West lag h − 1 and HLN correction, Holm correction.
5. **Results**: reproduction table (ours vs paper), improvement table (mean ± std over seeds),
   statistical tests (`results/stats_<data>_h24.md`), one forecast plot per dataset.
6. **Discussion**: why our AR is stronger than the paper's (longer lag); what the ablation says
   about the paper's claim; when RevIN helps (Solar winter shift) and when not; limitations
   (one horizon, free Colab, authors' scaling leak).
7. **Conclusion**: one paragraph.
8. **AI usage**: from `AI_USAGE.md` (written at the end).
