# Reliable ECG classification with missing inputs

How does a deep ECG classifier behave when part of its input is missing, and when is it less reliable for some patients than for others?

This repository studies those two questions on PTB-XL, a public dataset of 21,799 clinical 12-lead ECGs (Wagner et al., 2020). It covers:

- **Missing modalities:** ECG leads that are absent at test time (E2a), and patient metadata that is missing in a systematic, non-random way (E2b).
- **Uncertainty:** deep ensembles of five networks (Lakshminarayanan et al., 2017). From the ensemble come predictive entropy (total uncertainty), mutual information (the part due to the model) and risk–coverage curves.
- **Calibration:** reliability diagrams, ECE and the Brier score.
- **Subgroup reliability:** discrimination and calibration by sex and age, with patient-level bootstrap tests of the gaps between groups (E3).
- **Monitoring after deployment:** whether a statistical test notices that the patient population has changed, and whether that change actually hurts the model (E4).

Every model performance number is computed on the official PTB-XL test fold. Confidence intervals are 95% intervals from a bootstrap that resamples patients, not individual ECGs, because one patient can have several recordings.

## Motivation

Models that work well on complete, curated benchmark data meet different conditions in practice:

- A wearable records one lead instead of twelve.
- A hospital site never records the patient's height.
- The patients the model serves differ from the ones it was tested on.

A model fit for practice should keep working when inputs are missing. It should also show its uncertainty when it cannot, and it should be equally reliable across patient groups. Each experiment tests one of these properties directly.

## Data and setup

| | |
|---|---|
| Dataset | PTB-XL v1.0.3, 100 Hz, 10 s, 12 leads |
| Task | Multi-label classification into the 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP |
| Records | 21,388 with at least one superclass label, from 18,617 patients |
| Split | Official stratified folds: 1–8 train (17,084), 9 validation (2,146), 10 test (2,158) |
| Model | 1D ResNet, about 1.1M parameters, in the style of the PTB-XL benchmark networks |
| Lead and metadata inputs | The network receives a lead mask, so it knows which leads are present. Metadata enters through a small MLP whose output is concatenated with the pooled ECG features (late fusion). |
| Training | Binary cross-entropy, AdamW (lr 1e-3, weight decay 1e-2), cosine schedule, early stopping on validation macro-AUROC |
| Ensembles | 5 seeds per configuration |
| Metrics | Macro-AUROC, Brier score and ECE (15 bins per class, averaged). ECE is reported as a point value: binned ECE is biased upward, so its bootstrap intervals are not trustworthy. |

All model and strategy choices are made on the validation fold. The test fold is used only for the reported numbers.

## Results

### E1: baseline

The 12-lead ensemble reaches **macro-AUROC 0.930 [0.923, 0.937]** on the test fold, with Brier 0.079 [0.075, 0.084] and ECE 0.021. This matches the published PTB-XL benchmark for the same task, sampling rate and splits: 0.930 for resnet1d_wang and 0.928 for xresnet1d101 (Strodthoff et al., 2021, Table II). Per-class AUROC [95% CI]: NORM 0.950 [0.941, 0.958], STTC 0.940 [0.927, 0.950], MI 0.930 [0.918, 0.940], CD 0.924 [0.907, 0.938], HYP 0.908 [0.886, 0.926]. The reliability diagrams stay close to the diagonal for all five classes.

![E1 reliability diagrams](figures/e1_reliability.png)

### E2a: missing leads

Leads are removed at test time to simulate three reduced setups:

- the 6 limb leads, as recorded without chest electrodes;
- lead I, as recorded by a smartwatch;
- lead II, the usual monitoring lead.

Three strategies are compared:

1. **Trained on 12 leads:** the E1 model, used as is.
2. **Lead dropout:** a single model trained on random lead subsets. Half of the training examples keep all 12 leads.
3. **Specialist:** a separate model trained for each lead setup. This is the reference upper bound.

| Test macro-AUROC | 12 leads | 6 limb leads | lead II | lead I |
|---|---|---|---|---|
| Trained on 12 leads | 0.930 | 0.794 | 0.525 | 0.481 |
| Lead dropout | 0.929 | 0.896 | 0.854 | 0.834 |
| Specialist per setup | 0.930 | 0.897 | 0.857 | 0.838 |

- **The 12-lead model fails when leads are missing.** On a single lead it performs at chance level, and its ECE rises from 0.02 to about 0.25. Zeroed leads resemble pathology to this model: its mean predicted probability of NORM drops to 0.08–0.10, against a true prevalence of 0.45.
- **One lead-dropout model handles every setup.** It comes within 0.004 AUROC of the specialists, costs nothing in discrimination on full 12-lead input (0.929 vs 0.930), and keeps ECE at about 0.03 throughout.

![E2a missing leads](figures/e2a_leads.png)

**Uncertainty.** As leads are removed from the lead-dropout model, its predictive entropy rises (0.23 on 12 leads, 0.36 on lead I), while its mutual information falls (0.020 to 0.008). The added uncertainty therefore comes from ambiguity in the input, not from disagreement between ensemble members.

**Abstaining.** On lead I, if the lead-dropout model answers only on the more certain half of the records, its Brier score on those records falls from 0.121 to 0.080. The 12-lead model, given lead I, gains almost nothing (0.254 to 0.249): its entropy does not single out the records it gets wrong. A model's uncertainty is only useful if it was trained on the kind of input it receives. The Brier score is the risk measure here; AUROC within the retained records is not, because the class mix of the retained records changes with coverage.

![E2a risk–coverage](figures/e2a_risk_coverage.png)

### E2b: missing metadata

Age, sex, height and weight are added to the ECG through late fusion. Age and sex are always recorded. Height is missing for 68% of records and weight for 56%. The missingness is far from random:

- It depends on the recording site. At the largest site (8,671 records) height is missing 99.8% of the time; at the second largest (6,242 records), only 5.5% of the time.
- Site, device and recording year predict whether height is missing with a cross-validated AUROC of 0.88 (0.79 for weight).
- Records without height have a higher MI prevalence (28.2% vs 20.2%).

Three ways of handling missing values are compared:

- **Mean imputation:** missing values are replaced by the training mean.
- **Missing indicators:** imputed values plus one binary "is missing" flag per field.
- **Modality dropout:** the indicator inputs, with 30% of training records having all their metadata removed.

Each model is evaluated in three conditions: metadata as recorded, only records with complete metadata, and all metadata removed.

![E2b metadata fusion](figures/e2b_metadata.png)

- **Adding metadata does not change discrimination.** With metadata as recorded, test macro-AUROC is 0.930 for ECG only and 0.930–0.932 for the three fusion models, differences of at most 0.002. The ECG itself already carries much of the information in age and sex: sex can be predicted from the ECG alone with AUC 0.92 (Strodthoff et al., 2021).
- **No strategy breaks when metadata is removed.** AUROC stays at 0.930–0.931.
- **On the subset with complete metadata (n = 672), every model scores higher, including the ECG-only model** (0.942 vs 0.930). The difference comes from *which patients* have complete metadata (mostly one site), not from the metadata. Evaluating only on patients with complete data would overstate the effect of adding metadata.
- **Missing indicators give a higher ECE** (0.031 vs 0.021), with the Brier score almost unchanged (0.081 vs 0.079). One possible reason is that the indicators partly encode the recording site.

### E3: subgroups by sex and age

Per-group results for the 12-lead E1 ensemble and for the lead-dropout model on lead I only:

![E3 subgroups](figures/e3_subgroups.png)

The gaps are tested directly with a bootstrap that resamples patients from both groups together. Overlapping per-group intervals alone do not show whether two groups differ.

| Gap in test macro-AUROC [95% CI] | 12 leads (E1) | Lead-dropout model, lead I |
|---|---|---|
| Male − female | +0.018 [0.004, 0.033] | +0.015 [−0.007, 0.037] |
| ≥80 years − 60–79 years | −0.036 [−0.064, −0.011] | −0.033 [−0.066, 0.002] |
| <40 years − 60–79 years | +0.004 [−0.021, 0.031] | −0.009 [−0.075, 0.055] |

- **Sex.** The 12-lead model discriminates slightly worse for women. The Brier score is also worse for women (−0.012 [−0.021, −0.003], male minus female). Label prevalences are similar between the groups except for STTC (21% in men, 28% in women). The gap shows up in NORM, MI, STTC and CD but not in HYP. It persists in the fusion model, which receives sex as an input.
- **Age.** Patients aged 80 and over are the least well served: lower AUROC and a worse Brier score (+0.021 [0.008, 0.033] vs 60–79). Their ECE is also the highest (0.057, rising to 0.094 on lead I), although ECE values from groups of different size are only descriptive. Part of this reflects a harder label mix rather than the model alone. NORM prevalence falls from 83% under 40 to 7% at 80 and over, and older patients more often carry several diagnoses at once.
- **The model's uncertainty rises with its error.** Mean predictive entropy grows from 0.16 (<40) to 0.33 (≥80), so the ensemble signals lower confidence in the group where it performs worst.
- **Comparing groups needs care.** Subgroup AUROC is compared between single groups (here against the largest age group) rather than against "everyone else". AUROC computed over pooled groups also rewards separating those groups from each other, which inflates it.

### E4: detecting a population shift

After deployment the mix of patients can change without anyone noticing. Can a test on the model's unlabeled inputs and outputs tell, and how many ECGs does it need?

The validation fold is the reference. Simulated deployment batches are drawn from the test fold, with a share of each batch (25%, 50% or 100%) taken from one subgroup: patients aged 80 and over (15% of the test fold), women (49%) or recordings from site 0 (22%). Batches with no subgroup share measure false alarms. Each setting is repeated 100 times. Two tests, both at α = 0.05:

- **Outputs:** a Kolmogorov–Smirnov test per class on the 12-lead ensemble's predicted probabilities, Bonferroni-corrected over the five classes (Rabanser et al., 2019).
- **Features:** a kernel two-sample test (MMD, Gretton et al., 2012) on the network's internal features (one ensemble member, reduced to 32 dimensions by PCA fitted on the reference), with 200 permutations.

![E4 shift detection](figures/e4_detection.png)

- **False alarms stay at the nominal level:** 1–6% of unshifted batches are flagged, for both tests and every batch size.
- **Older patients and site 0 are caught quickly.** When half of each batch comes from patients aged 80+, 88–89% of batches of 50 ECGs are already flagged. A 25% share needs about 500 ECGs for 80% detection, for both subgroups.
- **A shift towards women is almost invisible.** Batches made only of women are flagged in 46% of cases at 500 ECGs and 78–86% at 1,000; with a 50% share, never more than 9%. The model's Brier score is nonetheless worse on these batches (+0.006), in line with the sex gap in E3. Sex can be read from a 12-lead ECG (AUC 0.92, Strodthoff et al., 2021), yet this shift barely registers in the diagnostic model's outputs or features.
- **An alarm is not the same as harm.** Batches from site 0 are flagged as reliably as batches of older patients, but the model does better on them (Brier −0.018), while on older patients it does worse (+0.030). A shift test says *that* something changed, not whether it matters; E5 estimates the second part.
- **Features and outputs perform almost the same.** The feature test is slightly more sensitive at small shares (for example 49% vs 38% of batches flagged, aged 80+, 25% share, 100 ECGs), never by a large margin.

## Limitations and next steps

- **One dataset, no external validation.** All results come from PTB-XL's own test fold. Validation on an independent cohort is the obvious next step.
- **Cross-sectional only.** Each ECG is classified on its own. Repeated recordings of the same patient are handled in the statistics but not modelled, and there is no modelling of disease progression.
- **No causal analysis.** The subgroup gaps and the missingness patterns are described, not explained causally.
- **Metadata is a weak modality here.** Four demographic fields add little to a 12-lead ECG. Modalities that carry more independent information, such as clinical history or lab values, would test fusion under missingness more strongly.
- **Age and sex play two roles.** They are inputs to the fusion model and also define the subgroups in E3. The primary subgroup analysis therefore uses the ECG-only model.
- **No correction for multiple comparisons.** E3 reports 6 gaps for each of 4 models with unadjusted 95% intervals.
- **Simulated lead removal.** Removing leads from 12-lead recordings is not the same as recording with a single-lead device, whose signal quality and electrode placement differ.

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
.venv/bin/pip install -e .
DEVICE=cuda:0 bash scripts/run_all.sh
```

`run_all.sh` runs the whole pipeline in order:

1. downloads PTB-XL from PhysioNet;
2. runs the tests;
3. runs the missingness analysis;
4. trains and evaluates every ensemble (E1, E2a, E2b);
5. computes the subgroup analysis (E3);
6. draws the figures.

Ensembles that already have checkpoints are loaded instead of retrained. Results are written to `results/` and figures to `figures/`; up to GPU nondeterminism, they match the committed ones.

```
src/ptbxl_reliable/   data loading, lead masks, model, training, metrics, uncertainty, metadata features, subgroups
scripts/              one script per experiment, plus figures and run_all.sh
tests/                unit tests (pytest), runnable without the dataset
results/              all reported numbers as CSV
figures/              the figures above
```

## References

- Wagner, P. et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data* 7, 154.
- Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. (2021). Deep learning for ECG analysis: benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics* 25(5), 1519–1528.
- Rabanser, S., Günnemann, S., Lipton, Z. C. (2019). Failing loudly: an empirical study of methods for detecting dataset shift. *NeurIPS*.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research* 13, 723–773.
- Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
