# Progress Test 2 — N18 — thyroid_ultrasound_cross_dataset_augmentation

- Fork: https://github.com/trangnguyenthu1002-eng/hmyt-project · Branch: gd5-proposed-model
- Members (student ID or alias):
  - 202414875
  - 202414879
- Instructors: PhD Hoang Quang Huy / PhD Tran Anh Vu
- Topic: Improving cross-dataset generalization of deep-learning thyroid nodule classifiers  through ultrasound-specific data augmentation / preprocessing 


## A. Literature review and research gaps
### A1. Search strategy

**Topic:** Improving cross-dataset generalization of deep-learning thyroid nodule classifiers through ultrasound-specific data augmentation / preprocessing

| Source (PubMed, Scopus, IEEE Xplore, Google Scholar…) | Keywords / query | Year range | No. of hits | No. kept |
|---|---|---|---|---|
| PubMed (MeSH + tiab) | `("Thyroid Nodule"[MeSH] OR "thyroid nodule*"[tiab] OR "thyroid cancer"[tiab] OR "thyroid neoplasm*"[tiab]) AND (Ultrasonography[MeSH] OR ultrasound[tiab] OR ultrasonograph*[tiab] OR sonograph*[tiab]) AND ("Deep Learning"[MeSH] OR "deep learning"[tiab] OR "convolutional neural network*"[tiab] OR CNN[tiab] OR transformer*[tiab]) AND (generaliz*[tiab] OR "external validation"[tiab] OR "cross-dataset"[tiab] OR "cross-domain"[tiab] OR "domain shift"[tiab] OR "domain adaptation"[tiab] OR "multi-center"[tiab] OR multicenter[tiab] OR robust*[tiab])` | 2015 – 11/2026 | Not recorded | Not recorded |
| PubMed (augmentation/preprocessing block) | Same first three blocks AND `(augmentation[tiab] OR "data augmentation"[tiab] OR preprocess*[tiab] OR "pre-process*"[tiab] OR despeckl*[tiab] OR speckle[tiab] OR "histogram equalization"[tiab] OR CLAHE[tiab] OR "intensity normalization"[tiab] OR "ROI crop*"[tiab] OR mixup[tiab] OR cutmix[tiab] OR "style transfer"[tiab] OR "generative adversarial"[tiab] OR "synthetic image*"[tiab])` | 2015 – 11/2026 | Not recorded | Not recorded |
| Scopus | `TITLE-ABS-KEY(("thyroid nodule*" OR "thyroid cancer") AND (ultrasound OR ultrasonograph* OR sonograph*) AND ("deep learning" OR CNN OR "neural network*" OR transformer*) AND (generaliz* OR "external validation" OR "cross-dataset" OR "domain shift" OR "domain adaptation" OR multicenter OR "multi-center") AND (augmentation OR preprocess* OR despeckl* OR "contrast enhancement" OR normaliz* OR segmentation OR "style transfer"))` | 2015 – 11/2026 | Not recorded | Not recorded |
| IEEE Xplore | `("All Metadata":"thyroid nodule" OR "All Metadata":"thyroid ultrasound") AND ("All Metadata":"deep learning" OR "All Metadata":"CNN") AND ("All Metadata":"data augmentation" OR "All Metadata":"preprocessing" OR "All Metadata":"generalization" OR "All Metadata":"domain adaptation")` | 2015 – 11/2026 | Not recorded | Not recorded |
| Web of Science Core Collection | `TS=(("thyroid nodule*" OR "thyroid cancer") AND (ultrasound OR ultrasonograph*) AND ("deep learning" OR "convolutional neural network*") AND (generaliz* OR "cross-dataset" OR "external validation" OR "domain shift") AND (augmentation OR preprocess* OR despeckl*))` | 2015 – 11/2026 | Not recorded | Not recorded |
| Embase (optional, if institutional access) | `('thyroid nodule'/exp OR 'thyroid cancer'/exp) AND 'echography'/exp AND ('deep learning'/exp OR 'convolutional neural network'/exp) AND (generaliz*:ti,ab OR 'external validation':ti,ab OR 'domain shift':ti,ab) AND (augmentation:ti,ab OR preprocess*:ti,ab)` | 2015 – 11/2026 | Not recorded | Not recorded |
| Google Scholar | `"thyroid nodule" ultrasound "deep learning" "data augmentation" generalization` and `"thyroid" ultrasound "cross-dataset" OR "external validation" OR "domain shift" classification` (screen the first 100–200 results of each) | 2015 – 11/2026 | Not recorded | Not recorded |
| arXiv / medRxiv / bioRxiv | `thyroid nodule ultrasound deep learning generalization augmentation` and `thyroid ultrasound domain adaptation OR domain shift` | 2018 – 11/2026 | Not recorded | Not recorded |
| Dataset / benchmark papers (hand search) | Public datasets and challenges: ThyroidXL, TN3K, DDTI, TN-SCUI 2020, Stanford thyroid cine-clip dataset, plus papers that cite them | 2015 – 11/2026 | Not recorded | Not recorded |
| Citation chasing | Backward and forward citations of key papers (e.g. the reviews already in the literature table) via Scopus or Google Scholar "Cited by" | None | Not recorded | Not recorded |

## Inclusion / exclusion criteria

### Inclusion

- **Population and modality:** B-mode (grayscale) thyroid ultrasound images or videos of thyroid nodules.
- **Task:** nodule classification (benign vs. malignant, or TI-RADS category). Studies that combine classification with segmentation are included if classification performance is reported.
- **Method:** deep learning (CNN, Transformer, or hybrid).
- **Intervention:** the paper studies or reports ultrasound-specific augmentation or preprocessing, such as speckle or noise handling, contrast or histogram normalization, ROI cropping, intensity or scale standardization, geometric or elastic augmentation, mixup/CutMix, GAN or diffusion synthesis, style transfer, or segmentation-guided input.
- **Generalization evidence:** the paper reports cross-dataset, external, or multi-center/multi-device testing, or a domain adaptation or harmonization experiment. Papers that only compare augmentation within a single dataset are kept as background (tag them separately).
- **Outcomes:** AUC, accuracy, sensitivity, specificity, or F1 on at least one held-out dataset or domain.
- **Publication:** peer-reviewed articles, conference papers, and preprints with enough methodological detail. English language, 2015 to 11/2026.

### Exclusion

- Non-ultrasound modalities (CT, MRI, cytology or histopathology images, scintigraphy) or non-thyroid organs (breast, liver).
- Classical machine learning or radiomics only, with no deep learning component.
- Segmentation-only or detection-only studies with no classification output.
- No external, cross-dataset, or multi-center evaluation and no augmentation or preprocessing analysis (e.g. generic CNN papers with a random train/test split).
- Augmentation or preprocessing described only as boilerplate with no justification, ablation, or comparison.
- Reviews, editorials, case reports, and abstract-only items without full text. Keep reviews separately for background and citation chasing, but do not count them as included studies.
- Duplicate or overlapping publications from the same cohort (keep the most complete one).
- Non-English papers and papers with no accessible full text.


### A2. Literature matrix (≥ 10 papers, preferably from the last 3–5 years)

| # | Paper (authors, year) | DOI / link | Dataset (n, source) | Method | Main result (metric, value) | External validation? (yes / no) | Main limitation |
|---|---|---|---|---|---|---|---|
| 1 | Weng et al., 2022 | arXiv:2207.13765 | 378 nodules, nine scanner types | Existing CNN applied to an independent dataset | AUC 0.69 overall; 0.56 (HDI 5000) to 0.75 (LOGIQ E9) by scanner | Yes (independent dataset) | Small scanner subgroups (e.g. 64 nodules), wide overlapping CIs; source model trained mostly on one vendor; no mitigation |
| 2 | Zhang et al., SHAN, MICCAI 2024 | LNCS 15004, pp. 732–741 | TN3K + private six-domain set | Shape-prior guided network, cross-domain *segmentation* | Not verified — confirm Dice from the full paper | Yes (cross-domain) | Segmentation, not classification; avoids intensity augmentation instead of repairing it |
| 3 | Duong Viet et al., ThyroidXL, MICCAI 2025 | 10.1007/978-3-032-05182-0_60 · huggingface.co/datasets/hunglc007/ThyroidXL | 11,635 images, 4,093 patients; Vietnam National Hospital of Endocrinology, Hitachi Aloka Arietta V70; patient-disjoint 80/20 split (9,541 / 2,094 images) | Benchmark: AlexNet, ResNet34/50, EfficientNet-B6/B7, VGG-11/13-BN classifiers (224×224, AdamW, 100 epochs), plus detection and segmentation baselines | Classification best F1 0.824 (VGG-13 BN), then 0.820 (EfficientNet-B7); patient-level results by weighted majority voting | No (single hospital, single scanner) | No external validation, no CIs; paper says models "might not generalize well to images from different equipment or clinical settings" and lists augmentation / domain adaptation as future work |
| 4 | RCAF, Sci. Rep. 2026 | s41598-026-61342-8 | Thyroid US, patient-level; Not verified — confirm sample size from the full paper | Region-aware fusion + attention MIL, leakage-free protocol | Not verified — confirm metric from the full paper | Not verified — confirm in the original source | Image–mask channel concat is shortcut-sensitive; no cross-dataset shift or augmentation study |
| 5 | Tupper & Gagné, 2025 | arXiv:2501.13193 | Several ultrasound classification tasks | Benchmark of natural-image vs ultrasound-specific augmentations | Brightness helped most tasks; contrast mixed | No (not cross-dataset thyroid) | Not thyroid; does not tie intensity changes to nodule-relative echogenicity |
| 6 | Zhang et al., TN5000, Sci. Data 2025 | 10.1038/s41597-025-05757-4 · 10.6084/m9.figshare.28455641.v1 | 5,000 B-mode images; official split 3,500 / 500 / 1,000 | Dataset (detection + classification) | Paper: 1,428 benign / 3,572 malignant (local XML audit: 1,426 / 3,574) | n/a (dataset) | Image-level split, no patient IDs; annotation anomalies (004092, 002589, 003813) |
| 7 | Zhuang et al., AMIA 2023 (publ. 2024) | AMIA Annu. Symp. Proc. vol. 2023 | Thyroid US, patient level; Not verified — confirm sample size from the full paper | Attention MIL on fused multi-scale features | Not verified — confirm metric from the full paper | Not verified — confirm in the original source | Single-source evaluation Not verified — confirm in the full paper |
| 8 | TN3K (Gong), Hugging Face dataset | huggingface.co/datasets/haifan-gong/TN3K | 3,493 images, multi-center | Dataset (segmentation + binary labels) | n/a | n/a | Not split by patient in all subsets; calipers reported |
| 9 | Pedraza et al., DDTI, SPIE 9287, 2015 | SPIE vol. 9287 | 347 images, 299 patients | Open-access database, TI-RADS labels | n/a | n/a | XML polygon faults; TI-RADS needs thresholding → label noise |
| 10 | Zhou et al., MixStyle, ICLR 2021 | ICLR 2021 | Natural-image DG benchmarks | Feature-statistics style mixing | Not verified — confirm in the original source | Yes (DG) | Physics-blind for ultrasound; planned only as optional baseline |
| 11 | Yang & Soatto, FDA, CVPR 2020 | IEEE/CVF CVPR 2020 | Segmentation DA benchmarks | Fourier amplitude swap | Not verified — confirm in the original source | Yes | Style images must come from training sources only; not ultrasound-specific |
| 12 | Tessler et al., ACR TI-RADS, 2017 | J. Am. Coll. Radiol. 14(5):587–595 | Expert consensus lexicon | Echogenicity graded relative to parenchyma / strap muscle | n/a (guideline) | n/a | Motivates the nodule-to-background ratio; not an ML study |


### A3. Research gaps

| ID | Gap (1–2 sentences) | Evidence (# in A2) | How your project addresses it |
|---|---|---|---|
| G1 | Intensity augmentation is known to threaten diagnostic grayscale, but the reviewed work avoids it rather than constraining it. | #2, #5 | EAA keeps the nodule-to-background intensity ratio fixed while changing global appearance/texture. |
| G2 | No reviewed work applies, during training, an augmentation that explicitly holds relative echogenicity fixed (review not systematic — novelty not yet claimed). | #2, #5 (absence of such a method) | Define EAA (Eqs. 1–5) and isolate the anchor via arm A4 vs A5; run a targeted augmentation search before claiming novelty. |
| G3 | Multi-source thyroid studies rarely control scanner, caliper artifacts, label noise and source-size imbalance within one protocol. | #1, #3, #8, #9 | Domain-balanced sampling, caliper audit of crops, patient-level splits, unseen external target (TN5000 official test; DDTI as extension). |
| G4 | Preprocessing and augmentation are usually fixed together in one pipeline, so their separate effects on external thyroid classification are are unknown. | #5 | One-at-a-time arms: P1 (CLAHE), P2 (SRAD), Geo, Crop, then combinations. |

### A4. PROBAST+AI appraisal of the most important paper in A2
Paper #: **3 — Duong Viet et al., ThyroidXL, MICCAI 2025** · Why this paper: it is the source of our training data and of the baseline models we compare against, and it names our gap itself — it warns that models "might not generalize well to images from different equipment or clinical settings" and lists data augmentation and domain adaptation as future work (p. 8).

Page numbers are the printed page numbers of the MICCAI open-access PDF. PROBAST+AI is built for prediction models, so it is applied here to the benchmark classification models of Sec. 4.1.

| Domain | Risk of bias (low / high / unclear) | Justification (quote or page number in the paper) |
|---|---|---|
| Participants and data sources | Unclear | Single hospital, single scanner. Selection criteria: "(1) normal thyroid function, (2) the thyroid nodules' diameters exceeding 5mm, and (3) no history of prior interventions" (pp. 3–4). "In cases where multiple nodules are present, we select the most suspicious one" and images are screened "avoiding blurring or motion artifacts" (p. 4). Consecutive enrolment is not stated; the restricted spectrum limits the population the results apply to. |
| Predictors | Low | Input is the ultrasound image only (224×224, p. 6). Non-ultrasound regions with patient information are cropped and patient IDs remapped (p. 4). Nothing derived from the outcome enters the input. |
| Outcome | Unclear | "labels were directly derived from cytological and pathological findings, eliminating inter-annotator variability" (p. 4), so no incorporation bias. But FNAC is not definitive, handling of Bethesda indeterminate categories is not described, and "most patients diagnosed with malignant nodules undergo surgical excision" (p. 4), i.e. malignant and benign nodules are probably verified differently. |
| Analysis | High | Patient-disjoint 80/20 split (9,541 images / 3,354 patients vs 2,094 / 739; p. 4) is a strength. Weaknesses: no external validation, no confidence intervals, no calibration; "For each model, we select the two versions that yield the best results" (p. 6) with no separate selection set; benign:malignant is about 65:35 in training vs 50:50 in the test set (p. 4), which shifts PPV and calibration; no sample-size justification; missing data not discussed. |

Applicability to your team's problem: Same task (benign/malignant) and the dataset we train on, with a published baseline (best F1 0.824, VGG-13 BN; EfficientNet-B7 0.820). It has no ResNet18, so our baseline must be re-trained, and our numbers are only comparable if we use the same split and metrics. Its single-hospital, single-scanner design is exactly the shift we test with TN5000.

---

## B. Results and timeline

### B1. Current results

| Model | Evaluation set (split method, n) | Main metric (value, 95% CI if available) | Secondary metrics | Source of numbers (file / notebook / W&B) |
|---|---|---|---|---|
| Baseline 1 (Milestone 2) — A1 ResNet18, ThyroidXL only, no augmentation | Internal: patient-level validation; External: TN5000 official test (n = 1,000, or 999 if image 004092 is excluded) | Not run | Not run | — |
| Proposed model (Milestone 3) — A5 ResNet18, multi-source + EAA | same as above | Not run | Not run | — |

Planned metrics (to avoid changing them later): primary F1 (malignant = positive, as in the ThyroidXL paper) with AUC reported alongside accuracy, precision, sensitivity, specificity; patient-level sensitivity/specificity on ThyroidXL only (TN5000 has no patient IDs); bootstrap 95% CI; paired DeLong tests.

### B2. Ablation (Milestone 3)
Planned arms (none run yet):

| Configuration (component removed / changed) | Main metric | Difference vs full model | Source |
|---|---|---|---|
| A5 full model (multi-source + EAA) | Not run | — | — |
| A4: ultrasound transforms without the echogenicity anchor | Not run | Not run | — |
| A3: ColorJitter (brightness/contrast) instead of EAA | Not run | Not run | — |
| A2: multi-source, no augmentation | Not run | Not run | — |
| A1: single source (ThyroidXL), no augmentation | Not run | Not run | — |
| P1 CLAHE only / P2 SRAD only / Geo (vertical flip) / Crop (full, tight, margin, black background) | Not run | Not run | — |

### B3. Comparison with the literature
No own numbers yet, so the comparison below is the reference frame we will fill in after the first runs.

| Paper | Their number | Our number | Why the comparison may be unfair |
|---|---|---|---|
| Weng et al. (#1) | AUC 0.69 on nine scanners | Not run | Different test set (378 nodules), pretrained model from another source, no training-set control |
| ThyroidXL baselines (#3) | Best F1 0.824 (VGG-13 BN); 0.820 (EfficientNet-B7) | Not run | We subsample ThyroidXL and crop with context; split and metric definition must match theirs before comparing |

Any comparison is unfair if the dataset, split (patient- vs image-level), crop/preprocessing or metric differs; TN5000 is split at image level, so leakage across train/test patients cannot be excluded.

### B4. Timeline to completion

| Task | Owner (student ID / role) | Milestone (M3 week 10 · M4 week 12 · Final week 15) | Status (Not started / In progress / Done + evidence) |
|---|---|---|---|
| Literature review + research gaps | Both students | M3 week 10 | Done — report Sec. III; targeted augmentation search still open |
| Dataset audit TN5000 (hash, XML/JPEG pairs, label counts) | Both students | M3 week 10 | Done — reports/tn5000_data_audit.md, outputs/extracted/tn5000_image_audit.csv |
| Dataset audit ThyroidXL (structure, labels, sizes, duplicates) | Both students | M3 week 10 | Done — slides 13–18; label total 11,635 matches the paper|
| Research protocol v1.6 (preprocessing §10) | Both students | M3 week 10 | In progress — proposed, not approved |
| Patient-level split + duplicate check (ThyroidXL) | Both students | M3 week 10 | In progress — README reports completion, but the supporting manifest/split artifact is not linked in the GitHub repository; verify and add its exact path before marking Done |
| Baseline 1: ResNet18, single source, no augmentation | Both students | M3 week 10 | Not started |
| Implement EAA + unit tests for the acceptance condition (Eq. 5) | Both students | M4 week 12 | Not started |
| Arms A2–A5, P1, P2, Geo, Crop | Both students | M4 week 12 | Not started |
| Calibration, bootstrap CIs, DeLong tests | Both students | M4 week 12 | Not started |
| Final report + TRIPOD+AI checklist + datasheet | Both students | Final week 15 | Not started |

### B5. Risks / blockers / questions for the instructor
- **No results yet:** is a ResNet18 baseline on ThyroidXL → TN5000 enough for Milestone 3, or must EAA already be tested?
- TN5000 label counts differ from the paper (1,426/3,574 vs 1,428/3,572); image 004092 has an XML/JPEG size mismatch → proposed exclusion (n = 999). Needs instructor approval.
- EAA needs a nodule mask but TN5000 gives only bounding boxes → mask must be derived, or EAA restricted to box crops.
- Ground-truth boxes are used for cropping; at deployment a detector would be needed (applicability limit).

---

## C. Self-assessment against the FAIR, TRIPOD+AI, PROBAST+AI appendix

### C1. FAIR — data
| Item | Answer | Evidence |
|---|---|---|
| Source, version, download date; checksum of the training file | TN5000: Figshare release v1, local archive TN5000_forReview.zip (206,183,840 bytes), MD5 matches the official release, audited 2026-10-26. ThyroidXL: Hugging Face `hunglc007/ThyroidXL'| reports/tn5000_data_audit.md; ThyroidXL checksum not recorded yet |
| Licence / DUA; basis for being allowed to use the data | both are public, published datasets> | 
| Datasheet: population, inclusion/exclusion criteria, variable definitions and units, limitations | Partial. TN5000 audit covers files, labels, splits, size categories and anomalies. ThyroidXL: structure, label/split balance, image sizes; formal datasheet not written | 
| No patient identifiers | TN5000: no patient-ID table. ThyroidXL: file names start with an 8-digit patient ID (e.g. 00000058_…) — a pseudonymous ID; confirm it is not traceable and that images carry no burned-in text/names | 

### C2. Reporting guideline
- Your research question: Does ultrasound-specific data augmentation and preprocessing improve cross-dataset generalization of pretrained CNNs (ResNet18) for benign/malignant thyroid nodule classification, tested on an unseen external dataset (TN5000)?
- Matching reporting guideline (TRIPOD+AI / STARD-AI / DECIDE-AI) and why (Appendix §3.2): **TRIPOD+AI** — we develop an AI classifier and evaluate it internally and externally. STARD-AI would fit a diagnostic-accuracy study of a fixed index test; DECIDE-AI applies to early live clinical evaluation, which we do not do. 
- Outcome, label, prediction time T0, variables available at T0: Outcome = malignant vs benign (malignant positive). Labels: ThyroidXL pathology-validated; TN5000 benign/malignant from the dataset XML. T0 = time of the thyroid ultrasound exam. Variables available at T0 = one B-mode image + nodule bounding box; no clinical covariates. (The box is expert-annotated, so at true T0 it would have to come from a detector — flagged as an applicability limit.)

### C3. PROBAST+AI — self-appraisal of your project
| Domain | Risk of bias (low / high / unclear) | Evidence in the repo (file, line) | What you will do to reduce the risk |
|---|---|---|---|
| Participants and data sources | Unclear | TN5000 audit: reports/tn5000_data_audit.md; inclusion criteria and acquisition details Not yet cited — exact repository file and line need verification | Document inclusion/exclusion per dataset in the datasheet; note ThyroidXL is single-hospital/single-vendor (Hitachi) |
| Predictors | Unclear | Preprocessing in protocol v1.6 §10 (box + 15%/30% context, 224×224, ImageNet normalisation); Not yet cited — exact code file and line need verification | Visual caliper/text audit of crops; inpaint or stratify if marks survive; fit any normalisation/CLAHE parameters on training only |
| Outcome (watch for incorporation bias) | Low for ThyroidXL (pathology-validated); unclear for TN5000 | Label counts audited (1,426/3,574 vs paper 1,428/3,572)| Confirm TN5000 label source; report results on the pathology-confirmed subset if available; the image is not used to define its own label, so incorporation bias is low |
| Analysis (patient-level split, fit on train only, sample size, missing data) | High until runs exist | Patient-level split planned for ThyroidXL; TN5000 official split is image-level, no patient IDs; no sample-size calculation. 

Applicability (population, setting where the model would be used): Training population is Vietnamese tertiary-hospital data (single vendor); TN5000 is a multi-scanner external set. The model would be a decision-support tool on cropped nodule ROIs; applicability is limited to adult thyroid nodules already localised by a clinician or detector, and to the scanners/settings represented.

### C4. Quick checklist (Appendix §7)
| Item | Met / Not yet | Evidence |
|---|---|---|
| Datasheet | Not yet (partial) | TN5000 audit report; ThyroidXL datasheet missing |
| Data version + checksum | Partly met | TN5000 MD5 matches Figshare v1; ThyroidXL checksum not recorded |
| No identifiers; basis for data use | Not yet | ThyroidXL patient-ID prefix and licence need confirming |
| Correct reporting guideline + completed checklist | Not yet | TRIPOD+AI chosen; checklist not completed |
| Outcome, label, T0 defined | Met | Report Sec. VI and slides 2, 6, 12 (see C2) |
| Patient-level split; preprocessing fit on train only | Not yet (evidence incomplete) | README reports a patient-level split, but the manifest/split artifact and evidence of train-only preprocessing are not linked here; verify both before marking Met |
| Baseline of original ThyroidXL | Not yet | Image task: baseline is ResNet18 without augmentation (arm A1/A2)|
| AUC + calibration (curve, in-the-large, slope) + Brier, with 95% CI | Not yet | AUC/F1/CI planned; calibration not yet in the protocol — add it |
| Is the sample size sufficient; applicability limits | Not yet | TN5000 test n = 1,000 (999); no formal calculation; limits listed in C3 |

