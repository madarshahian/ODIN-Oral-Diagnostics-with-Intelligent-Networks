# ODIN — Oral Diagnostics with Intelligent Networks

Code, trained weights, and results for the article:

> **Multi-View Deep Learning for Differentiating Maxillary Cyst-like Lesions
> from Mucous Retention Cysts on Cone Beam Computed Tomography**
> Najmeh Movahhedian, Farimah Heidari, Ramin Madarshahian.
> *(Manuscript under review; citation will be updated on publication.)*

ODIN is a dual-branch convolutional neural network that analyzes paired
coronal and sagittal CBCT slices of the maxillary sinus and classifies them as
**healthy**, **mucous retention cyst (MRC)**, or **maxillary alveolar
cyst-like lesion (MACL)**. Each branch is refined by a Convolutional Block
Attention Module (CBAM), and a learned gate fuses the two views. The model has
about 185,000 parameters.

> **Research use only.** ODIN is not a medical device and has not been
> validated for clinical use. External, multi-device validation is required
> before any clinical application.

## Results (5-fold patient-level cross-validation, 234 patients)

| Measure | Value |
|---|---|
| Mean accuracy (± SD across folds) | 94.44% ± 4.84% |
| Mean macro-F1 | 0.941 ± 0.045 |
| Pooled out-of-fold accuracy (95% CI, patient-level cluster bootstrap) | 0.949 (0.915–0.973) |
| Pooled out-of-fold macro-F1 (95% CI) | 0.950 (0.917–0.971) |
| One-vs-rest AUC: healthy / MACL / MRC | 0.997 / 0.989 / 0.994 |
| MACL-vs-MRC AUC (95% CI) | 0.993 (0.980–0.999) |
| Patient-level accuracy (majority vote) | 94.0% (220/234) |

## Data availability

The CBCT images are **not** included. They are clinical data from the School
of Dentistry, Shiraz University of Medical Sciences (ethics approval
IR.SUMS.DENTAL.REC.1403.035) and cannot be shared publicly. Figures that show
patient images (Grad-CAM overlays, misclassified-image panels) are also
excluded. The saved predictions in `logs/` are sufficient to reproduce every
reported metric without the images.

## Repository structure

```
kfold_model_final_edit_Jun_2026/             main experiment (cross-patient pairing)
├── 00_rename_data.ipynb                     anonymize and rename source images (P###)
├── 01_Organize_data_kfold.py                patient-level 5-fold split, padding, leakage audit
├── 02_model_training_kfold.py               ODIN model and training (one fold per run)
├── 03_aggregate_kfold_results.py            pooled metrics, ROC/PR, confusion matrix, figures
├── 03_post_analysis.py                      Grad-CAM, t-SNE, patient-level analysis (needs images)
├── 04_bootstrap_ci.py                       patient-level cluster-bootstrap 95% CIs
├── 05_macl_vs_mrc_auc.py                    MACL-vs-MRC AUC with bootstrap CI
├── copy_paper_figures.sh                    copy figures into the manuscript folder
├── logs/
│   ├── data_prep_kfold_*/                   fold assignment, leakage report, image statistics
│   └── fold_{1..5}/run_*/
│       ├── best_model_composite.keras       trained weights (checkpoint used in the paper)
│       ├── run_metadata.json                seeds, settings, package versions
│       └── artifacts/test_predictions.pkl   out-of-fold predictions
└── output_figures/                          aggregate figures (no patient images)

kfold_model_final_edit_Jun_2026_no_mixing/   ablation: within-patient pairing only
```

Class names in the code and saved files: `healthy`, `lesion` (= MACL), `mrc`.

## Reproduce the reported metrics (no images needed)

```bash
pip install -r requirements.txt
cd kfold_model_final_edit_Jun_2026

python 03_aggregate_kfold_results.py --n-folds 5 --logs-dir logs \
       --out-dir output_figures/kfold_summary      # pooled metrics and figures
python 04_bootstrap_ci.py --logs-dir logs           # 95% CIs (cluster bootstrap)
python 05_macl_vs_mrc_auc.py --logs-dir logs        # MACL-vs-MRC AUC
```

## Train on your own data

Organize the images as `<data>/<class>/<view>/P###_<class>_<index>_<CR|SG>.jpg`,
then run `01_Organize_data_kfold.py` (split and audit) and
`02_model_training_kfold.py --fold k` for k = 1…5. Training used an NVIDIA
GeForce RTX 4060 (8 GB) in float32. Fold 5 of the main experiment used
gradient-norm clipping (clipnorm = 1.0); see the article for details.

## License

- **Code:** MIT License — see [`LICENSE`](LICENSE).
- **Trained weights, saved results, and figures:** CC BY-NC 4.0 — see
  [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS).

## Citation

If you use this code or the trained models, please cite the article above
(full reference to be added on publication).

## Contact

Ramin Madarshahian — madarshahian@gmail.com
