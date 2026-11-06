#!/usr/bin/env bash
#
# copy_paper_figures.sh
# ---------------------------------------------------------------------------
# Copies every figure the manuscript references into manuscript/paper_figures/,
# renaming where the manuscript expects a different name, in all three formats
# (.pdf, .png, .svg).
#
# Run from anywhere; paths below are absolute to your project.
# Safe to re-run (it overwrites the copies in paper_figures/).
# ---------------------------------------------------------------------------
set -u  # error on unset variables; we handle missing files ourselves

PROJECT="$HOME/projects/ODIN-Oral-Diagnostics-with-Intelligent-Networks/Omni_model/kfold_model_final_edit_Jun_2026"
SRC_SUMMARY="$PROJECT/output_figures/kfold_summary"
SRC_FOLD4="$PROJECT/output_figures/post_analysis/fold_4/run_20260616_181905"
DEST="$PROJECT/manuscript/paper_figures"

mkdir -p "$DEST"

# Counters
ok=0
miss=0

# copy_fig <source-dir> <source-basename> <dest-basename>
# Copies source-basename.{pdf,png,svg} -> dest-basename.{pdf,png,svg}
copy_fig() {
    local sdir="$1" sbase="$2" dbase="$3"
    local ext found_any=0
    for ext in pdf png svg; do
        local src="$sdir/$sbase.$ext"
        local dst="$DEST/$dbase.$ext"
        if [[ -f "$src" ]]; then
            cp -f "$src" "$dst"
            found_any=1
        else
            echo "  ! missing: $src"
        fi
    done
    if [[ $found_any -eq 1 ]]; then
        echo "  ✓ $dbase  (from $sbase)"
        ok=$((ok+1))
    else
        echo "  ✗ NOT FOUND: $sbase in $sdir"
        miss=$((miss+1))
    fi
}

echo "Destination: $DEST"
echo

echo "== kfold summary figures (same names) =="
copy_fig "$SRC_SUMMARY" kfold_forest_summary     kfold_forest_summary
copy_fig "$SRC_SUMMARY" kfold_per_fold_overview  kfold_per_fold_overview
copy_fig "$SRC_SUMMARY" kfold_pooled_confusion   kfold_pooled_confusion
copy_fig "$SRC_SUMMARY" kfold_pooled_roc         kfold_pooled_roc
copy_fig "$SRC_SUMMARY" kfold_pooled_pr          kfold_pooled_pr
copy_fig "$SRC_SUMMARY" kfold_f1_heatmap         kfold_f1_heatmap
copy_fig "$SRC_SUMMARY" kfold_tsne_representative kfold_tsne_representative
copy_fig "$SRC_SUMMARY" kfold_tsne_grid          kfold_tsne_grid

echo
echo "== dataset composition (renamed) =="
copy_fig "$SRC_SUMMARY" kfold_dataset_composition fig_1_dataset_composition

echo
echo "== fold-4 interpretability figures (renamed, from post_analysis) =="
copy_fig "$SRC_FOLD4/patient_level" patient_level_rescue  fold4_patient_level_rescue
copy_fig "$SRC_FOLD4/post_fusion"   gradcam_panel_correct fold4_gradcam_correct

echo
echo "---------------------------------------------------------------"
echo "Done. $ok figure(s) copied, $miss not found."
echo "Verify what landed:"
echo "  ls -1 $DEST"
