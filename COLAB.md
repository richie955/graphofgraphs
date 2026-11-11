# Run Graph of Graphs in Colab

## Prepare a leakage-safe PrimeKG split package

Open [notebooks/prepare_primekg_splits.ipynb](notebooks/prepare_primekg_splits.ipynb) in Colab. Put raw PrimeKG `kg.csv` at `/content/drive/MyDrive/kg.csv` or upload it when prompted. Edit the parameters cell, run the notebook, and download `primekg_split_package.zip`. The ZIP contains the selected original facts, split labels, owner list, source and selection hashes, duplicate-row report, and validation report. The exact train/validation/test ratios may differ slightly because facts between one entity pair stay together.

To build internal graphs from that package, open [notebooks/primekg_llm_gog_split_safe.ipynb](notebooks/primekg_llm_gog_split_safe.ipynb), supply the ZIP, and run its cells. It uses only `split == "train"` facts for LLM prompts and graph construction. It exports held-out source facts as labeled ROOT evaluation targets; derived internal edges are training-only message channels and are excluded from supervised target loss. No synthetic cross-context semantic facts are exported. The new notebooks are local files until these project changes are pushed or the notebooks are uploaded to Colab.

The 200,000-fact default is a target for the first experiment, not a claim that a full PrimeKG run has been verified on this machine. The raw `kg.csv` is needed to run that check.

The project is already connected to GitHub at `https://github.com/richie955/graphofgraphs`. The Colab notebook clones that repository into `/content/graphofgraphs` and reads the dataset at `docs/primekg_gog_30000.json`. Google Drive is not used.

## Open the notebook

After pushing the project to the `main` branch, open this URL:

`https://colab.research.google.com/github/richie955/graphofgraphs/blob/main/notebooks/colab_train_gog.ipynb`

Choose **Runtime → Change runtime type → T4 GPU**, then run the cells from top to bottom. The training command is equivalent to:

```bash
python -m training.train docs/primekg_gog_30000.json \
  --model gog --device auto --output-dir runs/colab-primekg \
  --epochs 100 --patience 15 --batch-size 1024
```

The notebook zips and downloads the report and checkpoint at the end. Colab's `/content` storage is temporary, so download the results before disconnecting. The PrimeKG JSON in this repository is about 59 MB, within GitHub's 100 MB per-file limit.

## Push the notebook and current project changes

Run these commands in PowerShell from `C:\Users\richi\Desktop\Graph of Graphs`:

```powershell
git add COLAB.md notebooks/colab_train_gog.ipynb training/link_prediction.py runs/primekg-clean-comparison/comparison_report.json
git commit -m "Add Colab training notebook"
git push origin main
```

The training script and report are included because they currently have local edits. Review `git status` before committing if you have made other changes you want to include.
