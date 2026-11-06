# Run Graph of Graphs in Colab

The project is already connected to GitHub at `https://github.com/richie955/graphofgraphs`. The Colab notebook clones that repository into `/content/graphofgraphs` and reads the dataset at `docs/primekg_gog_30000.json`. Google Drive is not used.

## Open the notebook

After pushing the project to the `main` branch, open this URL:

`https://colab.research.google.com/github/richie955/graphofgraphs/blob/main/notebooks/colab_train_gog.ipynb`

Choose **Runtime → Change runtime type → T4 GPU**, then run the cells from top to bottom. The training command is equivalent to:

```bash
python train_stage9.py docs/primekg_gog_30000.json \
  --model gog --device auto --output-dir runs/colab-primekg \
  --epochs 100 --patience 15 --batch-size 1024
```

The notebook zips and downloads the report and checkpoint at the end. Colab's `/content` storage is temporary, so download the results before disconnecting. The PrimeKG JSON in this repository is about 59 MB, within GitHub's 100 MB per-file limit.

## Push the notebook and current project changes

Run these commands in PowerShell from `C:\Users\richi\Desktop\Graph of Graphs`:

```powershell
git add COLAB.md notebooks/colab_train_gog.ipynb stage9_training.py runs/primekg-clean-comparison/comparison_report.json
git commit -m "Add Colab training notebook"
git push origin main
```

The training script and report are included because they currently have local edits. Review `git status` before committing if you have made other changes you want to include.
