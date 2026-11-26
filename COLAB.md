# Run Graph of Graphs in Colab

## Prepare source splits

Open [notebooks/prepare_primekg_splits.ipynb](notebooks/prepare_primekg_splits.ipynb) in Colab. Provide PrimeKG `kg.csv` and download the resulting split package. This workflow creates leakage-safe source-fact splits; it does not build GoG structures.

## Construct GoG data

Open [notebooks/primekg_llm_gog_split_safe.ipynb](notebooks/primekg_llm_gog_split_safe.ipynb), provide the split package, and run the builder. It constructs internal graphs from training facts only and retains held-out source facts as evaluation labels.

The general trainer accepts a constructed GoG JSON from the repository root:

```bash
python -m training.train data/processed/primekg_gog_30000.json --model gog --device auto
```

See `data_pipeline/` for the data stages and `gog_embedding/` for the current embedding implementation.
