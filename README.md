# Designing failure-resilient CO$_2$ transport: Trade-offs between cost and emissions of redundant infrastructure

## Run

Create and activate the Conda environment:

```bash
conda env create -f requirements.yml
conda activate resilience-env
```

From the repository root, run:

```bash
python main.py
```

`main.py` reads its input CSV files from `inputs/` and builds the configured cost grids.
