# Reproducibility Notes

## Generate Synthetic Data

Generate a linear dataset:

```powershell
python src/make_demo_data.py --mode linear --n-samples 1000
```

Generate an interaction dataset:

```powershell
python src/make_demo_data.py --mode interaction --n-samples 1000
```

Generate a noisy dataset with 25% corrupted image-EHR pairings:

```powershell
python src/make_demo_data.py --mode noisy --n-samples 1000 --noise-rate 0.25
```

## Train MLP Encoder

Single-line PowerShell command:

```powershell
python src/train.py --ehr-encoder mlp --epochs 10 --batch-size 32 --num-workers 0 --output-dir outputs/example_mlp --checkpoint-dir checkpoints/example_mlp
```

Multiline PowerShell command:

```powershell
python src/train.py `
  --ehr-encoder mlp `
  --epochs 10 `
  --batch-size 32 `
  --num-workers 0 `
  --output-dir outputs/example_mlp `
  --checkpoint-dir checkpoints/example_mlp
```

## Train FT-Transformer Encoder

Single-line PowerShell command:

```powershell
python src/train.py --ehr-encoder ftt --epochs 10 --batch-size 32 --num-workers 0 --output-dir outputs/example_ftt --checkpoint-dir checkpoints/example_ftt
```

Multiline PowerShell command:

```powershell
python src/train.py `
  --ehr-encoder ftt `
  --epochs 10 `
  --batch-size 32 `
  --num-workers 0 `
  --output-dir outputs/example_ftt `
  --checkpoint-dir checkpoints/example_ftt
```

## Output Files

Training creates local output folders:

```text
outputs/
checkpoints/
```

These folders are intentionally ignored by Git because they contain generated experiment artifacts and model checkpoints.

## Public Repository Policy

The repository includes:

- Source code
- Synthetic metadata examples
- Documentation
- Aggregate results summary

The repository does not include:

- Real clinical data
- Medical images
- Model checkpoints
- Generated experiment outputs