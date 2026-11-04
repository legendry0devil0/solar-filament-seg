# Solar Filament Segmentation 

This repository contains our notebook for solar-filament semantic segmentation and instance-level post-processing on the **MAGFiLO / Filament Segmentation Challenge 2026** dataset.

The pipeline combines three U-Net models, multi-scale test-time augmentation (TTA), logit-level ensemble fusion, curvilinear post-processing, and COCO Run-Length Encoding (RLE) output.

> **Training provenance:** The three models were trained in an earlier version of this notebook with `CFG.force_retrain_all=True`. In v22, the trained checkpoint files are loaded for validation and inference; v22 is therefore the inference/post-processing version of the workflow unless retraining is explicitly enabled. For an exact rerun, use those same checkpoint files together with the same dataset release.

## Method overview

### Dataset and split

- The notebook discovers the competition data through KaggleHub or standard Kaggle/local input directories.
- Training annotations are read from the MAGFiLO COCO-compatible JSON file.
- Images are converted to single-channel grayscale. Sixteen-bit images are converted to 8-bit by dividing by 256.
- Multiple annotator masks are retained as instance labels. Their binary foreground masks are averaged to form a soft consensus training target.
- The validation split is made by observation date, using `seed=42` and `val_frac=0.10`. Images from the same date are kept in the same split to reduce cross-station/date leakage.
- The test images are discovered from the test directory. The notebook expects the competition test images to be available in the attached dataset.

### Models

Three independent U-Net segmentation models are used:

| Model | Encoder | Decoder |
|---|---|---|
| Model 1 | `timm-efficientnet-b3` | U-Net |
| Model 2 | `resnet34` | U-Net |
| Model 3 | `tu-convnext_tiny` | U-Net with `scse` decoder attention |

Each model takes one-channel input and predicts one foreground logit map. The models are initialized with ImageNet encoder weights when training from scratch (`CFG.pretrained=True`).

### Training configuration

The principal settings in v22 are:

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Training crop | 768 × 768 |
| Crops sampled per image | 4 |
| Positive-centred crop probability | 70% |
| Image batch size | 2 |
| Gradient accumulation | 2 steps |
| Effective crop batch size | 16 |
| Optimizer | AdamW |
| Maximum learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Scheduler | OneCycleLR |
| OneCycle warm-up fraction | 0.10 |
| Gradient clipping | 1.0 |
| Epochs: EfficientNet-B3 | 20 |
| Epochs: ResNet-34 | 18 |
| Epochs: ConvNeXt-Tiny | 20 |
| Loss | 0.40 Lovasz + 0.35 Soft Dice + 0.25 BCE |

Each epoch evaluates the model on the complete held-out validation split. The checkpoint with the highest validation Panoptic Quality (PQ) is saved. The PQ matching threshold is IoU > 0.50.

Training crops are sampled at native pixel scale. Horizontal/vertical flips and rotations by 0°, 90°, 180°, or 270° are applied. Per-image intensity standardization is applied to inputs.

### Inference and ensemble

For each model, inference uses 12 TTA views: scales 1.0, 1.125, and 1.25, each evaluated with the original image, horizontal flip, vertical flip, and both flips. Predictions are inverse-transformed and resized to the original image dimensions.

The three models are combined using a weighted logit ensemble:

- EfficientNet-B3: 0.40
- ResNet-34: 0.20
- ConvNeXt-Tiny + scSE: 0.40

The sigmoid is applied after ensemble fusion to obtain the foreground probability map.

### Instance generation and post-processing

The probability map is converted into instance labels using the v22 morphology-based post-processing pipeline. Important defaults include:

- Base probability threshold: 0.53.
- Minimum component area: 450 pixels.
- Elliptical dilation/grouping radius: 10 pixels.
- Observatory adjustment: Bh, Lh, and Ch use a threshold reduced by 0.04 (with a floor of 0.48) and a minimum area reduced by 30 pixels.
- Thickness-aware spine recovery is enabled.
- Collinear endpoint gap bridging is enabled, with a maximum gap of 35 pixels and maximum orientation difference of 25°.
- Compactness/solidity filtering is enabled.
- A global density calibration ceiling of 5.5 predicted instances per test image is applied after candidate generation.

The validation grid search tests base thresholds `{0.53, 0.54, 0.55}` and minimum areas `{390, 420, 450}`. The selected configuration is written to `postproc.json` in the working directory. The exact selected values depend on the validation data and checkpoint weights used for that run.

## Requirements

The models were trained in an earlier notebook version with `CFG.force_retrain_all=True`. The v22 notebook uses the resulting checkpoints by default rather than treating its configured epoch counts as evidence of a fresh training run. The workflow was developed for a Kaggle GPU notebook environment. The recorded environment included PyTorch 2.11.0 and CUDA 12.8, with an NVIDIA Tesla T4 GPU used for execution. The notebook installs several missing packages at runtime, but a clean environment should have the following dependencies available:

- Python 3
- PyTorch
- `segmentation-models-pytorch`
- `timm`
- `numpy`
- `pandas`
- `opencv-python`
- `scipy`
- `scikit-image`
- `torchmetrics`
- `kagglehub`
- `pycocotools`
- `matplotlib`

The notebook does not pin exact versions for every dependency. For strict reproducibility, record the Python version, package versions, CUDA/cuDNN versions, GPU model, dataset version, and checkpoint hashes from the actual run, then create a lock file or `requirements.txt` with those exact versions. Do not assume that installing the latest versions will reproduce an earlier result bit-for-bit.

## Running on Kaggle

### 1. Prepare the Kaggle environment

1. Create a Kaggle Notebook and select a GPU accelerator (the original execution used a Tesla T4).
2. Enable Internet if you intend to download the competition data or pretrained encoder weights.
3. Attach the MAGFiLO / Filament Segmentation Challenge 2026 competition dataset to the notebook.
4. If competition data access is restricted, first accept the competition rules and confirm that your Kaggle account has access.

### 2. Add the notebook and checkpoints

Upload or copy `solar-filament-v22.ipynb` into the Kaggle Notebook.

For checkpoint-based inference, attach a Kaggle Dataset containing all three trained checkpoint files:

```text
best.pt
best_resnet34.pt
best_convnext_tiny.pt
```

If your checkpoint dataset is mounted elsewhere, either copy the files into `/kaggle/working/` before running the notebook or update the three `find_checkpoint_model*()` functions in the notebook to include your paths.

**Important:** The default `CFG.force_retrain_all` is `False`, because v22 is intended to load the models trained in the earlier notebook version. Make sure all three checkpoint files are attached and found before running inference. If you want to repeat training from scratch (rather than reproduce the v22 checkpoint-based run), set:

```python
CFG.force_retrain_all = True
```

To retrain an individual model instead, set the relevant flag:

```python
CFG.force_retrain_model1 = True  # EfficientNet-B3
CFG.force_retrain_model2 = True  # ResNet-34
CFG.force_retrain_model3 = True  # ConvNeXt-Tiny + scSE
```

Do not enable full retraining if you intend to reproduce the v22 inference run using the previously trained checkpoints. Keep the checkpoint files unchanged and record their hashes for provenance.

### 3. Run the notebook

Run all cells from top to bottom. The notebook will:

1. Install selected missing Python packages.
2. Locate the competition dataset and COCO annotation JSON.
3. Build annotator-specific instance-label maps and soft consensus masks.
4. Create the date-based training/validation split.
5. Load the three checkpoints produced by the earlier training run (or retrain only if you deliberately change the configuration flags).
6. Cache validation predictions and search the post-processing grid.
7. Evaluate validation PQ/SQ/RQ and Dice, produce error/ablation analyses, and generate explainability outputs.
8. Run inference on the test images, post-process predicted instances, and encode masks as COCO RLE.

The notebook uses `/kaggle/working` as its working directory when running on Kaggle. Intermediate masks, checkpoints, configuration files, plots, and generated outputs are written there.

### 4. Reproducibility checklist

For a reproducible experiment, keep the following fixed and archive them with the run:

- The exact training notebook revision used to produce the weights, as well as the inference notebook revision (`solar-filament-v22.ipynb`).
- The exact competition dataset version and annotation JSON.
- The three model checkpoint files produced by the earlier training run, including SHA-256 hashes; retain the training logs/configuration if available.
- The values of all `CFG` settings, especially seed, validation fraction, crop settings, epochs, TTA scales, and ensemble weights.
- The selected `postproc.json` generated by validation grid search.
- Python, PyTorch, SMP, timm, OpenCV, pycocotools, CUDA, and cuDNN versions.
- GPU model and Kaggle accelerator type.
- The notebook output/logs, including validation split counts and selected PQ configuration.

A fixed random seed improves repeatability but does not guarantee bitwise-identical results across different GPUs, CUDA/cuDNN versions, or library versions. The notebook also enables cuDNN benchmarking, so exact deterministic execution is not guaranteed.

## Outputs

The main inference product is a set of predicted filament instances represented as COCO RLE masks. The notebook also saves intermediate artifacts such as:

- `best.pt`, `best_resnet34.pt`, `best_convnext_tiny.pt` — model state dictionaries.
- `postproc.json` — validation-selected post-processing parameters.
- `ablation_study.csv` — ablation comparison results.
- Grad-CAM visualizations and validation plots.

The exact competition submission filename and schema should be taken from the competition's current sample-submission file and evaluator instructions. Verify the final output against that schema before submitting; do not assume that a local RLE artifact is automatically a valid submission file.

## Notes and limitations

- This notebook is an end-to-end research/competition workflow, not a packaged command-line application.
- The model outputs are semantic foreground probabilities. Instance identities are generated during post-processing using connected-component grouping and refinement.
- Results depend on the dataset release, checkpoint weights, validation split, and selected post-processing configuration.
- For a paper or benchmark report, distinguish the earlier training run from the v22 inference run. Report the actual metrics and training history from the run that produced the checkpoints; configured epoch counts in v22 alone are not proof of a fresh training run.

## Citation

If you use this implementation in research, cite the associated paper or technical report once available, and cite the MAGFiLO / Filament Segmentation Challenge dataset according to its official citation instructions.

## License

The repository is currently shared for research and documentation purposes while the MAGFiLO / Filament Segmentation Challenge 2026 is ongoing.

No open-source license has been assigned at this stage. All rights are reserved by the repository owner unless otherwise stated.

The dataset, competition materials, and any third-party model weights remain subject to their respective licenses and terms of use.

A suitable repository license may be added after the competition concludes.