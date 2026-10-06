# Mode emergence in a pretrained diffusion model (AFHQv2)

The application runs in three stages. Stages 1 and 2 need a GPU and were run as
Kaggle notebooks (2 x NVIDIA T4); stage 3 runs on a CPU.

| Stage | Script | Runs on | Output | Time |
|---|---|---|---|---|
| 1. Generation and embedding | `kaggle_generate.py` | Kaggle, GPU T4 x2, Internet on | `results_core.zip`, `results_images.zip` | about 4-6 h |
| 2. Tests | `kaggle_analysis.py` | Kaggle, GPU T4 x2 | `analysis.zip` | about 3.5 h |
| 3. Figures and tables | `make_results_afhq.py` | local CPU | `results/application_afhq_diffusion_final/` | about 15-20 min |

The outputs of stages 1 and 2 are about 1.2 GB and are not stored in the
repository. They are reproducible: every trajectory uses a fixed seed, the sampler
is deterministic, and the data splits use a fixed seed.

## Stage 1: generation and embedding (`kaggle_generate.py`)

1. Create a Kaggle notebook with Accelerator "GPU T4 x2" and Internet enabled.
2. Paste `kaggle_generate.py` into one cell and run it ("Save Version", "Save & Run All").
3. The script
   * downloads AFHQv2 (official StarGAN v2 release) and resizes it to 64 x 64 with Lanczos filtering;
   * splits the training images into a calibration half A and a half B (seed 20261001);
   * samples 5,000 trajectories from each of the official EDM AFHQv2 checkpoints
     (`edm-afhqv2-64x64-uncond-vp.pkl`, `edm-afhqv2-64x64-uncond-ve.pkl`) with the
     deterministic 40-step Heun sampler (sigma from 80 to 0.002, rho = 7), seeds 20270001-20275000;
   * records the denoised prediction at all 40 steps and the final image;
   * embeds all real and generated images with CLIP ViT-L/14 and FID Inception-V3
     (pytorch-fid 0.3.0), and fits a 50-component PCA on half A.
4. A preflight check runs every component on a few samples first and stops with
   `preflight_report.txt` if anything fails.
5. Download `results_core.zip` and `results_images.zip` from the notebook's Output tab.

## Stage 2: tests (`kaggle_analysis.py`)

1. Upload `modal_transport.py` and `mt_tools.py` (repository root) as a Kaggle dataset.
2. Create a Kaggle notebook with Accelerator "GPU T4 x2", and add as inputs the output
   of the stage-1 notebook and the dataset from step 1.
3. Paste `kaggle_analysis.py` into one cell and run it.
4. The script estimates the calibration, reference, and generated modes for both
   embeddings and d = 2, ..., 8, runs the matched-mode tolerance tests at all steps with
   Holm's correction, and runs MMD and energy permutation tests (4,999 permutations) on
   the GPU. The reference sample is half B together with the AFHQv2 test images.
5. Download `analysis.zip` from the Output tab.

## Stage 3: figures and tables (`make_results_afhq.py`)

1. Place the three files in `application_afhq_diffusion_final/inputs/`:
   ```
   inputs/results_core.zip
   inputs/results_images.zip
   inputs/analysis.zip
   ```
2. From the repository root, run
   ```
   python application_afhq_diffusion_final/make_results_afhq.py --workers 3
   ```
3. Outputs:
   ```
   results/application_afhq_diffusion_final/figures/clip/                 CLIP figures
   results/application_afhq_diffusion_final/figures/inception_appendix/   Inception figures
   results/application_afhq_diffusion_final/tables/                       all numbers behind the figures
   ```

The tolerance used throughout is delta = 0.10 s, where s is the smallest distance
between the modes of the calibration half A.

## Requirements

Stages 1 and 2: the default Kaggle Python image (PyTorch, torchvision, transformers,
diffusers, scikit-learn); `pytorch-fid` is installed by the script. Stage 3: numpy,
scipy, matplotlib, joblib, pandas.
