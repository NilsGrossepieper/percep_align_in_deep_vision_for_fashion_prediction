# Evaluation of Perceptual Alignment in Deep Vision Models for Fashion Sales Prediction

## Project Overview

This repository contains the code and experiments developed for my Master’s thesis. The thesis investigates whether fine-tuning deep vision models using human similarity judgments (perceptual alignment) improves the quality of image embeddings for fashion sales prediction.

Deep vision models are aligned using human-annotated image triplets from both a large, domain-agnostic dataset (NIGHTS) and a small, fashion-specific dataset. The resulting image embeddings are combined with contextual metadata and used to train multiple sales prediction models. The repository includes code for perceptual alignment, embedding extraction, sales forecasting, and experiment evaluation. The deep vision models used in this thesis include ResNet-50, CLIP32, CLIP16, DINOv1, DINOv2, DINOv3 which can be fine-tuned either with MLP or LoRA. The prediction models used in this thesis are kNN regression, ridge regression, random forest regression and gradient boosting regression.

## Experimental Pipeline

The experiments in this repository follow a four-stage pipeline:
1. **Perceptual alignment**: Deep vision models are fine-tuned using human similarity judgments.
2. **Embedding extraction**: Fashion product images are embedded using the (fine-tuned or vanilla) vision models.
3. **Sales prediction**: Regression models are trained on the image embeddings together with contextual metadata.
4. **Statistical evaluation**: A paired bootstrap test is applied to compare prediction errors and assess whether perceptual alignment leads to statistically significant improvements.

The overall training process is illustrated below.

![Training Process](visualizations/training_process.png)

## Key Findings

- **Perceptual alignment improves fashion sales prediction**: Fine-tuning deep vision models on human similarity judgments leads to statistically significant improvements in sales forecasting accuracy.

- **General and domain-specific alignment both work**: Alignment using a large, domain-agnostic dataset (NIGHTS) as well as a very small fashion-specific dataset (190 triplets) improves downstream prediction performance, highlighting the sample efficiency of human judgment data.

- **Prediction model choice matters more than vision backbone**: Across all experiments, more expressive models such as Gradient Boosting and Random Forest outperform simpler models like kNN and Ridge Regression, regardless of the vision model used.

- **Nonlinear models benefit most from alignment**: Improvements from perceptually aligned embeddings are primarily captured by nonlinear prediction models, while simpler models show limited or no gains.

- **Smaller vision models benefit disproportionately**: Although larger vision models perform better in their vanilla form, perceptual alignment yields larger relative improvements for smaller models, narrowing or even closing the performance gap.

- **CNNs and Vision Transformers respond similarly**: Both architecture types benefit from perceptual alignment, suggesting that the effect is largely architecture-agnostic.

- **Fine-tuning strategy has limited impact**: No consistent advantage is observed between LoRA-based fine-tuning and MLP-based adaptation; the main gains stem from the alignment objective itself.

- **Early stopping is crucial**: Excessive fine-tuning on perceptual alignment datasets can harm downstream performance, whereas early stopping consistently leads to better generalization.

Below are some example prediction for some random fashion items for the DINOv1 model.

![Example Predictions](visualizations/prediction_examples_dinov1_lora.png)

## Repository Structure

```text
├── README.md
├── requirements.txt
├── config/                         # YAML configuration files for experiment settings
├── datasets/                       # Perceptual alignment datasets (NIGHTS, Fashion Triplets) and fashion sales datasets
├── notebooks/                      # Contains notebooks for visualizations
├── src/                            # Source code for data processing, training, and evaluation
├── visualizations/                 # Generated figures and visual assets
├── vision_model_training_script/   # Scripts for fine-tuning deep vision models (MLP / LoRA)
├── experiments/                    # Stored experiment results (created during execution, not tracked)
├── models/                         # Trained prediction models (created during execution, not tracked)
├── studies/                        # Hyperparameter optimization studies (created during execution, not tracked)
├── main_training_script.ipynb      # Trains sales prediction models on extracted embeddings
├── statistical_testing.ipynb       # Paired bootstrap tests for statistical significance
├── visualization.ipynb             # Creates figures and plots for analysis
└── wandb_key                       # Weights & Biases API key file (not tracked, required for logging)
```

## Setup

The experiments in this repository were primarily developed and executed on **Google Colab**. When running the code in a different environment, a few adaptations are required.

### 1. Environment
- The notebooks contain **Google Colab–specific code** (e.g., Colab paths and setup cells).
- If you do not use Google Colab, the **first cell of each notebook must be removed**, and the project root paths must be adapted accordingly.
- The Python scripts in `src/` are environment-agnostic and do not require modification.

### 2. Datasets
- The `datasets/` folder contains a notebook that loads and prepares:
  - the **NIGHTS** perceptual alignment dataset,
  - the **Visuelle2.0** fashion sales dataset.
- These datasets are required to run the experiments.
- The same notebook also prepares the **fashion triplet dataset** used in this thesis; however, this dataset is **not publicly available** and must be provided separately.

### 3. Vision Models
- The `vision_model_training_script/` directory contains a notebook that loads and stores all deep vision models used in this thesis (e.g., ResNet-50, CLIP, DINO variants).
- This step must be executed before embedding extraction or sales prediction experiments.

### 4. Experiment Tracking (Optional)
- Experiment tracking is performed using **Weights & Biases (W&B)**.
- To enable logging, a file named `wandb_key` containing a valid W&B API key must be created locally.
- This file is **not tracked** in the repository and can be omitted if experiment tracking is not required.

### 5. Dependencies
- All required Python packages are listed in `requirements.txt`.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt

## Usage

The experimental workflow in this repository is configuration-driven and mainly relies on three notebooks. All experiments are controlled via YAML configuration files located in the `config/` directory.

### 1. Fine-tune Vision Models
- Use the notebook in `vision_model_training_script/` to fine-tune deep vision models using perceptual alignment.
- The notebook loads the experiment configuration from a YAML file and performs fine-tuning based on the selected dataset, model architecture, and fine-tuning method (MLP or LoRA).
- The resulting models are stored locally and used for embedding extraction.

### 2. Train Sales Prediction Models
- Run `main_training_script.ipynb` to train sales prediction models on the extracted image embeddings.
- The notebook trains multiple prediction models (kNN, Ridge Regression, Random Forest, Gradient Boosting) based on the configuration settings.
- Experiment results and trained models are saved automatically.

### 3. Statistical Evaluation
- Use `statistical_testing.ipynb` to evaluate whether perceptual alignment leads to statistically significant improvements.
- The notebook applies a paired bootstrap test to compare prediction errors between aligned and vanilla embeddings and reports significance results.

### Configuration
- Each experiment is defined by a YAML configuration file specifying:
  - the vision model and fine-tuning strategy,
  - the alignment dataset,
  - the prediction model,
  - training and evaluation parameters.
- To run a new experiment, create or modify a configuration file and execute the corresponding notebook.

## Thesis Reference and Acknowledgements

This repository accompanies the Master’s thesis:

**Evaluating Perceptual Alignment in Deep Vision Models for Fashion Sales Prediction**  
- Author: Nils Großepieper 
- University: Eberhard Karls Universität Tübingen  
- Year: 2025

If you use this code or build upon this work, please cite the thesis accordingly.

### External Code and Datasets

This project builds upon the following publicly available resources:

- **DreamSim / NIGHTS dataset**  
  Parts of the perceptual alignment code and the NIGHTS dataset are based on the DreamSim repository by Sundaram et al.:  
  https://github.com/ssundaram21/dreamsim

- **Visuelle2.0 dataset**  
  Fashion sales prediction experiments use the Visuelle2.0 dataset provided by the Humatics Lab:  
  https://humaticslab.github.io/forecasting/visuelle

I gratefully acknowledge the authors of these resources for making their code and datasets publicly available.
