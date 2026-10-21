# Customer Service Text Detoxification Benchmark & Curriculum Reinforcement Learning

This repository provides an end-to-end framework for synthesizing, stratifying, training, and evaluating text detoxification models in high-stakes customer-service domains. The project investigates curriculum reinforcement learning (PPO) structured across difficulty tiers, addresses length bias and clause stripping, and evaluates systems through both automatic metrics and blinded human assessments.

---

## Repository Structure

```text
customer-service-text-detoxification/
├── 01_data_generation/          # Phase 1: Benchmark synthesis, taxonomy calibration & LLM prompting
├── 02_data_preparation/         # Phase 2: Classifier audit, splitting, threshold calibration & stratification
├── 03_training/                 # Phase 3: SFT baseline & multi-seed PPO curriculum training (GPU)
├── 04_evaluation/               # Phase 4: Web-based human evaluation, gold labels & meta-evaluation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation & reproduction guide
```

---

## Installation & Environment Setup

### 1. Requirements
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (required for **Phase 03**, optional for other phases)

### 2. Install Dependencies
```bash
git clone https://github.com/your-username/customer-service-text-detoxification.git
cd customer-service-text-detoxification
pip install -r requirements.txt
```

### 3. Hardware Recommendations
| Module | Recommended Hardware | Execution Environment |
|---|---|---|
| `01_data_generation` | CPU / Local Machine | Local Terminal / Jupyter |
| `02_data_preparation` | CPU / Local Machine | Local Terminal / Jupyter |
| `03_training` | **High-VRAM GPU (T4 / V100 / A100)** | **Google Colab / Cloud GPU** |
| `04_evaluation` | CPU / Local Machine | Local Terminal / Jupyter |

---

## End-to-End Workflow & Reproduction Guide

### Phase 01: Benchmark Dataset Generation (`01_data_generation/`)
*This directory can be run on CPU / local machine.*

Generates and balances the domain-specific customer service benchmark corpus ($n = 9,631$ samples) spanning 8 industries, 3 registers, and 10 complaint subtypes:
1. **Pilot Study (`step1_pilot_estimation.py`)**: Computes empirical subtype distribution and reproduces Table 1.
2. **Round 1 Generation (`step2_round1_generation.py`)**: Synthesizes 7,755 samples using Claude 3 Haiku via OpenRouter.
3. **Round 2 Supplementary (`step3_round2_supplementary.py`)**: Synthesizes 2,533 candidate samples using Qwen via GroqCloud.
4. **Balancing & Assembly (`step4_balance_dataset.py`)**: Filters LLM refusals, removes duplicates, applies stratified downsampling, and outputs `data/final_corpus_9631.csv`.
5. **Corpus Audit (`table3_corpus_composition.ipynb`)**: Interactive notebook reproducing Table 3 and classifier calibration.

---

### Phase 02: Data Preparation & Stratification (`02_data_preparation/`)
*This directory can be run on CPU / local machine.*

Performs pre-trained classifier auditing, optimal threshold recalibration, and quantile difficulty tiering:
1. **ROC-AUC Classifier Audit (`step1_split_and_audit.py`)**: Audits `unitary/unbiased-toxic-roberta` on the full corpus and generates 55% Train ($5,297$), 15% Validation ($1,444$), and 30% Test ($2,890$) stratified splits (`data/01_full_splits/`).
2. **Optimal Threshold Recalibration (`step2_recalibrate_threshold.py`)**: Sweeps thresholds on the validation set to find $\tau_{\mathrm{opt}} = 0.0100$ maximizing Youden's $J$ index, filtering subsets with $s(x) \ge 0.0100$ (`data/02_filtered_splits/`).
3. **Quantile Difficulty Stratification (`step3_difficulty_stratification.py`)**: Partitions filtered toxic training samples into Easy (47.4%), Medium (31.1%), and Hard (21.5%) tiers (`data/03_stratified_benchmark/`).
4. **All-in-One Notebook (`data_preparation_pipeline.ipynb`)**: Runs the entire data preparation pipeline interactively.

---

### Phase 03: Model Training (`03_training/`)
> **IMPORTANT: Training MUST be performed on a GPU (Google Colab or Cloud GPU instance).**

This module implements Supervised Fine-Tuning (SFT) initialization and multi-seed Reinforcement Learning with PPO across a 3-phase curriculum (Easy $\to$ Medium $\to$ Hard).

#### Strict Training Sequence:
1. **Step 1: Supervised Fine-Tuning (SFT) Baseline First**
   - Open and run [`03_training/T5_base/t5_base_paradetox_sft.ipynb`](03_training/T5_base/t5_base_paradetox_sft.ipynb) on GPU.
   - Fine-tunes `t5-base` on ParaDetox (`s-nlp/paradetox`).
   - Serves as the initialization checkpoint for all subsequent PPO configurations.

2. **Step 2: Sequential RL Training Across Configurations**
   - Train the configuration folders sequentially:
     `RL_Config_A/` $\to$ `RL_Config_B/` $\to$ `RL_Config_C/` $\to$ `RL_Config_D/` $\to$ `RL_Config_E/` $\to$ `RL_Config_F/` $\to$ `RL_Config_G/` and `Flat_C/`.
   - In **each** configuration folder:
     - **Run Version 1 (`_v1.ipynb`) first**: Performs multi-seed curriculum PPO training across seeds `42`, `123`, and `777`.
     - **Run Version 2 (`_v2_check_post_val.ipynb`) next**: Runs post-validation diagnostics across fluency, semantic similarity, length ratio, and toxicity criteria.

3. **Step 3: Save All Checkpoints & Output Metrics**
   - **Crucial**: Ensure all training outputs are saved and downloaded from Colab:
     - Per-sample generation CSV files (`per_sample_seed*.csv`, `baseline_per_sample.csv`).
     - Evaluation metrics, reward logs, and checkpoint artifacts.
   - Master comparison: Run [`03_training/Paired_Curriculum_C_vs_Flat_C.ipynb`](03_training/Paired_Curriculum_C_vs_Flat_C.ipynb) to reproduce Table 9.

---

### Phase 04: Human Evaluation & Meta-Evaluation (`04_evaluation/`)
*This directory can be run on CPU / local machine.*

Evaluates system outputs using a rigorous blinded human annotation protocol and tests the validity of automated metrics.

#### Human Evaluation Pipeline & Data Lifecycle:
1. **Origin of Raw Outputs**:
   - The raw model outputs evaluated by humans are taken directly from **Folder 03**:
     - **SFT Baseline**: Greedy decoding outputs (`baseline_per_sample.csv`).
     - **Curriculum Config C (v1)**: Multi-seed outputs (`per_sample_seed*.csv`).
     - **Flat C Baseline (v1)**: Multi-seed outputs (`per_sample_seed*.csv`).
   - Stored under `04_evaluation/data_raw/`.

2. **Data Processing for Web Annotation (`04_evaluation/data_processed/`)**:
   - 100 representative toxic test inputs (33 Easy, 33 Medium, 34 Hard) were sampled and matched with outputs from SFT, Config C, and Flat C across seeds 42, 123, 777.
   - Outputs were deduplicated into 234 unique input–output pairs (`data_processed/task_b/task_b_pair_mapping_300.csv` and `task_b_web_annotation_234_pairs.csv`).

3. **Web-Based Human Annotation Protocol**:
   - Uploaded to a dedicated web annotation platform.
   - Three independent university annotators (IELTS 6.5–7.0) blinded to the generating systems assessed the samples:
     - **Task A (Toxicity Validation)**: Validates toxic vs. non-toxic labels on 240 samples.
     - **Task B (System Quality Assessment)**: Evaluates 234 pairs across 4 criteria:
       1. *Output Toxicity* (Binary: yes / no)
       2. *Content Preservation* (Ordinal: yes / partial / no)
       3. *Naturalness* (Binary: yes / no)
       4. *Request Retention* (Ordinal: yes / partial / no / no_clear_request)

4. **Annotation Reception & Adjudication (`04_evaluation/data_received/`)**:
   - Raw annotator sheets are stored in `data_received/task_a/` and `data_received/task_b/` (`*_A01.csv`, `*_A02.csv`).
   - Disagreements between A01 and A02 were exported (`disagreements_task_*.csv`) and adjudicated by a 3rd annotator (A03).

5. **Gold Label Consolidation (`04_evaluation/data_human_gold_label/`)**:
   - Adjudicated gold labels are consolidated into:
     - `task_a_goldlabel_240.csv` (240 validation samples).
     - `task_b_editbehavior_goldlabel_234.csv` (234 unique pairs).
     - `task_b_system_outputs_300_goldlabel.csv` (300 system outputs).
     - `task_b_system_outputs_300_goldlabel_with_metrics.csv` (300 outputs mapped with automatic metrics: STA, Sim, Flu, Joint, STAindep).

6. **Running Evaluation Notebooks**:
   - **Task A Evaluation ([`task_a_evaluation.ipynb`](04_evaluation/task_a_evaluation.ipynb))**:
     - Computes inter-annotator agreement and Cohen's $\kappa$ on prompted vs. adjudicated human labels.
     - Evaluates classifier ROC-AUC on prompted ($0.587$) vs. human gold labels ($0.560$).
     - Generates 2x2 contingency table ($77/174$ prompted-toxic marked non-toxic by humans).
   - **Task B Evaluation ([`task_b_evaluation.ipynb`](04_evaluation/task_b_evaluation.ipynb))**:
     - Computes inter-annotator agreement (unweighted $\kappa$ and quadratic weighted $\kappa$).
     - Calculates post-stratified population-weighted rates across difficulty bands (Table 10).
     - Evaluates head-to-head paired comparison between Config C and Flat C (76 Ties, 15 Flat C Wins, 9 Config C Wins).
     - Performs paired McNemar statistical significance tests ($p < 0.001$ for content, $p = 0.039$ for requests).
     - Computes 2,000-iteration cluster bootstrap 95% CI for Spearman $\rho$.
     - Conducts meta-evaluation of automatic metrics against human judgments (SLR sensitivity, ROC-AUC of Joint vs. length ratio).

---

## Citation & Contact

If you find this benchmark, training configurations, or evaluation protocols useful in your research, please cite our paper or contact the project authors.
