# Customer Service Text Detoxification Benchmark & Curriculum Reinforcement Learning

This repository provides an end-to-end framework for synthesizing, preparing, training, and evaluating text detoxification models in customer service domains. The workflow encompasses synthetic benchmark generation, classifier auditing and quantile difficulty stratification, curriculum reinforcement learning (PPO), and human evaluation protocols.

---

## Repository Structure

```text
customer-service-text-detoxification/
├── 01_data_generation/          # Phase 1: Benchmark synthesis and balancing
├── 02_data_preparation/         # Phase 2: Classifier audit, splitting, and difficulty stratification
├── 03_training/                 # Phase 3: SFT baseline and PPO reinforcement learning
├── 04_evaluation/               # Phase 4: Human evaluation and statistical analysis
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

---

## Installation & Environment Setup

### 1. Requirements
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (required for **Phase 03**, optional for other phases)

### 2. Install Dependencies
```bash
git clone https://github.com/roxannguyen/customer-service-text-detoxification.git
cd customer-service-text-detoxification
pip install -r requirements.txt
```

### 3. Hardware Recommendations
| Module | Recommended Hardware | Execution Environment |
|---|---|---|
| `01_data_generation` | CPU / Local Machine | Local Terminal / Jupyter |
| `02_data_preparation` | CPU / Local Machine | Local Terminal / Jupyter |
| `03_training` | **GPU (T4 / V100 / A100)** | **Google Colab / Cloud GPU** |
| `04_evaluation` | CPU / Local Machine | Local Terminal / Jupyter |

---

## Workflow Overview

### Phase 01: Benchmark Dataset Generation (`01_data_generation/`)
*Focus: Data synthesis, filtering, and corpus assembly across customer service domains.*

- **Files & Scripts**:
  - `step1_pilot_estimation.py`: Pilot estimation to determine distribution parameters across complaint subtypes.
  - `step2_round1_generation.py`: Initial generation round prompting LLMs across domains and scenarios.
  - `step3_round2_supplementary.py`: Supplementary generation focusing on under-represented complaint categories.
  - `step4_balance_dataset.py`: Filters refusals, removes duplicates, and balances the final benchmark corpus.
  - `table3_corpus_composition.ipynb`: Analyzes macro-label distributions and corpus composition.
- **Data Directory (`data/`)**: Contains intermediate generation rounds and the final assembled benchmark dataset.

---

### Phase 02: Data Preparation & Stratification (`02_data_preparation/`)
*Focus: Pre-trained classifier auditing, dataset partitioning, and difficulty tiering.*

- **Notebook**:
  - `data_preparation_pipeline.ipynb`: Interactive pipeline performing classifier auditing on the full corpus, stratified splitting, optimal threshold calibration via Youden's $J$, and quantile-based difficulty stratification.
- **Data Directory (`data/`)**:
  - `01_full_splits/`: Stratified train, validation, and test splits.
  - `02_filtered_splits/`: Subsets filtered by the calibrated toxicity threshold.
  - `03_stratified_benchmark/`: Final benchmark splits partitioned into difficulty bands (Easy, Medium, Hard).

---

### Phase 03: Model Training (`03_training/`)
*Focus: Supervised fine-tuning initialization and curriculum reinforcement learning.*

> **Note**: Training notebooks are designed to be executed on a GPU instance (e.g., Google Colab).

- **Workflow**:
  1. **Supervised Fine-Tuning (SFT)**:
     - `T5_base/t5_base_paradetox_sft.ipynb`: Fine-tunes the base model on parallel detoxification data to establish the baseline and initialize subsequent RL policies.
  2. **Reinforcement Learning Training (PPO)**:
     - Configuration directories (`RL_Config_A/` through `RL_Config_G/` and `Flat_C/`): Each contains training notebooks (`_v1.ipynb`) across multiple random seeds and post-training validation notebooks (`_v2_check_post_val.ipynb`) evaluating criteria such as fluency, similarity, and toxicity reduction.
  3. **Comparative Analysis**:
     - `Paired_Curriculum_C_vs_Flat_C.ipynb`: Direct comparison between curriculum-based and flat reinforcement learning configurations.
     - `Master_Evaluation_Final.ipynb`: Aggregated analysis across all trained configurations.

---

### Phase 04: Human Evaluation & Statistical Analysis (`04_evaluation/`)
*Focus: Blinded human annotation and statistical significance testing.*

- **Data Lifecycle**:
  - `data_raw/`: Collected generation outputs from baseline and candidate models selected for evaluation.
  - `data_processed/`: Sampled and deduplicated input–output pairs prepared for the annotation platform (`processdata.ipynb`).
  - `data_received/`: Raw blinded annotations collected across multiple annotators, including disagreement logs.
  - `data_human_gold_label/`: Consolidated gold-standard labels established through adjudication.
- **Evaluation Notebooks**:
  - `task_a_evaluation.ipynb`: Computes inter-annotator agreement and evaluates toxicity classifier performance against human judgments (Task A).
  - `task_b_evaluation.ipynb`: Assesses detoxification quality (content preservation, naturalness, request retention), performs paired McNemar tests, bootstrap confidence intervals, and correlation analysis against automated metrics (Task B).

---

## Citation & Contact

For questions regarding this benchmark, training configurations, or evaluation protocols, please open an issue in this repository or contact the authors.
