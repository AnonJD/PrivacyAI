
## Instructions

### 1. Prepare Data

Download the following files and place them in the `data/` directory:

- `obfuscated_data_06.json`
- `pii_true_entities.csv`
- `original_transcripts.txt`
- `placeholder_locations_new.txt`

---

### 2. Dataset Creation

Run the following script to create the Base train set, Verifier train set, and Test set from CRAPII:

```bash
python mk_train.py
```

---

### 3. Inference on CRAPII Dataset

#### Using Presidio

To run inference using Presidio.

```bash
python presidio_inference.py trf
python presidio_inference.py lg
```

This runs the `trf` and `lg` variants on the test set.

#### Using Prompted GPT-4o-mini

Open and run all cells in:

- `prompted_gpt_inference.ipynb`

#### Using Fine-tuned GPT-4o-mini

To fine-tune the model on the CRAPII Base train set:

- Run all cells in `ft_gpt_training.ipynb`

Then, to evaluate the fine-tuned model on the CRAPII Test set:

- Run all cells in `ft_gpt_inference.ipynb`

---

### 4. Verifier Setup

To create a dataset and train the Verifier models:

- Run all cells in `verifier_inference.ipynb`

Then, to perform inference on the Test set:

- Run all cells in `verifier_training.ipynb`

---

### 5. Azure Model Inference

To run the Azure model on the CRAPII Test set:

- Run all cells in `azure_inference.ipynb`

---

### 6. TSCC Dataset

To create the TSCC dataset and split it into train/test:

```bash
python TSCC_dataset_creation.py
```

To fine-tune GPT-4o-mini on the TSCC train set:

- Run all cells in `gpt_ft_tscc.ipynb`

To run experiments on the TSCC dataset:

- Run all notebooks that start with `TSCC`
