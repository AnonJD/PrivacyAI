# Enhancing the De-identification of Personally Identifiable Information in Educational Data

[![arXiv](https://img.shields.io/badge/arXiv-2501.09765-b31b1b.svg)](https://arxiv.org/abs/2501.09765)

> Code for our experiments on PII detection using Presidio, Azure AI Language, prompted GPT-4o-mini, fine-tuned GPT-4o-mini, and Verifier models on the CRAPII and TSCC datasets.

---

## 0. Datasets

### CRAPII (Cleaned Repository of Annotated PII)
- **Kaggle dataset**: https://www.kaggle.com/datasets/langdonholmes/cleaned-repository-of-annotated-pii/data  
- **Paper**: https://educationaldatamining.org/edm2024/proceedings/2024.EDM-posters.88/index.html  
- **Citation**:  
  Holmes, L., Crossley, S. A., Wang, J., & Zhang, W. (2024). *Cleaned Repository of Annotated PII*. Proceedings of the 17th International Conference on Educational Data Mining (EDM).

### TSCC
To access the TSCC dataset, see the paper (instructions inside for how to request the data):  
https://ecp.ep.liu.se/index.php/sltc/article/view/575

---

## 1. Prepare Data

Download the following files and place them in the `data/` directory:

- `obfuscated_data_06.json`
- `pii_true_entities.csv`
- `original_transcripts.txt`
- `placeholder_locations_new.txt`

---

## 2. Dataset Creation

Create the **Base Train Set**, **Verifier Train Set**, and **Test Set** from CRAPII:

```bash
python mk_train.py
```

---

## 3. Inference on the CRAPII Dataset

### 3.1 Presidio

Run inference with Presidio (`en_core_web_lg` and `en_core_web_trf`) on the **Test Set**:

```bash
python presidio_inference.py lg 
python presidio_inference.py trf
```

### 3.2 Azure AI Language

Run the Azure model on the CRAPII **Test Set**:

- Run all cells in `azure_inference.ipynb`

### 3.3 Prompted GPT-4o-mini

Open and run all cells in:

- `prompted_gpt_inference.ipynb`

### 3.4 Fine-tuned GPT-4o-mini

Fine-tune on the CRAPII **Base Train Set**:

- Run all cells in `ft_gpt_training.ipynb`

Then evaluate on the CRAPII **Test Set**:

- Run all cells in `ft_gpt_inference.ipynb`

### 3.5 Verifier Setup

Create the dataset and train the Verifier models:

- Run all cells in `verifier_training.ipynb`

Then perform inference on the **Test Set**:

- Run all cells in `verifier_inference.ipynb`

---

## 4. TSCC Dataset

Create and split the TSCC dataset into train/test:

```bash
python TSCC_dataset_creation.py
```

Fine-tune GPT-4o-mini on the TSCC train split:

- Run all cells in `gpt_ft_tscc.ipynb`

Run experiments on TSCC:

- Execute all notebooks that start with `TSCC`

---

## 5. Paper

If you use this repository, please cite our paper:  
**arXiv:** https://arxiv.org/abs/2501.09765

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
