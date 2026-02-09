# Fine-Tuning DistilBERT for Multiple-Choice NLP Classification

## Overview
This project focuses on **multiple-choice text classification** using **DistilBERT**, a lightweight transformer model distilled from BERT.  
The goal is to **fine-tune a pretrained DistilBERT model** to solve tasks where each question is paired with several candidate answers and the model must select the correct option.

The notebook covers the full workflow, including dataset preparation, tokenization, model adaptation for multiple-choice learning, training, and evaluation.

---

## Objectives
- Build an end-to-end pipeline for **multiple-choice NLP classification**.
- Fine-tune **DistilBERT** on a multiple-choice dataset.
- Adapt the model head for multiple-choice inference.
- Train the model with appropriate hyperparameters.
- Evaluate performance and analyze results.

---

## Dataset
The dataset is designed for **Multiple-Choice Question Answering / Classification**, where each example includes:
- A prompt/question
- Several answer options
- The correct label (index of the right option)

### Preprocessing
- Convert samples to the input format expected by DistilBERT.
- Jointly tokenize prompts/questions and each option.
- Apply padding and truncation to a maximum sequence length.
- Create PyTorch-ready tensors.
- Split data into training and evaluation sets.

---

## Model

### DistilBERT
- A compressed version of BERT that retains strong language understanding while being more efficient.
- Transformer-based architecture suitable for fine-tuning on supervised NLP tasks.

### Multiple-choice adaptation
- A task-specific classification head is used to score each option.
- The model outputs one logit per option and selects the highest-scoring choice.

---

## Methodology
1. Import libraries and check GPU availability.
2. Load and inspect the dataset.
3. Preprocess and tokenize inputs (question + options).
4. Initialize a pretrained DistilBERT model.
5. Configure the model for multiple-choice learning.
6. Define training arguments and hyperparameters.
7. Train (fine-tune) the model.
8. Evaluate using accuracy and compare predictions vs. ground-truth labels.
9. Review results and discuss observations.

---

## Evaluation
Model performance is evaluated with:
- **Accuracy** on the validation/test set
- Error inspection (examples where the model fails)

---

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Datasets
- NumPy
- Scikit-learn

---

## Project Structure
```text
├── Fine-tuning de DistilBERT para Tareas de Elección Múltiple.ipynb
└── README.md
