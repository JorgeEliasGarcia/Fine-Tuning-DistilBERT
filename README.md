# Fine-Tuning DistilBERT for Multiple-Choice NLP Classification

This project focuses on **multiple-choice text classification** using **DistilBERT**, a lightweight transformer model distilled from BERT.  
The goal is to **fine-tune a pretrained DistilBERT model** to solve tasks where each prompt/context is paired with several candidate endings and the model must select the correct option.

The notebook covers the full workflow, including dataset preparation, tokenization, model setup for multiple-choice learning, training, evaluation, and simple inference on custom examples.

---

## Objectives
- Build an end-to-end pipeline for **multiple-choice NLP classification**.
- Fine-tune **DistilBERT** on a multiple-choice dataset.
- Use a model configuration suitable for multiple-choice inference.
- Train the model with appropriate hyperparameters.
- Evaluate performance and analyze results.

---

## Dataset
The notebook uses **SWAG (regular)**, a **Multiple-Choice Sentence Completion** dataset where each example includes:
- A context/prompt
- Several candidate endings (options)
- The correct label (index of the right option)

### Preprocessing
- Convert each sample to the input format expected for multiple-choice modeling.
- Jointly tokenize the context and each option.
- Apply padding and truncation.
- Produce a tokenized dataset compatible with the Hugging Face training pipeline.
- Use the dataset’s existing splits and work with small subsets for faster experimentation.

---

## Model

### DistilBERT
- A compressed version of BERT that retains strong language understanding while being more efficient.
- Transformer-based architecture suitable for fine-tuning on supervised NLP tasks.

### Multiple-choice adaptation
- Use a task-specific multiple-choice head to score each option.
- The model outputs one logit per option and selects the highest-scoring choice.

---

## Methodology
1. Import libraries and check GPU availability.
2. Load and inspect the SWAG dataset.
3. Preprocess and tokenize inputs (context + options).
4. Initialize a pretrained DistilBERT and adapt the classification head for the multiple-choice task.
5. Define training arguments and hyperparameters.
6. Train (fine-tune) the model using the Hugging Face Trainer API.
7. Evaluate using accuracy and compare predictions vs. ground-truth labels.
8. Run inference on custom examples and inspect outputs.
9. Review results and discuss observations.

---

## Evaluation
Model performance is evaluated with:
- **Accuracy** on the validation set
- Performance on the customized examples.

---

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- NumPy
- Scikit-learn
