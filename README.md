# Named Entity Recognition with BERT-base-cased

## Description

This project implements Named Entity Recognition (NER) using a fine-tuned BERT-base-cased model with the Groningen Meaning Bank (GMB) dataset.
A Streamlit web interface is provided to allow users to input text and visualize recognized entities with color highlights, along with automatically generated fill-in-the-blank questions.

## Abstract

A sentence may contain various named entities such as people, locations, organizations, and more. Extracting these entities manually requires significant effort. NER with BERT provides an automated solution that improves efficiency.
In this research, BERT-base-cased was fine-tuned on the GMB dataset with multiple hyperparameter configurations. The best setup (learning rate 5e-5, batch size 32, and 2 epochs) achieved an F1-score of 0.8368.

## Features

* Fine-tuned BERT-base-cased for Named Entity Recognition (NER).
* Interactive Streamlit interface.
* Entity highlighting with distinct colors (PER, LOC, ORG, GPE, TIM, ART, EVE, NAT, etc.).
* Automatic generation of fill-in-the-blank style questions based on detected entities.

## Dataset

* **GMB (Groningen Meaning Bank)**

  * More than 10,000 documents
  * Over 1 million tokens
  * Rich set of NER tags: `PER`, `LOC`, `ORG`, `GEO`, `GPE`, `TIM`, `ART`, `EVE`, `NAT`, etc.

## Model

* **Model:** [HuggingFace – skripsi-bert-ner-Final\_Configuration\_XIII-model](https://huggingface.co/farizkuy/skripsi-bert-ner-Final_Configuration_XIII-model)
* **Base:** `bert-base-cased`
* **Best Hyperparameters:**

  * Learning rate: `5e-5`
  * Batch size: `32`
  * Epochs: `2`
* **Evaluation Metric:** `F1-score = 0.8368`

## Getting Started

### 1. Clone repository

```bash
git clone https://github.com/username/skripsi-ner-bert.git
cd skripsi-ner-bert
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run View.py
```

## Example Usage

**Input text:**

```
Andrew Malik started working at Google in Southern Canada this morning.
```

**Output:**

* Highlighted entities (`PER`, `ORG`, `LOC`, `TIM`, etc.).
* BIO format explanation.
* Auto-generated questions, e.g.:

  * "\_\_\_\_\_\_ started working at Google in Southern Canada this morning." (Entity: PER)
  * "Andrew Malik started working at \_\_\_\_\_\_ in Southern Canada this morning." (Entity: ORG)

## Project Structure

```
├── View.py              # Streamlit interface
├── NER_Modeling.py      # NER logic (tokenizer, model, question generator)
├── requirements.txt     # Project dependencies
├── README.md            # Documentation
```

## References

* Dao & Aizawa (2023): Impact of capitalization on NER performance.
* Liu & Ritter (2022): Dataset comparisons (CoNLL-2003 vs CoNLL++).
* Z. Liu et al. (2021): NER-BERT with domain-specific pretraining.

## Publication

This work is also available in paper format:
[Download Paper (Google Drive)](https://drive.google.com/file/d/1cRcMMScl7pJ14a_GgAy0ecq4xJ5OkX3w/view?usp=sharing)

---
