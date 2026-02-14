# 🩸 LeukemiaScope - AI Blood Cell Analysis

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/good2idnan/medgemma-1.5-4b-it-leukemia-lora)
[![Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/chaudhrysuleman/Leukemia-AI)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)

**LeukemiaScope** is an AI-powered screening tool that analyzes microscopic blood cell images to detect signs of **Acute Lymphoblastic Leukemia (ALL)** using a fine-tuned [MedGemma](https://ai.google.dev/gemma/docs/medgemma) model.

Built for the [MedGemma Impact Challenge 2026](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle.

---

## 🌟 Advanced Agentic AI Demo

Check out our **Agentic AI** version, which features a multi-agent workflow with clinical reasoning and report generation:

👉 **[LeukemiaScope Agentic AI Demo](https://github.com/chaudhrysuleman/medgemma-1.5-4b-it-leukemia-demo)**  
_(Powered by LangGraph + MedGemma + Gemini)_

---

## 🎯 Why LeukemiaScope?

Leukemia is the most common childhood cancer. Early detection can improve 5-year survival rates from **20% to over 85%**. However, manual microscopic examination is time-consuming and requires specialized hematologists — a scarce resource in many parts of the world.

LeukemiaScope provides **instant AI-powered screening** to assist healthcare workers, especially in resource-limited settings.

---

## 📊 Model Performance

| Metric              | Value  |
| ------------------- | ------ |
| **Accuracy**        | 78.15% |
| **Leukemia Recall** | 83.10% |
| **Specificity**     | 68.83% |
| **F1-Score**        | 83.24% |
| **Training Images** | 9,701  |
| **Test Images**     | 1,867  |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A [HuggingFace](https://huggingface.co/) account and Access Token
- GPU recommended (runs on CPU but slower)

### 1. Clone the Repository

```bash
git clone https://github.com/good2idnan/LeukemiaScope-AI.git
cd LeukemiaScope-AI
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure HuggingFace Token

Create a `.env` file in the `app/` directory:

```bash
echo "HF_TOKEN=your_huggingface_token_here" > app/.env
```

> **Note:** You need a HuggingFace token with access to `google/medgemma-1.5-4b-it`. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 5. Run the App

```bash
cd app
python app.py
```

The app will launch at `http://127.0.0.1:7860` with a public share link.

---

## 🖥️ How to Use

1. **Upload** a microscopic blood cell image (PNG/JPG)
2. **Click** "🔬 Analyze Image"
3. **View** the AI prediction — Normal or Leukemia

### Sample Images

You can test with images from the [C-NMC Leukemia Dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification) on Kaggle.

---

## 📁 Repository Structure

```
leukemiascope/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # Apache 2.0
├── app/
│   ├── app.py                # Gradio web application
│   └── requirements.txt      # App-specific dependencies
├── notebooks/
│   ├── medgemma_leukemia_finetune_v2.ipynb   # Fine-tuning notebook
│   └── evaluate_finetuned_model_v2.ipynb     # Evaluation notebook
└── docs/
    └── model_card.md         # Model documentation
```

---

## 🔬 Technical Details

### Model Architecture

- **Base Model:** [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **LoRA Config:** r=16, alpha=64, dropout=0.05
- **Target Modules:** q_proj, k_proj, v_proj, o_proj
- **Training Data:** C-NMC Leukemia + Blood Cell Cancer datasets

### Classification Report

```
              precision    recall  f1-score   support
      Normal       0.68      0.69      0.69       648
    Leukemia       0.83      0.83      0.83      1219

    accuracy                           0.78      1867
```

### Reproduce Results

1. Open `notebooks/medgemma_leukemia_finetune_v2.ipynb` in Google Colab
2. Follow the steps to fine-tune the model
3. Use `notebooks/evaluate_finetuned_model_v2.ipynb` to evaluate

---

## ⚠️ Disclaimer

> **IMPORTANT: This tool is for educational and research purposes only.**
>
> LeukemiaScope is **NOT** a certified medical device or diagnostic tool. It should **never** be used as a substitute for professional medical diagnosis, advice, or treatment.
>
> - All predictions are AI-generated and may contain errors
> - False negatives (missed leukemia cases) are possible
> - False positives may cause unnecessary concern
> - Results **must** be confirmed by qualified healthcare professionals (hematologists/oncologists)
> - Do not make medical decisions based on this tool alone
>
> The developers assume no liability for any clinical decisions made using this tool.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

## 🏆 MedGemma Impact Challenge 2026

This project was built for the [Google MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge) on Kaggle.

| Resource             | Link                                                                              |
| -------------------- | --------------------------------------------------------------------------------- |
| **Fine-tuned Model** | [HuggingFace](https://huggingface.co/good2idnan/medgemma-1.5-4b-it-leukemia-lora) |
| **Live Demo**        | [HuggingFace Spaces](https://huggingface.co/spaces/chaudhrysuleman/Leukemia-AI)   |
| **Competition**      | [Kaggle](https://www.kaggle.com/competitions/med-gemma-impact-challenge)          |

---

## 👥 Team

- [good2idnan](https://github.com/good2idnan)
- [chaudhrysuleman](https://github.com/chaudhrysuleman)

## 🙏 Acknowledgments

- **Google Research** for the MedGemma model
- **Kaggle** for hosting the MedGemma Impact Challenge
- **C-NMC Dataset** creators for the training data
