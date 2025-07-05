# 🧠 trbxai — Trustworthy & Explainable AI for Clinical and Synthetic Health Data

<p align="center">
  <picture>
    <!-- Logo per modalità dark -->
    <source srcset="assets/trbxAi_darkmode.svg" media="(prefers-color-scheme: dark)" />
    <!-- Logo per modalità light (default) -->
    <img src="assets/trbxAi_lightmode.svg" width="250" alt="trbxai logo" />
  </picture>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![PyPI](https://img.shields.io/badge/pypi-coming_soon-lightgrey)]()

> `trbxai` is a Python package for **interpretable analysis** of clinical and **synthetically generated** health data,  
> combining SHAP-based explainability and GAN-powered synthetic generation.  
> Designed for **clinical researchers**, **AI developers**, and scientists working with **imbalanced, rare, or sensitive datasets**.

---

## ✅ Key Features

- 🧬 **Synthetic Data Generation (ctGAN)**  
  Generate new, realistic samples based on real clinical datasets.

- 🧠 **Explainability with SHAP**  
  Visualize the most influential features in your model, per subject or globally.

- 📊 **SHAPSet Plot (coming soon)**  
  Visualize interactions between feature sets using `shapiq`.

- 💬 **Natural Language Output**  
  Narrative summaries explaining key predictors in plain language.

---

## 🔧 Installation

### Development version:
git clone https://github.com/DanteTrb/trbxai.git

pip install -e .

## 🚀 Quickstart

1. Generate synthetic patients

from trbxai.generate.ctgan_generator import generate_synthetic_patients
synthetic_df = generate_synthetic_patients(real_df, num_samples=100)
2. Explain model predictions

from trbxai.explain.shapset_explainer import get_shap_values
shap_values, explainer = get_shap_values(model, X, feature_names=X.columns.tolist())

## 🧪 Clinical Use Case
You have only 46 patients with hereditary cerebellar ataxia. You want to:
Augment the dataset with synthetic patients
Train a robust classifier
Understand which gait variables predict disease
With trbxai, you can do it — in just a few steps.

## 📁 Project Structure
trbxai/
├── generate/               # ctGAN module
│   └── ctgan_generator.py
├── explain/                # SHAP and explainability module
│   └── shapset_explainer.py
├── tests/                  # Unit tests with pytest
├── pyproject.toml
├── README.md
└── setup.cfg

## 🧪 Testing
To run all tests:
pytest

## 📘 Docs and Examples
🧪 Example notebooks (coming soon, stay tuned)

🧠 Streamlit app integration (in development)

📊 /examples folder will be available soon

## ❤️ Contributing
Open an issue, submit a pull request, or suggest new features!
If you use trbxai in your research, consider citing the repo (BibTeX coming soon).

## 📜 License
Distributed under the MIT License.

## 🔮 Vision
We aim to make trbxai the go-to library for Explainable AI in clinical research,
bridging data science and clinical decision-making with tools that are trustworthy, interpretable and deployable.
