# AI Mini Project: Spam SMS Classification

**An end-to-end NLP pipeline that learns to tell legitimate text messages apart from spam.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-NLP-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## About

- **What:** A Jupyter-notebook walkthrough that trains and compares classical ML classifiers on the UCI SMS Spam Collection.
- **Who:** Built by Gyanesh Samanta as a solo coursework mini-project.
- **When:** First commit 2022-04-18, last touched 2026-03-28.
- **Where:** Submitted as part of an undergraduate Artificial Intelligence course.
- **Why:** To get hands-on with the full text-classification pipeline — preprocessing, vectorization, model selection, and evaluation — on a real, messy, real-world dataset.

## The Story

The dataset bundled here is the classic **UCI SMS Spam Collection**: **5,574 raw SMS messages**, each tagged `ham` (legitimate) or `spam`. The class split is famously skewed — only roughly one in seven messages is spam — which makes plain accuracy a misleading metric and pushes the project toward precision/recall and confusion-matrix thinking.

The notebook walks through the standard text pipeline: lower-casing, tokenization, stop-word removal and lemmatization, then turning the cleaned tokens into numeric features with both Count and TF-IDF vectorizers. From there it pits **Multinomial Naive Bayes**, **Random Forest**, and **SVM** against each other, looking at accuracy, precision, recall, and F1 to figure out which model handles the imbalance best.

The takeaway is that simple, linear models on top of TF-IDF features remain a remarkably strong baseline for short-text spam detection — and that the failure modes (false positives on promotional ham, false negatives on novel spam phrasings) are far more interesting than the headline accuracy number.

---

## Tech Stack

- **Language:** Python 3
- **ML / NLP:** scikit-learn, NLTK
- **Data:** pandas, NumPy
- **Viz:** Matplotlib, Seaborn
- **Notebook:** Jupyter

## Repo Structure

```
AI-mini-project/
├── Spam SMS Classication.ipynb   # full walkthrough notebook
├── dataset/
│   └── Spam SMS Collection       # 5,574 labelled SMS messages
└── README.md
```

## Getting Started

```bash
# 1. Clone
git clone https://github.com/GyaneshSamanta/AI-mini-project.git
cd AI-mini-project

# 2. Install dependencies
pip install jupyter scikit-learn pandas numpy nltk matplotlib seaborn

# 3. Download NLTK corpora (first run only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 4. Launch the notebook
jupyter notebook "Spam SMS Classication.ipynb"
```

Run the cells top to bottom to reproduce the preprocessing, feature extraction, training, and evaluation results.

## Contributing

This is an archived coursework project, but bug fixes and clean-up PRs are welcome. Please open an issue first to discuss any larger changes.

## License

Released under the MIT License. The bundled SMS dataset is the UCI SMS Spam Collection and retains its original terms of use for research.

## Credits

- **Author:** Gyanesh Samanta ([@GyaneshSamanta](https://github.com/GyaneshSamanta))
- **Dataset:** [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
