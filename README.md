# Insurance Reviews NLP Dashboard

Interactive Streamlit application for exploring and analyzing insurance customer reviews with NLP.

## Features

The app includes 5 main sections:

- **Prediction**
  - Predicts the **sentiment** of a review
  - Predicts the **rating**
  - Predicts the **main topic**
  - Displays simple feature-based explanations for each prediction

- **Insurer Analysis**
  - Number of reviews per insurer
  - Average rating
  - Dominant sentiment and dominant topic
  - Sentiment distribution
  - Topic distribution
  - Average rating by topic
  - Representative reviews

- **Review Search**
  - Semantic search over the review corpus
  - Filters by insurer, sentiment, topic, and rating

- **Summary & Explanation**
  - Summary by insurer or by topic
  - Top terms
  - Representative reviews

- **QA / Ask the dataset**
  - Natural language question answering over the review dataset
  - Retrieves the most relevant reviews as evidence

---

## Project file required

Place the following file in the same folder as the app:

- `insurance_reviews_topics.csv`

The application also checks for this alternative path:

- `/mnt/data/insurance_reviews_topics.csv`

---

## Expected columns in the CSV

The app works best if the dataset contains these columns:

- `avis_en`
- `avis_spell_corrected`
- `text_clean`
- `sentiment`
- `note`
- `assureur`
- `lda_topic_label`

Optional but supported:

- `type` (if present with values such as `train` / `test`, the app preferentially trains on the `train` subset)

---

## Installation

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

If `python` does not work in your terminal, try:

```bash
py -m pip install -r requirements.txt
```

---

## Launch the application

Run the following command in the folder containing `streamlit_app.py`:

```bash
python -m streamlit run streamlit_app.py
```

Alternative on Windows:

```bash
py -m streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Notes about the models

The app trains lightweight NLP models automatically when it starts:

- **TF-IDF + Logistic Regression** for sentiment prediction
- **TF-IDF + Logistic Regression** for rating prediction
- **TF-IDF + Logistic Regression** for topic prediction
- **TF-IDF semantic search** for retrieval and QA support

No external model files are required.

---

## Troubleshooting

### 1. `python` is not recognized
Use:

```bash
py -m streamlit run streamlit_app.py
```

or open **Anaconda Prompt** and run the same command.

### 2. `streamlit` is not recognized
Use:

```bash
python -m streamlit run streamlit_app.py
```

instead of:

```bash
streamlit run streamlit_app.py
```

### 3. File not found error
Make sure `insurance_reviews_topics.csv` is in the same directory as `streamlit_app.py`.

### 4. App opens but no prediction works
Check that your CSV contains the required columns listed above.

---

## GitHub

You can find the github link here : 



## Authors

Made by **Sandy TEFOUEGOUM & Harold TAGNY**
