import os
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Insurance Reviews NLP App", layout="wide")

DATA_CANDIDATES = [
    "insurance_reviews_topics.csv",
    "/mnt/data/insurance_reviews_topics.csv",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    path = None
    for candidate in DATA_CANDIDATES:
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        raise FileNotFoundError("insurance_reviews_topics.csv not found.")

    df = pd.read_csv(path)
    # Basic cleanup / fallbacks
    for col in ["avis_en", "avis_spell_corrected", "text_clean", "sentiment", "note", "assureur", "lda_topic_label"]:
        if col not in df.columns:
            df[col] = np.nan

    df["text_app"] = (
        df["avis_spell_corrected"].fillna("").astype(str).str.strip()
    )
    mask_empty = df["text_app"].eq("")
    df.loc[mask_empty, "text_app"] = df.loc[mask_empty, "avis_en"].fillna("").astype(str).str.strip()
    mask_empty = df["text_app"].eq("")
    df.loc[mask_empty, "text_app"] = df.loc[mask_empty, "text_clean"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["text_app", "sentiment", "note", "lda_topic_label", "assureur"]).copy()
    df = df[df["text_app"].str.len() > 0].copy()
    df["note"] = pd.to_numeric(df["note"], errors="coerce")
    df = df.dropna(subset=["note"]).copy()
    df["note"] = df["note"].astype(int)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    if "type" in df.columns and df["type"].astype(str).isin(["train", "test"]).any():
        train_df = df[df["type"].astype(str) == "train"].copy()
        if len(train_df) < 100:
            train_df = df.copy()
    else:
        train_df = df.copy()

    X = train_df["text_app"]

    sentiment_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    sentiment_pipe.fit(X, train_df["sentiment"])

    note_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    note_pipe.fit(X, train_df["note"])

    topic_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=3, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    topic_pipe.fit(X, train_df["lda_topic_label"])

    search_vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), stop_words="english")
    search_matrix = search_vectorizer.fit_transform(df["text_app"])

    return {
        "sentiment": sentiment_pipe,
        "note": note_pipe,
        "topic": topic_pipe,
        "search_vectorizer": search_vectorizer,
        "search_matrix": search_matrix,
    }


def short_text(text: str, max_len: int = 220) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "have", "from", "they", "were", "been", "their", "into",
    "about", "would", "could", "there", "than", "them", "when", "what", "where", "which", "because", "while",
    "insurance", "insurer", "review", "reviews", "customer", "customers", "company", "companies",
}


def top_terms_from_subset(texts: pd.Series, n: int = 12) -> List[str]:
    texts = texts.dropna().astype(str)
    if texts.empty:
        return []
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1]
    result = []
    for idx in order:
        term = terms[idx]
        if term.lower() not in STOPWORDS:
            result.append(term)
        if len(result) >= n:
            break
    return result


def representative_reviews(subset: pd.DataFrame, n: int = 3) -> List[str]:
    if subset.empty:
        return []
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(subset["text_app"].astype(str))
    centroid = np.asarray(X.mean(axis=0))
    sims = cosine_similarity(X, centroid).ravel()
    top_idx = np.argsort(sims)[::-1][: min(n, len(subset))]
    return [short_text(subset.iloc[i]["text_app"], 260) for i in top_idx]


def extract_feature_explanation(pipe: Pipeline, text: str, predicted_label, top_n: int = 8) -> List[Tuple[str, float]]:
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    x = vec.transform([text])
    feature_names = np.array(vec.get_feature_names_out())
    nz = x.nonzero()[1]
    if len(nz) == 0:
        return []

    if len(clf.classes_) == 2:
        class_idx = 1 if predicted_label == clf.classes_[1] else 0
        coefs = clf.coef_[0]
        contrib = x.toarray()[0][nz] * (coefs[nz] if class_idx == 1 else -coefs[nz])
    else:
        class_idx = list(clf.classes_).index(predicted_label)
        coefs = clf.coef_[class_idx]
        contrib = x.toarray()[0][nz] * coefs[nz]

    pairs = sorted(zip(feature_names[nz], contrib), key=lambda z: z[1], reverse=True)
    return [(term, float(score)) for term, score in pairs[:top_n] if score > 0]


def semantic_search(query: str, df: pd.DataFrame, vectorizer: TfidfVectorizer, matrix, top_k: int = 5,
                    insurer: Optional[str] = None, sentiment: Optional[str] = None,
                    topic: Optional[str] = None, note_values: Optional[List[int]] = None) -> pd.DataFrame:
    filt = pd.Series(True, index=df.index)
    if insurer and insurer != "All":
        filt &= df["assureur"] == insurer
    if sentiment and sentiment != "All":
        filt &= df["sentiment"] == sentiment
    if topic and topic != "All":
        filt &= df["lda_topic_label"] == topic
    if note_values:
        filt &= df["note"].isin(note_values)

    subset = df[filt].copy()
    if subset.empty:
        return subset

    subset_idx = subset.index.to_numpy()
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix[subset_idx]).ravel()
    subset = subset.assign(similarity=sims).sort_values("similarity", ascending=False)
    return subset.head(top_k)


def answer_question(question: str, base_df: pd.DataFrame, retrieved: pd.DataFrame) -> str:
    q = question.lower()
    subset = retrieved if len(retrieved) > 0 else base_df
    if subset.empty:
        return "I could not find matching reviews in the current filters."

    count = len(subset)
    avg_rating = subset["note"].mean()
    top_sentiment = subset["sentiment"].mode().iloc[0]
    top_topic = subset["lda_topic_label"].mode().iloc[0]

    if any(k in q for k in ["average rating", "avg rating", "mean rating", "rating"]) and "why" not in q:
        return f"Average rating in the retrieved set is {avg_rating:.2f}/5 based on {count} reviews."
    if "sentiment" in q or "positive" in q or "negative" in q or "neutral" in q:
        distr = (subset["sentiment"].value_counts(normalize=True) * 100).round(1).to_dict()
        parts = ", ".join(f"{k}: {v}%" for k, v in distr.items())
        return f"Dominant sentiment is {top_sentiment}. Distribution in the retrieved set: {parts}."
    if "topic" in q or "theme" in q or "issue" in q or "problem" in q:
        terms = top_terms_from_subset(subset["text_app"], n=6)
        return f"The dominant topic is '{top_topic}'. Common supporting terms are: {', '.join(terms)}."
    if "insurer" in q or "company" in q:
        insurer_counts = subset["assureur"].value_counts().head(5)
        parts = ", ".join(f"{name} ({cnt})" for name, cnt in insurer_counts.items())
        return f"Most represented insurers in the retrieved reviews are: {parts}."
    if "summary" in q or "summarize" in q or "what do people say" in q:
        reps = representative_reviews(subset, n=2)
        terms = top_terms_from_subset(subset["text_app"], n=6)
        return (
            f"Across {count} matching reviews, the average rating is {avg_rating:.2f}/5. "
            f"The most common sentiment is {top_sentiment}, and the dominant topic is {top_topic}. "
            f"Recurring terms include {', '.join(terms)}. "
            f"Representative feedback: {reps[0] if reps else ''}"
        )
    return (
        f"In the {count} most relevant reviews, average rating is {avg_rating:.2f}/5, "
        f"dominant sentiment is {top_sentiment}, and the main topic is {top_topic}."
    )


def page_footer() -> None:
    st.markdown("---")
    st.caption("Made by Sandy TEFOUEGOUM & Harold TAGNY")


def prediction_card(title: str, label, probas: Optional[np.ndarray], classes: List) -> None:
    st.subheader(title)
    st.metric("Prediction", str(label))
    if probas is not None:
        tmp = pd.DataFrame({"Label": [str(c) for c in classes], "Probability": probas})
        fig = px.bar(tmp, x="Label", y="Probability", text="Probability")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Insurance Reviews NLP Dashboard")
    st.caption("Prediction, insurer analysis, search, summary, and QA over insurance customer reviews.")

    df = load_data()
    assets = train_models(df)

    sentiment_pipe = assets["sentiment"]
    note_pipe = assets["note"]
    topic_pipe = assets["topic"]
    search_vectorizer = assets["search_vectorizer"]
    search_matrix = assets["search_matrix"]

    insurers = ["All"] + sorted(df["assureur"].dropna().unique().tolist())
    sentiments = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
    topics = ["All"] + sorted(df["lda_topic_label"].dropna().unique().tolist())
    notes = sorted(df["note"].dropna().unique().tolist())

    tabs = st.tabs(["Prediction", "Insurer Analysis", "Review Search", "Summary & Explanation", "QA / Ask the dataset"])

    with tabs[0]:
        st.header("Prediction")
        user_text = st.text_area("Enter a review", height=180, placeholder="Paste a customer review here...")
        if st.button("Run prediction"):
            if not user_text.strip():
                st.warning("Please enter a review first.")
            else:
                col1, col2, col3 = st.columns(3)

                sent_pred = sentiment_pipe.predict([user_text])[0]
                sent_proba = sentiment_pipe.predict_proba([user_text])[0]
                note_pred = note_pipe.predict([user_text])[0]
                note_proba = note_pipe.predict_proba([user_text])[0]
                topic_pred = topic_pipe.predict([user_text])[0]
                topic_proba = topic_pipe.predict_proba([user_text])[0]

                with col1:
                    prediction_card("Sentiment", sent_pred, sent_proba, list(sentiment_pipe.named_steps["clf"].classes_))
                with col2:
                    prediction_card("Rating", note_pred, note_proba, list(note_pipe.named_steps["clf"].classes_))
                with col3:
                    prediction_card("Topic", topic_pred, topic_proba, list(topic_pipe.named_steps["clf"].classes_))

                st.subheader("Explanation")
                exp_sent = extract_feature_explanation(sentiment_pipe, user_text, sent_pred)
                exp_note = extract_feature_explanation(note_pipe, user_text, note_pred)
                exp_topic = extract_feature_explanation(topic_pipe, user_text, topic_pred)
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.markdown("**Sentiment clues**")
                    st.write(pd.DataFrame(exp_sent, columns=["Term", "Contribution"]) if exp_sent else "No strong clues found.")
                with e2:
                    st.markdown("**Rating clues**")
                    st.write(pd.DataFrame(exp_note, columns=["Term", "Contribution"]) if exp_note else "No strong clues found.")
                with e3:
                    st.markdown("**Topic clues**")
                    st.write(pd.DataFrame(exp_topic, columns=["Term", "Contribution"]) if exp_topic else "No strong clues found.")
        page_footer()

    with tabs[1]:
        st.header("Insurer Analysis")
        insurer = st.selectbox("Choose an insurer", sorted(df["assureur"].unique().tolist()))
        subset = df[df["assureur"] == insurer].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reviews", len(subset))
        c2.metric("Average rating", f"{subset['note'].mean():.2f}")
        c3.metric("Dominant sentiment", subset["sentiment"].mode().iloc[0])
        c4.metric("Dominant topic", subset["lda_topic_label"].mode().iloc[0])

        st.subheader("Insurer Summary")
        terms = top_terms_from_subset(subset["text_app"], 8)
        reps = representative_reviews(subset, 3)
        st.write(
            f"{insurer} has {len(subset)} reviews with an average rating of {subset['note'].mean():.2f}/5. "
            f"The dominant sentiment is {subset['sentiment'].mode().iloc[0]}, and the main topic is {subset['lda_topic_label'].mode().iloc[0]}. "
            f"Frequent terms include: {', '.join(terms)}."
        )

        left, right = st.columns(2)
        with left:
            sent_counts = subset["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            st.plotly_chart(px.bar(sent_counts, x="Sentiment", y="Count", title="Sentiment distribution"), use_container_width=True)
        with right:
            topic_counts = subset["lda_topic_label"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]
            st.plotly_chart(px.bar(topic_counts, x="Topic", y="Count", title="Topic distribution"), use_container_width=True)

        rating_by_topic = subset.groupby("lda_topic_label", as_index=False)["note"].mean().sort_values("note", ascending=False)
        st.plotly_chart(px.bar(rating_by_topic, x="lda_topic_label", y="note", title="Average rating by topic"), use_container_width=True)

        st.subheader("Representative reviews")
        for i, r in enumerate(reps, 1):
            st.markdown(f"**Review {i}:** {r}")
        page_footer()

    with tabs[2]:
        st.header("Review Search")
        q = st.text_input("Search query", placeholder="e.g. customer service delay, price increase, reimbursement")
        col1, col2, col3, col4 = st.columns(4)
        insurer_f = col1.selectbox("Insurer", insurers)
        sentiment_f = col2.selectbox("Sentiment", sentiments)
        topic_f = col3.selectbox("Topic", topics)
        notes_f = col4.multiselect("Rating", notes)
        top_k = st.slider("Number of results", 3, 20, 8)

        if st.button("Search reviews"):
            if not q.strip():
                st.warning("Please enter a search query.")
            else:
                results = semantic_search(q, df, search_vectorizer, search_matrix, top_k, insurer_f, sentiment_f, topic_f, notes_f)
                if results.empty:
                    st.info("No results found for the current query and filters.")
                else:
                    st.success(f"{len(results)} result(s) found.")
                    view = results[["assureur", "sentiment", "note", "lda_topic_label", "similarity", "text_app"]].copy()
                    view["text_app"] = view["text_app"].apply(lambda x: short_text(x, 260))
                    st.dataframe(view, use_container_width=True)
        page_footer()

    with tabs[3]:
        st.header("Summary & Explanation")
        mode = st.radio("Summarize by", ["Insurer", "Topic"], horizontal=True)
        if mode == "Insurer":
            choice = st.selectbox("Choose an insurer", sorted(df["assureur"].unique().tolist()), key="sum_insurer")
            subset = df[df["assureur"] == choice].copy()
        else:
            choice = st.selectbox("Choose a topic", sorted(df["lda_topic_label"].unique().tolist()), key="sum_topic")
            subset = df[df["lda_topic_label"] == choice].copy()

        terms = top_terms_from_subset(subset["text_app"], 12)
        reps = representative_reviews(subset, 3)
        st.subheader("Summary")
        st.write(
            f"The selected subset contains {len(subset)} reviews. Average rating is {subset['note'].mean():.2f}/5. "
            f"Dominant sentiment is {subset['sentiment'].mode().iloc[0]}. Most frequent topic is {subset['lda_topic_label'].mode().iloc[0]}."
        )

        st.subheader("Top terms")
        st.write(", ".join(terms) if terms else "No terms available.")

        st.subheader("Representative reviews")
        for i, r in enumerate(reps, 1):
            st.markdown(f"**Review {i}:** {r}")
        page_footer()

    with tabs[4]:
        st.header("QA / Ask the dataset")
        st.caption("Ask a question in natural language. The app retrieves relevant reviews and produces a short answer from the data.")
        qa_q = st.text_input("Your question", placeholder="e.g. What do people complain about for Direct Assurance?", key="qa_question")
        qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
        qa_insurer = qa_col1.selectbox("Insurer filter", insurers, key="qa_ins")
        qa_sent = qa_col2.selectbox("Sentiment filter", sentiments, key="qa_sent")
        qa_topic = qa_col3.selectbox("Topic filter", topics, key="qa_topic")
        qa_notes = qa_col4.multiselect("Rating filter", notes, key="qa_notes")
        qa_topk = st.slider("Retrieved reviews", 3, 15, 5, key="qa_topk")

        if st.button("Answer the question"):
            if not qa_q.strip():
                st.warning("Please type a question first.")
            else:
                retrieved = semantic_search(qa_q, df, search_vectorizer, search_matrix, qa_topk, qa_insurer, qa_sent, qa_topic, qa_notes)
                base_filter = pd.Series(True, index=df.index)
                if qa_insurer != "All":
                    base_filter &= df["assureur"] == qa_insurer
                if qa_sent != "All":
                    base_filter &= df["sentiment"] == qa_sent
                if qa_topic != "All":
                    base_filter &= df["lda_topic_label"] == qa_topic
                if qa_notes:
                    base_filter &= df["note"].isin(qa_notes)
                base_df = df[base_filter].copy()

                answer = answer_question(qa_q, base_df, retrieved)
                st.subheader("Answer")
                st.write(answer)

                if not retrieved.empty:
                    st.subheader("Retrieved evidence")
                    evidence = retrieved[["assureur", "sentiment", "note", "lda_topic_label", "similarity", "text_app"]].copy()
                    evidence["text_app"] = evidence["text_app"].apply(lambda x: short_text(x, 260))
                    st.dataframe(evidence, use_container_width=True)
        page_footer()


if __name__ == "__main__":
    main()
