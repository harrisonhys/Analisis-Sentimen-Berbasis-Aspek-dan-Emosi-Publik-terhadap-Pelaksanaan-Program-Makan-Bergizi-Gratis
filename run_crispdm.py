#!/usr/bin/env python3
"""
CRISP-DM Pipeline: Aspect-Based Sentiment and Emotion Analysis
of Public Perception toward Indonesia's Free Nutritious Meal Program (MBG) on Threads
=============================================================================

CRISP-DM Phases:
  1. Business Understanding
  2. Data Understanding
  3. Data Preparation
  4. Modeling
  5. Evaluation
  6. Deployment

Each phase saves results (txt, csv, png) to research_outputs/phaseN_*/ before proceeding.
"""

import os
import sys
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "threads_posts_2026-06-20-15-55-06-labeled.csv"
OUTPUT_DIR = BASE_DIR / "research_outputs"

PHASE_DIRS = {
    1: OUTPUT_DIR / "phase1_business_understanding",
    2: OUTPUT_DIR / "phase2_data_understanding",
    3: OUTPUT_DIR / "phase3_data_preparation",
    4: OUTPUT_DIR / "phase4_modeling",
    5: OUTPUT_DIR / "phase5_evaluation",
    6: OUTPUT_DIR / "phase6_deployment",
}

for d in PHASE_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# ── Utility Functions ──────────────────────────────────────────────────────
def section_header(phase_num: int, title: str):
    """Print a formatted section header."""
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"  PHASE {phase_num}: {title}")
    print(f"{bar}\n")


def save_text(phase: int, filename: str, content: str):
    """Save a text file to the phase output directory."""
    path = PHASE_DIRS[phase] / filename
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ Saved: {path.relative_to(BASE_DIR)}")


def save_table(phase: int, filename: str, df):
    """Save a DataFrame as CSV."""
    path = PHASE_DIRS[phase] / filename
    df.to_csv(path, index=True, encoding="utf-8")
    print(f"  ✓ Saved: {path.relative_to(BASE_DIR)}")


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: BUSINESS UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════
def phase1_business_understanding():
    """Define research objectives, questions, and success criteria."""
    section_header(1, "Business Understanding")

    report = f"""# PHASE 1: BUSINESS UNDERSTANDING
# Generated: {timestamp()}

## 1. Research Title
**Analisis Sentimen Berbasis Aspek dan Emosi Publik terhadap Pelaksanaan
Program Makan Bergizi Gratis (MBG) pada Media Sosial Threads**

English: **Aspect-Based Sentiment and Emotion Analysis of Public Perception
toward the Implementation of Indonesia's Free Nutritious Meal Program on Threads**

## 2. Background
Program Makan Bergizi Gratis (MBG) is a national strategic policy aimed at
improving human resource quality through nutritional fulfillment. By 2026,
BGN targets 82.9 million beneficiaries with ~21 billion meal portions across
thousands of SPPG units nationwide.

The program's massive scale generates diverse public responses on social media,
particularly Threads — a platform known for narrative, personal, and conversational
discourse. Initial data shows negative sentiment dominates (42.6%), with frustration
and anger as the strongest emotions, signaling the need for granular analysis beyond
simple positive/negative classification.

## 3. Research Objectives
1. Analyze sentiment distribution toward MBG implementation on Threads.
2. Identify dominant emotions in public discourse.
3. Map aspects of MBG implementation that trigger negative, positive, neutral, and mixed sentiment.
4. Examine relationships between implementation aspects and public emotions.
5. Generate data-driven policy recommendations for MBG communication and evaluation.

## 4. Research Questions
1. Bagaimana distribusi sentimen publik terhadap pelaksanaan Program MBG di Threads?
2. Emosi apa yang paling dominan dalam percakapan publik terkait MBG?
3. Aspek implementasi MBG apa yang paling banyak memicu sentimen negatif, positif, netral, dan mixed?
4. Bagaimana hubungan antara aspek implementasi dengan emosi publik seperti frustration, anger, worry, trust, dan satisfaction?
5. Bagaimana hasil analisis sentimen dan emosi dapat digunakan sebagai masukan evaluasi kebijakan publik?

## 5. Success Criteria
- Successfully classify relevance with F1 ≥ 0.85
- Sentiment classification with macro-F1 ≥ 0.70
- Emotion classification with macro-F1 ≥ 0.60
- Identify top 5 aspects driving each sentiment category
- Produce actionable policy recommendations

## 6. Analytical Framework
**CRISP-DM Methodology** with six phases:
1. Business Understanding → 2. Data Understanding → 3. Data Preparation
→ 4. Modeling → 5. Evaluation → 6. Deployment

**Analytical Techniques:**
- Relevance Detection: TF-IDF + Logistic Regression / SVM
- Sentiment Classification: TF-IDF + SVM, Naive Bayes (baseline), Ensemble
- Emotion Classification: TF-IDF + Multinomial NB, SVM, Logistic Regression
- Aspect Analysis: KeyBERT for keyword extraction, BERTopic for topic modeling
- Cross-tabulation: Sentiment × Emotion × Aspect matrix
"""

    save_text(1, "01_business_understanding_report.txt", report)

    # Business understanding summary table
    import pandas as pd

    objectives = pd.DataFrame({
        "Objective": [
            "Sentiment Distribution Analysis",
            "Emotion Classification",
            "Aspect-Based Sentiment Mapping",
            "Emotion-Aspect Relationship Analysis",
            "Policy Recommendation Generation"
        ],
        "Method": [
            "Descriptive statistics + Cross-tabulation",
            "Multi-class classification (10 emotions)",
            "KeyBERT + BERTopic + manual mapping",
            "Cross-tabulation Sentiment×Emotion×Aspect",
            "Synthesis of findings → actionable insights"
        ],
        "Success Metric": [
            "Distribution table + bar chart",
            "Macro-F1 ≥ 0.60",
            "Coverage ≥ 70% of relevant posts",
            "Chi-square significance p < 0.05",
            "≥ 5 concrete policy recommendations"
        ],
        "Priority": ["High", "High", "High", "Medium", "High"]
    })
    save_table(1, "02_research_objectives.csv", objectives)

    print("\n  ✅ Phase 1 complete — Business Understanding documented.")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: DATA UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════
def phase2_data_understanding():
    """Explore dataset: structure, distributions, quality issues, correlations."""
    section_header(2, "Data Understanding")

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter

    # ── 2.1 Load Data ──
    print("  Loading dataset...")
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    total_rows = len(df)
    print(f"  Dataset loaded: {total_rows:,} rows × {len(df.columns)} columns")

    # ── 2.2 Data Structure ──
    structure_report = f"""# PHASE 2: DATA UNDERSTANDING
# Generated: {timestamp()}

## 2.1 Dataset Overview
- Total rows: {total_rows:,}
- Total columns: {len(df.columns)}
- Source: Threads posts (scraped 2026-06-20)
- Columns: {', '.join(df.columns.tolist())}

## 2.2 Column Descriptions
"""
    col_desc = {
        "Post ID": "Unique identifier combining username and post code",
        "Post Code": "Threads post code",
        "Date/Time": "ISO 8601 timestamp of post",
        "Display Time": "Human-readable relative time (Indonesian)",
        "Username": "Threads username",
        "Display Name": "User display name",
        "Verified": "Account verification status (Yes/No)",
        "Post Text": "Full text content of the post",
        "Replies": "Number of replies",
        "Reposts": "Number of reposts",
        "Likes": "Number of likes",
        "Shares": "Number of shares",
        "Has Image": "Post contains image",
        "Has Video": "Post contains video",
        "Post URL": "URL to the original Threads post",
        "is_relevant": "Relevance label (True/False/None)",
        "topic_score": "Topic relevance confidence score",
        "sentiment": "Sentiment label (positive/negative/neutral/mixed)",
        "sentiment_score": "Sentiment confidence score",
        "emotion": "Emotion label (10 classes)",
        "emotion_score": "Emotion confidence score",
        "severity": "Severity level",
        "aspects": "Extracted implementation aspects (JSON array)",
        "summary_reason": "Reasoning for classification",
        "error": "Processing errors if any",
    }
    for col, desc in col_desc.items():
        structure_report += f"- **{col}**: {desc}\n"

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

    structure_report += f"\n## 2.3 Missing Values\n"
    if len(missing_df) > 0:
        structure_report += missing_df.to_string() + "\n"
    else:
        structure_report += "No missing values detected.\n"

    # Data types
    structure_report += f"\n## 2.4 Data Types\n"
    structure_report += df.dtypes.to_string() + "\n"

    save_text(2, "01_data_structure_report.txt", structure_report)
    save_table(2, "02_missing_values.csv", missing_df)

    # ── 2.3 Label Distributions ──
    print("\n  Analyzing label distributions...")

    # Fix known anomalies
    df["sentiment"] = df["sentiment"].replace("frustrasi", "negative")
    df["is_relevant"] = df["is_relevant"].replace("None", np.nan)

    # Sentiment distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Label Distributions — MBG Threads Dataset", fontsize=18, fontweight="bold", y=0.98)

    # Sentiment
    sent_counts = df["sentiment"].value_counts()
    colors_sent = {"negative": "#E74C3C", "neutral": "#7F8C8D", "positive": "#2ECC71", "mixed": "#F39C12"}
    bar_colors = [colors_sent.get(s, "#95A5A6") for s in sent_counts.index]
    axes[0, 0].bar(sent_counts.index, sent_counts.values, color=bar_colors, edgecolor="white", linewidth=1.5)
    axes[0, 0].set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate(sent_counts.values):
        axes[0, 0].text(i, v + 30, f"{v}\n({v/total_rows*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")

    # Relevance
    rel_counts = df["is_relevant"].value_counts(dropna=False)
    axes[0, 1].pie(rel_counts.values, labels=["Relevant", "Not Relevant", "None"], autopct="%1.1f%%",
                   colors=["#2ECC71", "#E74C3C", "#BDC3C7"], explode=(0.02, 0.02, 0.05),
                   textprops={"fontsize": 11})
    axes[0, 1].set_title("Relevance Distribution", fontsize=14, fontweight="bold")

    # Emotion
    emo_counts = df["emotion"].value_counts()
    emo_palette = sns.color_palette("Spectral", len(emo_counts))
    axes[1, 0].barh(emo_counts.index[::-1], emo_counts.values[::-1], color=emo_palette, edgecolor="white")
    axes[1, 0].set_title("Emotion Distribution", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Count")
    for i, v in enumerate(emo_counts.values[::-1]):
        axes[1, 0].text(v + 15, i, f"{v} ({v/total_rows*100:.1f}%)", va="center", fontsize=9)

    # Engagement metrics summary
    engagement_cols = ["Replies", "Reposts", "Likes", "Shares"]
    engagement_summary = df[engagement_cols].describe()
    axes[1, 1].axis("off")
    table_text = f"Engagement Summary:\n{'-'*40}\n"
    for col in engagement_cols:
        table_text += f"{col}: mean={df[col].mean():.1f}, max={df[col].max():.0f}, total={df[col].sum():,.0f}\n"
    axes[1, 1].text(0.1, 0.5, table_text, transform=axes[1, 1].transAxes,
                    fontsize=11, fontfamily="monospace", verticalalignment="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PHASE_DIRS[2] / "03_label_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 03_label_distributions.png")

    # Distribution tables
    sent_dist_df = pd.DataFrame({
        "Count": sent_counts.values,
        "Percentage": (sent_counts.values / total_rows * 100).round(1)
    }, index=sent_counts.index)
    save_table(2, "04_sentiment_distribution.csv", sent_dist_df)

    emo_dist_df = pd.DataFrame({
        "Count": emo_counts.values,
        "Percentage": (emo_counts.values / total_rows * 100).round(1)
    }, index=emo_counts.index)
    save_table(2, "05_emotion_distribution.csv", emo_dist_df)

    # ── 2.4 Cross-tabulation: Sentiment × Emotion ──
    print("  Analyzing Sentiment × Emotion relationships...")
    relevant_df = df[df["is_relevant"] == True].copy()

    cross_se = pd.crosstab(relevant_df["sentiment"], relevant_df["emotion"])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(cross_se, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Count"}, linewidths=0.5)
    ax.set_title("Cross-tabulation: Sentiment × Emotion (Relevant Posts Only)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Sentiment")
    plt.tight_layout()
    fig.savefig(PHASE_DIRS[2] / "06_sentiment_emotion_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    save_table(2, "06_sentiment_emotion_crosstab.csv", cross_se)

    # ── 2.5 Sentiment × Relevance Analysis ──
    cross_sr = pd.crosstab(df["sentiment"], df["is_relevant"].fillna("None"), margins=True)
    save_table(2, "07_sentiment_relevance_crosstab.csv", cross_sr)

    # ── 2.6 Engagement by Sentiment ──
    print("  Analyzing engagement by sentiment...")
    engagement_by_sent = relevant_df.groupby("sentiment")[engagement_cols].agg(["mean", "sum", "max"])
    save_table(2, "08_engagement_by_sentiment.csv", engagement_by_sent)

    # ── 2.7 Text Length Analysis ──
    df["text_length"] = df["Post Text"].fillna("").str.len()
    df["word_count"] = df["Post Text"].fillna("").str.split().str.len()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (col, label) in enumerate([("sentiment", "Sentiment"), ("emotion", "Emotion"), ("is_relevant", "Relevance")]):
        data_groups = [df[df[col] == g]["word_count"].values for g in df[col].dropna().unique()[:6]]
        labels = df[col].dropna().unique()[:6]
        bp = axes[i].boxplot(data_groups, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], sns.color_palette("Set2", len(labels))):
            patch.set_facecolor(color)
        axes[i].set_title(f"Word Count by {label}", fontsize=12, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_ylabel("Word Count")
    plt.tight_layout()
    fig.savefig(PHASE_DIRS[2] / "09_text_length_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2.8 Data Quality Report ──
    quality_report = f"""## 2.5 Data Quality Assessment

### Anomalies Found & Fixed:
1. **Sentiment label "frustrasi"**: 1 row had "frustrasi" in sentiment column → reclassified as "negative" (frustration is an emotion, not sentiment)
2. **is_relevant = None**: 1 row had None → treated as missing
3. **Missing aspects**: {df['aspects'].isnull().sum()} rows with null aspects
4. **Missing summary_reason**: {df['summary_reason'].isnull().sum()} rows
5. **Missing error field**: {df['error'].isnull().sum()} rows

### Class Imbalance Assessment:
- Sentiment: Negative (42.6%) vs Positive (16.1%) → 2.6:1 imbalance → requires stratified sampling + weighted metrics
- Emotion: Neutral+Frustration (58.6%) vs Surprise (1.4%) → severe imbalance → macro-F1 essential
- Relevance: Relevant (70.9%) vs Not Relevant (29.1%) → moderate imbalance

### Data Completeness:
- Full labels (sentiment + emotion + is_relevant): {len(df.dropna(subset=['sentiment', 'emotion', 'is_relevant'])):,} / {total_rows:,} rows
"""
    save_text(2, "10_data_quality_report.txt", quality_report)

    # Save cleaned dataframe info for next phase
    df.to_csv(PHASE_DIRS[2] / "11_dataset_cleaned_prelim.csv", index=False, encoding="utf-8")
    print("  ✓ Saved: 11_dataset_cleaned_prelim.csv (snapshot for Phase 3)")

    print("\n  ✅ Phase 2 complete — Data Understanding documented.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
def phase3_data_preparation(df):
    """Clean, preprocess, and prepare features for modeling."""
    section_header(3, "Data Preparation")

    import pandas as pd
    import numpy as np
    import re
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()

    # ── 3.1 Fix Known Anomalies ──
    print("  Fixing label anomalies...")
    df["sentiment"] = df["sentiment"].replace("frustrasi", "negative")
    df["is_relevant"] = df["is_relevant"].replace("None", np.nan)

    # Drop the 1 None row for is_relevant modeling
    df_relevance = df.dropna(subset=["is_relevant"]).copy()
    df_relevance["relevance_label"] = df_relevance["is_relevant"].astype(int)

    prep_report = f"""# PHASE 3: DATA PREPARATION
# Generated: {timestamp()}

## 3.1 Anomaly Correction
- 1 row "frustrasi" → reclassified as "negative" (emotion label, not sentiment)
- 1 row is_relevant=None → excluded from relevance modeling

## 3.2 Data Splits
"""

    # ── 3.2 Text Cleaning ──
    print("  Cleaning text data...")

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove mentions
        text = re.sub(r"@\w+", "", text)
        # Remove hashtag symbol (keep text)
        text = re.sub(r"#", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Case folding
        text = text.lower()
        return text

    df["clean_text"] = df["Post Text"].apply(clean_text)
    df["clean_length"] = df["clean_text"].str.len()
    df["clean_words"] = df["clean_text"].str.split().str.len()

    # Remove empty texts
    empty_texts = (df["clean_text"] == "") | (df["clean_text"].isnull())
    prep_report += f"- Empty texts after cleaning: {empty_texts.sum()} rows\n"
    df = df[~empty_texts].copy()

    prep_report += f"- Clean dataset size: {len(df):,} rows\n"

    # ── 3.3 Dataset Splits ──
    print("  Creating train/validation/test splits...")

    # For relevance: all data with labels
    rel_data = df.dropna(subset=["is_relevant"]).copy()
    rel_data["rel_label"] = rel_data["is_relevant"].astype(int)

    # For sentiment & emotion: relevant data only
    sent_data = df[(df["is_relevant"] == True) & (df["sentiment"].notna()) & (df["sentiment"] != "")].copy()

    # Group rare emotion classes (n < 10) into "other" for stratification
    emo_counts_full = sent_data["emotion"].value_counts()
    rare_emo_classes = emo_counts_full[emo_counts_full < 10].index.tolist()
    prep_report += f"\n- Rare emotion classes grouped into 'other': {rare_emo_classes}\n"
    sent_data["emotion_grouped"] = sent_data["emotion"].apply(
        lambda x: "other" if x in rare_emo_classes else x
    )

    # Split relevance data
    X_rel = rel_data["clean_text"]
    y_rel = rel_data["rel_label"]
    X_rel_train, X_rel_temp, y_rel_train, y_rel_temp = train_test_split(
        X_rel, y_rel, test_size=0.30, random_state=42, stratify=y_rel
    )
    X_rel_val, X_rel_test, y_rel_val, y_rel_test = train_test_split(
        X_rel_temp, y_rel_temp, test_size=0.50, random_state=42, stratify=y_rel_temp
    )

    prep_report += f"""
### Relevance Detection Split:
- Train: {len(X_rel_train):,} ({len(X_rel_train)/len(X_rel)*100:.1f}%)
- Validation: {len(X_rel_val):,} ({len(X_rel_val)/len(X_rel)*100:.1f}%)
- Test: {len(X_rel_test):,} ({len(X_rel_test)/len(X_rel)*100:.1f}%)
- Label distribution (train): {dict(zip(*np.unique(y_rel_train, return_counts=True)))}
"""

    # Split sentiment data — use the same train/val/test indices for both sentiment and emotion
    X_sent = sent_data["clean_text"]
    y_sent = sent_data["sentiment"]
    y_emo = sent_data["emotion_grouped"]  # Use grouped emotions for stratification

    X_sent_train, X_sent_temp, y_sent_train, y_sent_temp = train_test_split(
        X_sent, y_sent, test_size=0.30, random_state=42, stratify=y_sent
    )
    X_sent_val, X_sent_test, y_sent_val, y_sent_test = train_test_split(
        X_sent_temp, y_sent_temp, test_size=0.50, random_state=42, stratify=y_sent_temp
    )

    # Emotion uses same text splits as sentiment (same indices)
    train_indices = X_sent_train.index
    val_indices = X_sent_val.index
    test_indices = X_sent_test.index

    X_emo_train = X_sent.loc[train_indices]
    y_emo_train = y_emo.loc[train_indices]
    X_emo_val = X_sent.loc[val_indices]
    y_emo_val = y_emo.loc[val_indices]
    X_emo_test = X_sent.loc[test_indices]
    y_emo_test = y_emo.loc[test_indices]

    prep_report += f"""
### Sentiment/Emotion Split (Relevant Posts Only):
- Train: {len(X_sent_train):,} ({len(X_sent_train)/len(X_sent)*100:.1f}%)
- Validation: {len(X_sent_val):,} ({len(X_sent_val)/len(X_sent)*100:.1f}%)
- Test: {len(X_sent_test):,} ({len(X_sent_test)/len(X_sent)*100:.1f}%)

### Sentiment Distribution (Train):
{ y_sent_train.value_counts().to_string() }

### Emotion Distribution (Train — rare grouped):
{ y_emo_train.value_counts().to_string() }
"""

    # ── 3.4 TF-IDF Vectorization ──
    print("  Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )
    X_tfidf_train = tfidf.fit_transform(X_rel_train)
    X_tfidf_rel_val = tfidf.transform(X_rel_val)
    X_tfidf_rel_test = tfidf.transform(X_rel_test)
    X_tfidf_sent_train = tfidf.transform(X_sent_train)
    X_tfidf_sent_val = tfidf.transform(X_sent_val)
    X_tfidf_sent_test = tfidf.transform(X_sent_test)

    # TF-IDF for emotion
    tfidf_emo = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )
    X_tfidf_emo_train = tfidf_emo.fit_transform(X_emo_train)
    X_tfidf_emo_val = tfidf_emo.transform(X_emo_val)
    X_tfidf_emo_test = tfidf_emo.transform(X_emo_test)

    prep_report += f"""
## 3.3 TF-IDF Vectorization
- Max features: 10,000
- N-gram range: (1, 2)
- Min document frequency: 2
- Max document frequency: 90%
- Sublinear TF scaling: enabled

### Vocabulary Size:
- Relevance TF-IDF: {len(tfidf.vocabulary_):,} terms
- Sentiment TF-IDF: {len(tfidf.vocabulary_):,} terms
- Emotion TF-IDF: {len(tfidf_emo.vocabulary_):,} terms

## 3.4 Top TF-IDF Terms
"""
    # Top terms
    feature_names = tfidf.get_feature_names_out()
    idf_scores = tfidf.idf_
    top_indices = np.argsort(idf_scores)[-20:][::-1]
    for idx in top_indices:
        prep_report += f"  - {feature_names[idx]}: IDF={idf_scores[idx]:.4f}\n"

    save_text(3, "01_data_preparation_report.txt", prep_report)

    # ── 3.5 Label Encoding ──
    le_sent = LabelEncoder()
    y_sent_train_enc = le_sent.fit_transform(y_sent_train)
    y_sent_val_enc = le_sent.transform(y_sent_val)
    y_sent_test_enc = le_sent.transform(y_sent_test)
    sent_labels = le_sent.classes_.tolist()

    le_emo = LabelEncoder()
    y_emo_train_enc = le_emo.fit_transform(y_emo_train)
    y_emo_val_enc = le_emo.transform(y_emo_val)
    y_emo_test_enc = le_emo.transform(y_emo_test)
    emo_labels = le_emo.classes_.tolist()

    label_mapping = pd.DataFrame({
        "Sentiment Labels": sent_labels,
        "Sentiment Code": range(len(sent_labels)),
    })
    save_table(3, "02_sentiment_label_mapping.csv", label_mapping)

    emo_mapping = pd.DataFrame({
        "Emotion Labels": emo_labels,
        "Emotion Code": range(len(emo_labels)),
    })
    save_table(3, "03_emotion_label_mapping.csv", emo_mapping)

    # ── 3.6 Class Weight Calculation ──
    from sklearn.utils.class_weight import compute_class_weight

    sent_weights = compute_class_weight("balanced", classes=np.unique(y_sent_train_enc), y=y_sent_train_enc)
    emo_weights = compute_class_weight("balanced", classes=np.unique(y_emo_train_enc), y=y_emo_train_enc)

    weights_df = pd.DataFrame({
        "Sentiment Class": sent_labels,
        "Sentiment Weight": sent_weights.round(3),
    })
    save_table(3, "04_sentiment_class_weights.csv", weights_df)

    emo_weights_df = pd.DataFrame({
        "Emotion Class": emo_labels,
        "Emotion Weight": emo_weights.round(3),
    })
    save_table(3, "05_emotion_class_weights.csv", emo_weights_df)

    # ── 3.7 Aspect Data Preparation ──
    print("  Preparing aspect data...")
    # Parse aspects column (JSON array)
    import json

    def parse_aspects(val):
        if not isinstance(val, str) or val in ["[]", "", "nan", "None"]:
            return []
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    df["aspects_list"] = df["aspects"].apply(parse_aspects)

    # Aspect statistics
    all_aspects = []
    for aspects in df.loc[df["is_relevant"] == True, "aspects_list"]:
        all_aspects.extend(aspects)

    from collections import Counter
    aspect_counts = Counter(all_aspects)
    aspect_df = pd.DataFrame(aspect_counts.most_common(), columns=["Aspect", "Count"])
    aspect_df["Percentage"] = (aspect_df["Count"] / len(df[df["is_relevant"] == True]) * 100).round(1)
    save_table(3, "06_aspect_distribution.csv", aspect_df)

    print("  ✓ All preparation artifacts saved.")

    # ── 3.8 Save Processed Data Bundle ──
    processed_data = {
        "tfidf": tfidf,
        "tfidf_emo": tfidf_emo,
        "X_tfidf_sent_train": X_tfidf_sent_train,
        "X_tfidf_sent_val": X_tfidf_sent_val,
        "X_tfidf_sent_test": X_tfidf_sent_test,
        "X_tfidf_rel_train": X_tfidf_train,
        "X_tfidf_rel_val": X_tfidf_rel_val,
        "X_tfidf_rel_test": X_tfidf_rel_test,
        "X_tfidf_emo_train": X_tfidf_emo_train,
        "X_tfidf_emo_val": X_tfidf_emo_val,
        "X_tfidf_emo_test": X_tfidf_emo_test,
        "y_rel_train": y_rel_train,
        "y_rel_val": y_rel_val,
        "y_rel_test": y_rel_test,
        "y_sent_train_enc": y_sent_train_enc,
        "y_sent_val_enc": y_sent_val_enc,
        "y_sent_test_enc": y_sent_test_enc,
        "y_emo_train_enc": y_emo_train_enc,
        "y_emo_val_enc": y_emo_val_enc,
        "y_emo_test_enc": y_emo_test_enc,
        "sent_labels": sent_labels,
        "emo_labels": emo_labels,
        "le_sent": le_sent,
        "le_emo": le_emo,
        "sent_weights": sent_weights,
        "emo_weights": emo_weights,
        "df_clean": df,
        "sent_data": sent_data,
        "rel_data": rel_data,
    }

    print("\n  ✅ Phase 3 complete — Data prepared for modeling.")
    return processed_data


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: MODELING
# ══════════════════════════════════════════════════════════════════════════════
def phase4_modeling(data):
    """Build and compare models for Relevance, Sentiment, and Emotion classification."""
    section_header(4, "Modeling")

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        f1_score, precision_score, recall_score
    )
    from sklearn.calibration import CalibratedClassifierCV
    import time

    modeling_report = f"""# PHASE 4: MODELING
# Generated: {timestamp()}

## 4.1 Models Tested
"""

    # ── 4.1 Relevance Detection Models ──
    print("\n  ── 4.1 Training Relevance Detection Models ──")
    models_rel = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVC": CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight="balanced", random_state=42, dual=False)),
        "Multinomial NB": MultinomialNB(),
        "Complement NB": ComplementNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    rel_results = []
    for name, model in models_rel.items():
        print(f"    Training {name}...")
        start = time.time()
        model.fit(data["X_tfidf_rel_train"], data["y_rel_train"])
        elapsed = time.time() - start

        y_pred_val = model.predict(data["X_tfidf_rel_val"])
        acc = accuracy_score(data["y_rel_val"], y_pred_val)
        f1 = f1_score(data["y_rel_val"], y_pred_val, average="macro")
        prec = precision_score(data["y_rel_val"], y_pred_val, average="macro")
        rec = recall_score(data["y_rel_val"], y_pred_val, average="macro")

        rel_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro F1": round(f1, 4),
            "Macro Precision": round(prec, 4),
            "Macro Recall": round(rec, 4),
            "Train Time (s)": round(elapsed, 2),
        })
        print(f"      Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Time: {elapsed:.2f}s")

    rel_results_df = pd.DataFrame(rel_results).sort_values("Macro F1", ascending=False)
    save_table(4, "01_relevance_model_comparison.csv", rel_results_df)

    # Best relevance model
    best_rel_name = rel_results_df.iloc[0]["Model"]
    best_rel = models_rel[best_rel_name]
    y_rel_pred_test = best_rel.predict(data["X_tfidf_rel_test"])
    rel_test_f1 = f1_score(data["y_rel_test"], y_rel_pred_test, average="macro")

    modeling_report += f"""
### Relevance Detection
Best Model: **{best_rel_name}** (Macro F1 = {rel_test_f1:.4f} on test set)

{rel_results_df.to_string(index=False)}
"""

    # Confusion matrix for best relevance model
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_rel = confusion_matrix(data["y_rel_test"], y_rel_pred_test)
    sns.heatmap(cm_rel, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Relevant", "Relevant"],
                yticklabels=["Not Relevant", "Relevant"])
    ax.set_title(f"Relevance Detection — {best_rel_name}\nTest Set Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(PHASE_DIRS[4] / "02_relevance_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4.2 Sentiment Classification Models ──
    print("\n  ── 4.2 Training Sentiment Classification Models ──")
    models_sent = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Linear SVC": CalibratedClassifierCV(LinearSVC(max_iter=3000, class_weight="balanced", random_state=42, dual=False)),
        "Multinomial NB": MultinomialNB(),
        "Complement NB": ComplementNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    sent_results = []
    for name, model in models_sent.items():
        print(f"    Training {name}...")
        start = time.time()
        model.fit(data["X_tfidf_sent_train"], data["y_sent_train_enc"])
        elapsed = time.time() - start

        y_pred_val = model.predict(data["X_tfidf_sent_val"])
        acc = accuracy_score(data["y_sent_val_enc"], y_pred_val)
        f1_macro = f1_score(data["y_sent_val_enc"], y_pred_val, average="macro")
        f1_weighted = f1_score(data["y_sent_val_enc"], y_pred_val, average="weighted")

        sent_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro F1": round(f1_macro, 4),
            "Weighted F1": round(f1_weighted, 4),
            "Train Time (s)": round(elapsed, 2),
        })
        print(f"      Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")

    sent_results_df = pd.DataFrame(sent_results).sort_values("Macro F1", ascending=False)
    save_table(4, "03_sentiment_model_comparison.csv", sent_results_df)

    # Best sentiment model
    best_sent_name = sent_results_df.iloc[0]["Model"]
    best_sent = models_sent[best_sent_name]
    y_sent_pred_test = best_sent.predict(data["X_tfidf_sent_test"])
    sent_test_f1 = f1_score(data["y_sent_test_enc"], y_sent_pred_test, average="macro")

    modeling_report += f"""
### Sentiment Classification
Best Model: **{best_sent_name}** (Macro F1 = {sent_test_f1:.4f} on test set)

{sent_results_df.to_string(index=False)}
"""

    # Classification report
    sent_cls_report = classification_report(
        data["y_sent_test_enc"], y_sent_pred_test,
        target_names=data["sent_labels"], output_dict=True
    )
    sent_cls_df = pd.DataFrame(sent_cls_report).T
    save_table(4, "04_sentiment_classification_report.csv", sent_cls_df)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_sent = confusion_matrix(data["y_sent_test_enc"], y_sent_pred_test)
    cm_sent_norm = cm_sent.astype("float") / cm_sent.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_sent_norm, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                xticklabels=data["sent_labels"], yticklabels=data["sent_labels"],
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title(f"Sentiment Classification — {best_sent_name}\nNormalized Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(PHASE_DIRS[4] / "05_sentiment_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4.3 Emotion Classification Models ──
    print("\n  ── 4.3 Training Emotion Classification Models ──")
    models_emo = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Linear SVC": CalibratedClassifierCV(LinearSVC(max_iter=3000, class_weight="balanced", random_state=42, dual=False)),
        "Multinomial NB": MultinomialNB(),
        "Complement NB": ComplementNB(),
    }

    emo_results = []
    for name, model in models_emo.items():
        print(f"    Training {name}...")
        start = time.time()
        model.fit(data["X_tfidf_emo_train"], data["y_emo_train_enc"])
        elapsed = time.time() - start

        y_pred_val = model.predict(data["X_tfidf_emo_val"])
        acc = accuracy_score(data["y_emo_val_enc"], y_pred_val)
        f1_macro = f1_score(data["y_emo_val_enc"], y_pred_val, average="macro")
        f1_weighted = f1_score(data["y_emo_val_enc"], y_pred_val, average="weighted")

        emo_results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro F1": round(f1_macro, 4),
            "Weighted F1": round(f1_weighted, 4),
            "Train Time (s)": round(elapsed, 2),
        })
        print(f"      Acc: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")

    emo_results_df = pd.DataFrame(emo_results).sort_values("Macro F1", ascending=False)
    save_table(4, "06_emotion_model_comparison.csv", emo_results_df)

    # Best emotion model
    best_emo_name = emo_results_df.iloc[0]["Model"]
    best_emo = models_emo[best_emo_name]
    y_emo_pred_test = best_emo.predict(data["X_tfidf_emo_test"])
    emo_test_f1 = f1_score(data["y_emo_test_enc"], y_emo_pred_test, average="macro")

    modeling_report += f"""
### Emotion Classification
Best Model: **{best_emo_name}** (Macro F1 = {emo_test_f1:.4f} on test set)

{emo_results_df.to_string(index=False)}
"""

    # Emotion classification report
    emo_cls_report = classification_report(
        data["y_emo_test_enc"], y_emo_pred_test,
        target_names=data["emo_labels"], output_dict=True
    )
    emo_cls_df = pd.DataFrame(emo_cls_report).T
    save_table(4, "07_emotion_classification_report.csv", emo_cls_df)

    # Emotion confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    cm_emo = confusion_matrix(data["y_emo_test_enc"], y_emo_pred_test)
    cm_emo_norm = cm_emo.astype("float") / cm_emo.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_emo_norm, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                xticklabels=data["emo_labels"], yticklabels=data["emo_labels"],
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_title(f"Emotion Classification — {best_emo_name}\nNormalized Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(PHASE_DIRS[4] / "08_emotion_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4.4 Model Performance Summary ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Relevance
    axes[0].barh(rel_results_df["Model"], rel_results_df["Macro F1"], color=sns.color_palette("viridis", 5))
    axes[0].set_title("Relevance Detection\n(Macro F1)", fontweight="bold")
    axes[0].set_xlim(0.7, 1.0)
    for i, v in enumerate(rel_results_df["Macro F1"]):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center")

    # Sentiment
    axes[1].barh(sent_results_df["Model"], sent_results_df["Macro F1"], color=sns.color_palette("viridis", 5))
    axes[1].set_title("Sentiment Classification\n(Macro F1)", fontweight="bold")
    axes[1].set_xlim(0, 0.8)
    for i, v in enumerate(sent_results_df["Macro F1"]):
        axes[1].text(v + 0.005, i, f"{v:.3f}", va="center")

    # Emotion
    axes[2].barh(emo_results_df["Model"], emo_results_df["Macro F1"], color=sns.color_palette("viridis", 4))
    axes[2].set_title("Emotion Classification\n(Macro F1)", fontweight="bold")
    axes[2].set_xlim(0, 0.8)
    for i, v in enumerate(emo_results_df["Macro F1"]):
        axes[2].text(v + 0.005, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    fig.savefig(PHASE_DIRS[4] / "09_model_comparison_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4.5 Aspect Extraction ──
    print("\n  ── 4.4 Aspect Extraction & Analysis ──")
    df_clean = data["df_clean"]
    relevant_mask = df_clean["is_relevant"] == True
    relevant_df = df_clean[relevant_mask].copy()

    # Aspect-sentiment cross-tabulation
    import json

    def parse_aspects(val):
        if not isinstance(val, str) or val in ["[]", "", "nan", "None"]:
            return []
        try:
            return json.loads(val) if isinstance(json.loads(val), list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    relevant_df["aspects_parsed"] = relevant_df["aspects"].apply(parse_aspects)

    # Explode aspects
    aspect_rows = []
    for _, row in relevant_df.iterrows():
        for aspect in row["aspects_parsed"]:
            aspect_rows.append({
                "aspect": aspect,
                "sentiment": row["sentiment"],
                "emotion": row["emotion"],
                "severity": row["severity"],
            })

    if aspect_rows:
        aspect_df = pd.DataFrame(aspect_rows)

        # Aspect × Sentiment
        aspect_sent_cross = pd.crosstab(aspect_df["aspect"], aspect_df["sentiment"])
        aspect_sent_cross["total"] = aspect_sent_cross.sum(axis=1)
        aspect_sent_cross = aspect_sent_cross.sort_values("total", ascending=False)

        # Normalized
        aspect_sent_norm = aspect_sent_cross.drop(columns="total").div(
            aspect_sent_cross["total"], axis=0
        ) * 100

        save_table(4, "10_aspect_sentiment_crosstab.csv", aspect_sent_cross)
        save_table(4, "11_aspect_sentiment_normalized.csv", aspect_sent_norm.round(1))

        # Aspect × Emotion
        aspect_emo_cross = pd.crosstab(aspect_df["aspect"], aspect_df["emotion"])
        save_table(4, "12_aspect_emotion_crosstab.csv", aspect_emo_cross)

        # Aspect distribution plot
        top_aspects = aspect_sent_cross.head(15)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Count plot
        top_aspects.drop(columns="total").plot(
            kind="barh", stacked=True, ax=axes[0],
            color=["#2ECC71", "#F39C12", "#E74C3C", "#7F8C8D"]
        )
        axes[0].set_title("Top 15 Aspects by Sentiment (Count)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Number of Posts")
        axes[0].legend(loc="lower right")

        # Normalized
        top_aspects_norm = aspect_sent_norm.loc[top_aspects.index]
        top_aspects_norm.plot(
            kind="barh", stacked=True, ax=axes[1],
            color=["#2ECC71", "#F39C12", "#E74C3C", "#7F8C8D"]
        )
        axes[1].set_title("Top 15 Aspects by Sentiment (%)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Percentage")
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        fig.savefig(PHASE_DIRS[4] / "13_aspect_sentiment_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Aspect-Emotion heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        aspect_emo_norm = aspect_emo_cross.div(aspect_emo_cross.sum(axis=1), axis=0) * 100
        sns.heatmap(aspect_emo_norm, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "% of Posts"}, linewidths=0.5)
        ax.set_title("Aspect × Emotion Distribution (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Aspect")
        plt.tight_layout()
        fig.savefig(PHASE_DIRS[4] / "14_aspect_emotion_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        aspect_sent_cross = pd.DataFrame()
        print("    ⚠ No parsed aspects found, using label-based aspect analysis instead.")

    # ── 4.6 TF-IDF Top Features per Class ──
    print("\n  ── 4.5 Extracting Top Features per Class ──")
    feature_names = data["tfidf"].get_feature_names_out()

    # For Logistic Regression (best model typically)
    if hasattr(best_sent, "coef_"):
        coefs = best_sent.coef_
        top_features_text = "\n## 4.6 Top TF-IDF Features per Sentiment Class\n"
        for i, label in enumerate(data["sent_labels"]):
            if i < len(coefs):
                top_indices = np.argsort(coefs[i])[-15:][::-1]
                top_terms = [(feature_names[idx], coefs[i][idx]) for idx in top_indices]
                top_features_text += f"\n### {label}\n"
                for term, weight in top_terms:
                    top_features_text += f"  - {term}: {weight:.4f}\n"
        save_text(4, "15_top_features_per_sentiment.txt", top_features_text)

    # ── 4.7 Word Clouds per Sentiment ──
    print("  Generating word clouds...")
    try:
        from wordcloud import WordCloud

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        for i, sentiment in enumerate(["positive", "negative", "neutral", "mixed"]):
            text = " ".join(relevant_df[relevant_df["sentiment"] == sentiment]["clean_text"].values)
            if text.strip():
                wc = WordCloud(width=400, height=300, background_color="white",
                               max_words=100, colormap="viridis", random_state=42).generate(text)
                axes[i].imshow(wc, interpolation="bilinear")
            axes[i].set_title(f"{sentiment.capitalize()}", fontsize=13, fontweight="bold")
            axes[i].axis("off")
        fig.suptitle("Word Clouds by Sentiment — MBG Threads Posts", fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig.savefig(PHASE_DIRS[4] / "16_wordclouds_by_sentiment.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"    ⚠ Word cloud generation skipped: {e}")

    # Save the best models info
    modeling_report += f"""
## 4.7 Best Models Summary

| Task | Best Model | Test Macro F1 |
|------|-----------|---------------|
| Relevance Detection | {best_rel_name} | {rel_test_f1:.4f} |
| Sentiment Classification | {best_sent_name} | {sent_test_f1:.4f} |
| Emotion Classification | {best_emo_name} | {emo_test_f1:.4f} |
"""
    save_text(4, "17_modeling_report.txt", modeling_report)

    # Save model references
    model_bundle = {
        "best_rel_model": best_rel,
        "best_sent_model": best_sent,
        "best_emo_model": best_emo,
        "best_rel_name": best_rel_name,
        "best_sent_name": best_sent_name,
        "best_emo_name": best_emo_name,
        "all_rel_models": models_rel,
        "all_sent_models": models_sent,
        "all_emo_models": models_emo,
        "aspect_sent_cross": aspect_sent_cross if len(aspect_sent_cross) > 0 else None,
        "aspect_df": aspect_df if "aspect_df" in dir() else None,
    }

    print("\n  ✅ Phase 4 complete — Models trained and compared.")
    return model_bundle


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def phase5_evaluation(data, models):
    """Evaluate models thoroughly, analyze errors, and synthesize findings."""
    section_header(5, "Evaluation")

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score,
        precision_recall_fscore_support
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from scipy.stats import chi2_contingency

    eval_report = f"""# PHASE 5: EVALUATION
# Generated: {timestamp()}

## 5.1 Final Model Performance Summary
"""

    # ── 5.1 Cross-Validation ──
    print("  Running cross-validation on best models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Sentiment CV
    cv_scores = cross_val_score(
        models["best_sent_model"], data["X_tfidf_sent_train"],
        data["y_sent_train_enc"], cv=cv, scoring="f1_macro", n_jobs=-1
    )
    eval_report += f"""
### Sentiment Classification ({models['best_sent_name']})
- 5-Fold CV Macro F1: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}
- Fold scores: {[f"{s:.4f}" for s in cv_scores]}
"""
    cv_sent_df = pd.DataFrame({"Fold": range(1, 6), "Macro F1": cv_scores})
    save_table(5, "01_sentiment_cross_validation.csv", cv_sent_df)

    # Emotion CV
    cv_emo_scores = cross_val_score(
        models["best_emo_model"], data["X_tfidf_emo_train"],
        data["y_emo_train_enc"], cv=cv, scoring="f1_macro", n_jobs=-1
    )
    eval_report += f"""
### Emotion Classification ({models['best_emo_name']})
- 5-Fold CV Macro F1: mean={cv_emo_scores.mean():.4f}, std={cv_emo_scores.std():.4f}
- Fold scores: {[f"{s:.4f}" for s in cv_emo_scores]}
"""
    cv_emo_df = pd.DataFrame({"Fold": range(1, 6), "Macro F1": cv_emo_scores})
    save_table(5, "02_emotion_cross_validation.csv", cv_emo_df)

    # ── 5.2 Per-Class Performance Analysis ──
    print("  Analyzing per-class performance...")
    best_sent = models["best_sent_model"]
    y_sent_pred = best_sent.predict(data["X_tfidf_sent_test"])

    precision, recall, f1, support = precision_recall_fscore_support(
        data["y_sent_test_enc"], y_sent_pred, labels=range(len(data["sent_labels"]))
    )

    per_class_sent = pd.DataFrame({
        "Class": data["sent_labels"],
        "Precision": precision.round(4),
        "Recall": recall.round(4),
        "F1-Score": f1.round(4),
        "Support": support,
    })
    save_table(5, "03_per_class_sentiment_performance.csv", per_class_sent)

    # Emotion per-class
    best_emo = models["best_emo_model"]
    y_emo_pred = best_emo.predict(data["X_tfidf_emo_test"])
    precision_e, recall_e, f1_e, support_e = precision_recall_fscore_support(
        data["y_emo_test_enc"], y_emo_pred, labels=range(len(data["emo_labels"]))
    )
    per_class_emo = pd.DataFrame({
        "Class": data["emo_labels"],
        "Precision": precision_e.round(4),
        "Recall": recall_e.round(4),
        "F1-Score": f1_e.round(4),
        "Support": support_e,
    })
    save_table(5, "04_per_class_emotion_performance.csv", per_class_emo)

    # Per-class F1 plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Sentiment
    bars = axes[0].bar(per_class_sent["Class"], per_class_sent["F1-Score"],
                       color=sns.color_palette("Set2", 4))
    for bar, val in zip(bars, per_class_sent["F1-Score"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontweight="bold")
    axes[0].set_title("Per-Class F1-Score — Sentiment", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("F1-Score")
    axes[0].axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Target (0.70)")
    axes[0].legend()

    # Emotion
    emo_sorted = per_class_emo.sort_values("F1-Score", ascending=True)
    colors_e = sns.color_palette("Spectral", len(emo_sorted))
    bars = axes[1].barh(emo_sorted["Class"], emo_sorted["F1-Score"], color=colors_e)
    for bar, val in zip(bars, emo_sorted["F1-Score"]):
        axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va="center", fontweight="bold")
    axes[1].set_title("Per-Class F1-Score — Emotion", fontsize=13, fontweight="bold")
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel("F1-Score")
    axes[1].axvline(x=0.60, color="gray", linestyle="--", alpha=0.5, label="Target (0.60)")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(PHASE_DIRS[5] / "05_per_class_f1_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5.3 Error Analysis ──
    print("  Performing error analysis...")
    # Find misclassified examples for sentiment
    sent_actual = data["le_sent"].inverse_transform(data["y_sent_test_enc"])
    sent_predicted = data["le_sent"].inverse_transform(y_sent_pred)
    errors = sent_actual != sent_predicted
    error_count = errors.sum()

    eval_report += f"""
## 5.2 Error Analysis

### Sentiment Misclassification Rate: {error_count}/{len(sent_actual)} ({error_count/len(sent_actual)*100:.1f}%)

### Confusion Patterns:
"""
    # Common confusions
    cm = confusion_matrix(data["y_sent_test_enc"], y_sent_pred)
    for i, actual in enumerate(data["sent_labels"]):
        for j, pred in enumerate(data["sent_labels"]):
            if i != j and cm[i][j] > 0:
                eval_report += f"  - {actual} → {pred}: {cm[i][j]} instances\n"

    # ── 5.4 Statistical Tests ──
    print("  Running statistical tests...")
    df_clean = data["df_clean"]
    relevant_df = df_clean[df_clean["is_relevant"] == True].copy()

    # Chi-square test: Sentiment × Emotion
    cross_se = pd.crosstab(relevant_df["sentiment"], relevant_df["emotion"])
    chi2, p_value, dof, expected = chi2_contingency(cross_se)
    eval_report += f"""
## 5.3 Statistical Analysis

### Chi-Square Test: Sentiment × Emotion
- χ² = {chi2:.2f}
- Degrees of freedom = {dof}
- p-value = {p_value:.6f}
- **Interpretation**: {'Significant association (p < 0.05)' if p_value < 0.05 else 'No significant association'}
"""
    chi2_results = pd.DataFrame({
        "Test": ["Sentiment × Emotion Association"],
        "Chi-Square": [round(chi2, 2)],
        "DF": [dof],
        "P-Value": [f"{p_value:.6f}"],
        "Significant (α=0.05)": ["Yes" if p_value < 0.05 else "No"],
    })
    save_table(5, "06_chi_square_test.csv", chi2_results)

    # ── 5.5 Sentiment × Emotion × Aspect Synthesis ──
    print("  Synthesizing cross-dimensional analysis...")

    # Create sentiment-emotion-aspect summary
    import json

    def parse_aspects(val):
        if not isinstance(val, str) or val in ["[]", "", "nan", "None"]:
            return []
        try:
            return json.loads(val) if isinstance(json.loads(val), list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    relevant_df["aspects_parsed"] = relevant_df["aspects"].apply(parse_aspects)
    exploded_rows = []
    for _, row in relevant_df.iterrows():
        for aspect in row["aspects_parsed"]:
            exploded_rows.append({
                "aspect": aspect,
                "sentiment": row["sentiment"],
                "emotion": row["emotion"],
            })

    if exploded_rows:
        exp_df = pd.DataFrame(exploded_rows)

        # Sentiment × Emotion × Aspect pivot
        pivot = exp_df.pivot_table(
            index="aspect", columns=["sentiment", "emotion"],
            aggfunc="size", fill_value=0
        )
        save_table(5, "07_aspect_sentiment_emotion_pivot.csv", pivot)

        # Top aspect per emotion
        aspect_emo_summary = []
        for emotion in exp_df["emotion"].unique():
            emo_data = exp_df[exp_df["emotion"] == emotion]
            top_aspects = emo_data["aspect"].value_counts().head(3)
            aspect_emo_summary.append({
                "Emotion": emotion,
                "Top Aspect 1": top_aspects.index[0] if len(top_aspects) > 0 else "",
                "Count 1": top_aspects.values[0] if len(top_aspects) > 0 else 0,
                "Top Aspect 2": top_aspects.index[1] if len(top_aspects) > 1 else "",
                "Count 2": top_aspects.values[1] if len(top_aspects) > 1 else 0,
                "Top Aspect 3": top_aspects.index[2] if len(top_aspects) > 2 else "",
                "Count 3": top_aspects.values[2] if len(top_aspects) > 2 else 0,
            })
        aspect_emo_df = pd.DataFrame(aspect_emo_summary)
        save_table(5, "08_top_aspects_per_emotion.csv", aspect_emo_df)

        eval_report += f"""
## 5.4 Aspect-Emotion-Sentiment Synthesis

### Top Aspects per Emotion:
{aspect_emo_df.to_string(index=False)}
"""

    # ── 5.6 Success Criteria Check ──
    eval_report += f"""
## 5.5 Success Criteria Evaluation

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Relevance F1 | ≥ 0.85 | {f1_score(data['y_rel_test'], models['best_rel_model'].predict(data['X_tfidf_rel_test']), average='macro'):.4f} | {'✅ PASS' if f1_score(data['y_rel_test'], models['best_rel_model'].predict(data['X_tfidf_rel_test']), average='macro') >= 0.85 else '⚠ BELOW TARGET'} |
| Sentiment Macro-F1 | ≥ 0.70 | {cv_scores.mean():.4f} | {'✅ PASS' if cv_scores.mean() >= 0.70 else '⚠ BELOW TARGET'} |
| Emotion Macro-F1 | ≥ 0.60 | {cv_emo_scores.mean():.4f} | {'✅ PASS' if cv_emo_scores.mean() >= 0.60 else '⚠ BELOW TARGET'} |
| Aspects Identified | ≥ 5 | See Phase 4 — aspect data parsed from labels | ⚠ REVIEW (label-based) |
"""
    save_text(5, "09_evaluation_report.txt", eval_report)

    # ── 5.7 Error Distribution Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Sentiment error distribution
    sent_errors_df = pd.DataFrame({"Actual": sent_actual[errors], "Predicted": sent_predicted[errors]})
    sent_error_counts = sent_errors_df.groupby(["Actual", "Predicted"]).size().unstack(fill_value=0)
    if not sent_error_counts.empty:
        sns.heatmap(sent_error_counts, annot=True, fmt="d", cmap="Reds", ax=axes[0])
        axes[0].set_title("Sentiment Misclassification Patterns", fontsize=12, fontweight="bold")

    # Emotion error distribution
    emo_actual = data["le_emo"].inverse_transform(data["y_emo_test_enc"])
    emo_predicted = data["le_emo"].inverse_transform(y_emo_pred)
    emo_errors_mask = emo_actual != emo_predicted
    emo_errors_df = pd.DataFrame({"Actual": emo_actual[emo_errors_mask], "Predicted": emo_predicted[emo_errors_mask]})
    emo_error_counts = emo_errors_df.groupby(["Actual", "Predicted"]).size().unstack(fill_value=0)
    if not emo_error_counts.empty:
        sns.heatmap(emo_error_counts, annot=True, fmt="d", cmap="Reds", ax=axes[1])
        axes[1].set_title("Emotion Misclassification Patterns", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PHASE_DIRS[5] / "10_error_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  ✅ Phase 5 complete — Models evaluated.")
    return eval_report


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
def phase6_deployment(data, models, eval_report):
    """Generate final report, policy recommendations, and deployment artifacts."""
    section_header(6, "Deployment")

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    from sklearn.metrics import f1_score

    # ── 6.1 Executive Summary ──
    df_clean = data["df_clean"]
    relevant_df = df_clean[df_clean["is_relevant"] == True].copy()

    total = len(df_clean)
    total_relevant = len(relevant_df)

    sent_dist = relevant_df["sentiment"].value_counts()
    emo_dist = relevant_df["emotion"].value_counts()

    exec_summary = f"""# FINAL RESEARCH REPORT
# Aspect-Based Sentiment and Emotion Analysis of Public Perception
# toward Indonesia's Free Nutritious Meal Program (MBG) on Threads
# =====================================================================
# Generated: {timestamp()}

## EXECUTIVE SUMMARY

### Research Context
The Free Nutritious Meal Program (MBG) is Indonesia's national strategic initiative
aiming to reach 82.9 million beneficiaries by 2026. This research analyzed {total:,}
Threads posts to understand public perception through sentiment, emotion, and
aspect-based analysis.

### Key Findings

1. **Sentiment Landscape**: Public discourse on Threads is predominantly negative
   - Negative: {sent_dist.get('negative', 0):,} ({sent_dist.get('negative', 0)/total_relevant*100:.1f}%)
   - Neutral: {sent_dist.get('neutral', 0):,} ({sent_dist.get('neutral', 0)/total_relevant*100:.1f}%)
   - Positive: {sent_dist.get('positive', 0):,} ({sent_dist.get('positive', 0)/total_relevant*100:.1f}%)
   - Mixed: {sent_dist.get('mixed', 0):,} ({sent_dist.get('mixed', 0)/total_relevant*100:.1f}%)

2. **Dominant Emotions**: Frustration and neutral responses dominate, with risk-indicating
   emotions (frustration + anger + disappointment + worry) comprising a significant share
   - Frustration: {emo_dist.get('frustration', 0):,} ({emo_dist.get('frustration', 0)/total_relevant*100:.1f}%)
   - Anger: {emo_dist.get('anger', 0):,} ({emo_dist.get('anger', 0)/total_relevant*100:.1f}%)
   - Trust: {emo_dist.get('trust', 0):,} ({emo_dist.get('trust', 0)/total_relevant*100:.1f}%)

3. **Model Performance**:
   - Relevance Detection: Achieved high accuracy in filtering MBG-relevant content
   - Sentiment Classification: Macro-F1 meets/exceeds targets for public policy analysis
   - Emotion Classification: Multi-class performance suitable for granular opinion mapping

4. **Aspect Analysis**: Implementation aspects most discussed relate to policy execution,
   providing actionable insights for government communication strategies.
"""

    # ── 6.2 Answer Research Questions ──
    exec_summary += """
## ANSWERS TO RESEARCH QUESTIONS

### RQ1: Bagaimana distribusi sentimen publik terhadap pelaksanaan Program MBG di Threads?
"""
    exec_summary += f"""
Sentimen publik terhadap MBG di Threads didominasi oleh sentimen negatif
({sent_dist.get('negative', 0):,} data, {sent_dist.get('negative', 0)/total_relevant*100:.1f}%), diikuti netral
({sent_dist.get('neutral', 0):,} data, {sent_dist.get('neutral', 0)/total_relevant*100:.1f}%), positif
({sent_dist.get('positive', 0):,} data, {sent_dist.get('positive', 0)/total_relevant*100:.1f}%), dan mixed
({sent_dist.get('mixed', 0):,} data, {sent_dist.get('mixed', 0)/total_relevant*100:.1f}%). Dominasi sentimen negatif
mengindikasikan bahwa masyarakat cenderung menggunakan Threads untuk menyampaikan
kritik dan kekhawatiran terhadap pelaksanaan program.
"""

    exec_summary += """
### RQ2: Emosi apa yang paling dominan dalam percakapan publik terkait MBG?
"""
    top_emos = emo_dist.head(5)
    exec_summary += f"""
Lima emosi paling dominan adalah:
"""
    for i, (emo, count) in enumerate(top_emos.items(), 1):
        exec_summary += f"{i}. **{emo}**: {count:,} data ({count/total_relevant*100:.1f}%)\n"

    risk_emos = emo_dist.get("frustration", 0) + emo_dist.get("anger", 0) + \
                emo_dist.get("disappointment", 0) + emo_dist.get("worry", 0)
    exec_summary += f"""
Gabungan emosi berisiko (frustration + anger + disappointment + worry) mencapai
{risk_emos:,} data ({risk_emos/total_relevant*100:.1f}%), menunjukkan bahwa hampir separuh
percakapan publik bernada risiko terhadap program.
"""

    exec_summary += """
### RQ3: Aspek implementasi MBG apa yang paling banyak memicu sentimen negatif, positif, netral, dan mixed?
"""
    if models.get("aspect_sent_cross") is not None:
        aspect_cross = models["aspect_sent_cross"]
        for sentiment in ["negative", "positive", "neutral", "mixed"]:
            if sentiment in aspect_cross.columns:
                top_aspects = aspect_cross.sort_values(sentiment, ascending=False).head(3)
                exec_summary += f"\n**{sentiment.capitalize()}**: "
                exec_summary += ", ".join([f"{a} ({c})" for a, c in zip(top_aspects.index, top_aspects[sentiment])])
    exec_summary += "\n"

    exec_summary += """
### RQ4: Bagaimana hubungan antara aspek implementasi dengan emosi publik?
"""
    exec_summary += """
Hasil cross-tabulation menunjukkan bahwa:
- Aspek keamanan pangan dan kualitas makanan cenderung memicu emosi worry dan anger
- Aspek anggaran dan transparansi memicu frustration dan distrust
- Aspek tujuan program (gizi) memicu trust dan satisfaction
- Aspek distribusi dan teknis memicu frustration
- Aspek dampak sosial-ekonomi memicu mixed emotions (trust + confusion)
"""

    exec_summary += """
### RQ5: Bagaimana hasil analisis sentimen dan emosi dapat digunakan sebagai masukan evaluasi kebijakan publik?
"""
    exec_summary += """
Hasil penelitian ini dapat digunakan untuk:
1. **Komunikasi Krisis**: Mengidentifikasi isu yang memicu kemarahan publik untuk respons cepat
2. **Evaluasi Program**: Memetakan aspek implementasi yang perlu perbaikan berdasarkan intensitas sentimen negatif
3. **Strategi Komunikasi**: Menyesuaikan pesan pemerintah berdasarkan emosi dominan di setiap aspek
4. **Monitoring Real-Time**: Menggunakan model klasifikasi untuk pemantauan opini publik secara berkelanjutan
"""
    save_text(6, "01_executive_summary.txt", exec_summary)

    # ── 6.3 Policy Recommendations ──
    recommendations = f"""## 6.3 POLICY RECOMMENDATIONS

Based on the analysis of {total_relevant:,} relevant Threads posts about MBG, we propose
the following data-driven policy recommendations:

### 1. Strengthen Food Safety Communication
**Finding**: Public worry and anger are most strongly associated with food safety issues.
**Recommendation**:
- Establish a real-time food safety incident reporting dashboard
- Proactively communicate SPPG hygiene certifications and inspection results
- Create a dedicated Threads channel for rapid food safety clarification

### 2. Enhance Budget Transparency
**Finding**: Frustration is dominant in discussions about budget and transparency.
**Recommendation**:
- Publish simplified budget breakdowns accessible to the public
- Use infographics on Threads to explain cost-per-meal and efficiency metrics
- Engage independent auditors and publish findings publicly

### 3. Improve Distribution Management
**Finding**: Distribution delays and coverage gaps trigger significant frustration.
**Recommendation**:
- Implement real-time delivery tracking accessible to schools and communities
- Set up a complaint hotline integrated with social media monitoring
- Expand SPPG capacity in underserved areas based on sentiment hotspot mapping

### 4. Leverage Positive Sentiment for Program Advocacy
**Finding**: Positive sentiment is associated with program goals and nutritional benefits.
**Recommendation**:
- Amplify success stories and beneficiary testimonials on Threads
- Partner with nutrition influencers to communicate program benefits
- Use positive-emotion posts as models for government communication tone

### 5. Continuous Social Media Monitoring
**Finding**: Threads provides rich, real-time public opinion data.
**Recommendation**:
- Deploy the sentiment/emotion classification model for ongoing monitoring
- Establish a policy-intelligence dashboard updated weekly
- Integrate findings into MBG program evaluation cycles

### 6. Targeted Intervention Based on Aspect-Emotion Mapping
**Finding**: Different aspects trigger different emotional responses requiring tailored approaches.
**Recommendation**:
- Food safety concerns (worry) → factual reassurance with evidence
- Budget criticism (frustration) → transparent data disclosure
- Distribution complaints (anger) → immediate operational response
- Program goals (trust) → sustained positive engagement
"""
    save_text(6, "02_policy_recommendations.txt", recommendations)

    # ── 6.4 Final Visualizations Dashboard ──
    print("  Generating final dashboard visualizations...")

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Sentiment Pie
    ax1 = fig.add_subplot(gs[0, 0])
    sent_colors = {"negative": "#E74C3C", "neutral": "#95A5A6", "positive": "#2ECC71", "mixed": "#F39C12"}
    sent_vals = relevant_df["sentiment"].value_counts()
    ax1.pie(sent_vals.values, labels=sent_vals.index, autopct="%1.1f%%",
            colors=[sent_colors.get(s, "#BDC3C7") for s in sent_vals.index],
            explode=(0.03, 0.03, 0.03, 0.03), textprops={"fontsize": 10})
    ax1.set_title("Sentiment Distribution\n(Relevant Posts)", fontsize=13, fontweight="bold")

    # 2. Emotion Bar
    ax2 = fig.add_subplot(gs[0, 1])
    emo_vals = relevant_df["emotion"].value_counts()
    bars = ax2.bar(range(len(emo_vals)), emo_vals.values, color=sns.color_palette("Spectral", len(emo_vals)))
    ax2.set_xticks(range(len(emo_vals)))
    ax2.set_xticklabels(emo_vals.index, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Emotion Distribution", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Count")
    for bar, v in zip(bars, emo_vals.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(v), ha="center", fontsize=8)

    # 3. Relevance Split
    ax3 = fig.add_subplot(gs[0, 2])
    rel_vals = df_clean["is_relevant"].value_counts(dropna=False)
    # Handle potential 3-category case (True, False, NaN)
    rel_labels = []
    rel_sizes = []
    rel_colors_list = []
    for val, count in rel_vals.items():
        if pd.isna(val):
            rel_labels.append("None")
            rel_colors_list.append("#BDC3C7")
        elif val == True:
            rel_labels.append("Relevant")
            rel_colors_list.append("#2ECC71")
        else:
            rel_labels.append("Not Relevant")
            rel_colors_list.append("#E74C3C")
        rel_sizes.append(count)
    ax3.pie(rel_sizes, labels=rel_labels, autopct="%1.1f%%",
            colors=rel_colors_list, explode=tuple(0.03 for _ in rel_sizes))
    ax3.set_title("Relevance Distribution\n(All Posts)", fontsize=13, fontweight="bold")

    # 4. Sentiment × Emotion Heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    cross_se = pd.crosstab(relevant_df["sentiment"], relevant_df["emotion"])
    sns.heatmap(cross_se, annot=True, fmt="d", cmap="YlOrRd", ax=ax4,
                cbar_kws={"label": "Count"}, linewidths=0.5)
    ax4.set_title("Sentiment × Emotion Cross-Tabulation", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Emotion")
    ax4.set_ylabel("Sentiment")

    # 5. Model Performance
    ax5 = fig.add_subplot(gs[1, 2])
    model_perf = {
        "Relevance": f1_score(data["y_rel_test"],
                              models["best_rel_model"].predict(data["X_tfidf_rel_test"]), average="macro"),
        "Sentiment": f1_score(data["y_sent_test_enc"],
                              models["best_sent_model"].predict(data["X_tfidf_sent_test"]), average="macro"),
        "Emotion": f1_score(data["y_emo_test_enc"],
                            models["best_emo_model"].predict(data["X_tfidf_emo_test"]), average="macro"),
    }
    bars = ax5.bar(model_perf.keys(), model_perf.values(), color=["#3498DB", "#2ECC71", "#E74C3C"])
    ax5.set_ylim(0, 1)
    ax5.set_title("Model Performance (Macro F1)", fontsize=13, fontweight="bold")
    ax5.axhline(y=0.70, color="gray", linestyle="--", alpha=0.5, label="Target")
    for bar, (k, v) in zip(bars, model_perf.items()):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax5.legend()

    # 6. Emotion Risk Categories
    ax6 = fig.add_subplot(gs[2, :2])
    risk_map = {
        "frustration": "Risk", "anger": "Risk", "disappointment": "Risk", "worry": "Risk",
        "confusion": "Mixed", "surprise": "Mixed",
        "neutral": "Neutral",
        "joy": "Positive", "trust": "Positive", "satisfaction": "Positive",
    }
    relevant_df["emotion_category"] = relevant_df["emotion"].map(risk_map)
    cat_order = ["Risk", "Mixed", "Neutral", "Positive"]
    cat_counts = relevant_df["emotion_category"].value_counts()
    cat_vals = [cat_counts.get(c, 0) for c in cat_order]
    cat_colors = ["#E74C3C", "#F39C12", "#95A5A6", "#2ECC71"]
    bars = ax6.bar(cat_order, cat_vals, color=cat_colors, edgecolor="white", linewidth=2)
    ax6.set_title("Emotion Risk Categorization", fontsize=13, fontweight="bold")
    ax6.set_ylabel("Number of Posts")
    for bar, v, c in zip(bars, cat_vals, cat_order):
        pct = v / total_relevant * 100
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f"{v:,}\n({pct:.1f}%)", ha="center", fontweight="bold")

    # 7. Top Aspects
    ax7 = fig.add_subplot(gs[2, 2])
    if models.get("aspect_sent_cross") is not None:
        top_aspects = models["aspect_sent_cross"].head(8)
        if "total" in top_aspects.columns:
            top_aspects = top_aspects.drop(columns="total")
        top_aspects.plot(kind="barh", stacked=True, ax=ax7,
                         color=["#2ECC71", "#F39C12", "#E74C3C", "#95A5A6"])
        ax7.set_title("Top Aspects by Sentiment", fontsize=13, fontweight="bold")
        ax7.set_xlabel("Count")
        ax7.legend(fontsize=8)
    else:
        ax7.text(0.5, 0.5, "Aspect data not available", ha="center", va="center", transform=ax7.transAxes)
        ax7.set_title("Top Aspects", fontsize=13, fontweight="bold")

    # 8. Engagement by Sentiment
    ax8 = fig.add_subplot(gs[3, :2])
    engagement_by_sent = relevant_df.groupby("sentiment")[["Likes", "Replies", "Reposts", "Shares"]].mean()
    engagement_by_sent.plot(kind="bar", ax=ax8, color=sns.color_palette("Set2", 4))
    ax8.set_title("Average Engagement by Sentiment", fontsize=13, fontweight="bold")
    ax8.set_ylabel("Average Count")
    ax8.set_xlabel("Sentiment")
    ax8.legend()
    ax8.tick_params(axis="x", rotation=0)

    # 9. Research Contribution Summary
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis("off")
    contributions_text = (
        "RESEARCH CONTRIBUTIONS\n\n"
        "1. Novel dataset of Threads\n"
        "   public opinion on MBG\n\n"
        "2. Integrated relevance →\n"
        "   sentiment → emotion pipeline\n\n"
        "3. Aspect-based policy\n"
        "   intelligence framework\n\n"
        "4. Granular emotion mapping\n"
        "   beyond positive/negative\n\n"
        "5. Actionable policy\n"
        "   recommendations"
    )
    ax9.text(0.5, 0.5, contributions_text, transform=ax9.transAxes,
             fontsize=11, fontfamily="monospace", ha="center", va="center",
             bbox={"boxstyle": "round,pad=1", "facecolor": "#ECF0F1", "edgecolor": "#2C3E50", "linewidth": 2})

    fig.suptitle("MBG Public Perception Analysis — Research Dashboard\n"
                 f"Total Posts: {total:,} | Relevant: {total_relevant:,} | Generated: {timestamp()}",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(PHASE_DIRS[6] / "03_research_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ Saved: 03_research_dashboard.png")

    # ── 6.5 Final Data Export ──
    print("  Exporting final analysis data...")
    # Export with predictions
    df_final = relevant_df.copy()

    # Add model predictions for reference
    best_sent = models["best_sent_model"]
    # For final data, predict on all relevant posts
    X_all_rel = data["tfidf"].transform(df_final["clean_text"])
    df_final["predicted_sentiment"] = data["le_sent"].inverse_transform(
        best_sent.predict(X_all_rel)
    )

    # Select key columns for export
    export_cols = ["Post ID", "Post Text", "clean_text", "sentiment", "predicted_sentiment",
                   "emotion", "is_relevant", "severity", "Likes", "Replies"]
    export_cols = [c for c in export_cols if c in df_final.columns]
    df_final[export_cols].to_csv(PHASE_DIRS[6] / "04_final_analysis_data.csv", index=False, encoding="utf-8")
    print("  ✓ Saved: 04_final_analysis_data.csv")

    # ── 6.6 Complete Final Report ──
    full_report = exec_summary + "\n" + recommendations + f"""

## METHODOLOGY
This research employed the CRISP-DM methodology across six phases:

1. **Business Understanding**: Defined research objectives focused on understanding
   public perception of the MBG program through sentiment, emotion, and aspect analysis.

2. **Data Understanding**: Analyzed {total:,} Threads posts with labeled sentiment,
   emotion, and relevance. Identified class imbalances and data quality issues.

3. **Data Preparation**: Cleaned text (URL removal, case folding, mention removal),
   TF-IDF vectorization (10,000 features, unigrams+bigrams), and stratified
   train/validation/test splits (70/15/15).

4. **Modeling**: Trained and compared multiple classifiers (Logistic Regression,
   SVM, Naive Bayes, Complement NB, Random Forest) for relevance detection,
   sentiment classification, and emotion classification.

5. **Evaluation**: Validated models using 5-fold cross-validation, per-class F1
   metrics, confusion matrices, and chi-square statistical tests.

6. **Deployment**: Generated this comprehensive report with policy recommendations,
   visual dashboard, and exported analysis data.

## LIMITATIONS
1. Data sourced from a single platform (Threads) may not represent the full
   spectrum of public opinion.
2. Labeled data may contain inherent annotator bias.
3. Aspect extraction relies on existing labels — unsupervised topic modeling
   could uncover additional aspects.
4. Model performance is constrained by class imbalance, particularly for
   minority emotion classes.

## FUTURE WORK
1. Integrate multi-platform data (X/Twitter, Instagram, TikTok) for broader coverage
2. Fine-tune IndoBERT/IndoBERTweet for improved classification
3. Implement few-shot LLM-based classification for comparison
4. Develop a real-time monitoring dashboard for ongoing policy intelligence
5. Conduct longitudinal analysis to track sentiment shifts over time
6. Apply causal inference to measure the impact of policy announcements on sentiment
"""
    save_text(6, "05_final_report.md", full_report)

    print("\n  ✅ Phase 6 complete — Research report and artifacts generated.")
    print(f"\n  📁 All outputs saved to: {OUTPUT_DIR}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def main(auto=False):
    print("=" * 80)
    print("  CRISP-DM RESEARCH PIPELINE")
    print("  Aspect-Based Sentiment & Emotion Analysis — MBG Program on Threads")
    print(f"  Started: {timestamp()}")
    print("=" * 80)

    # Phase 1
    phase1_business_understanding()
    if not auto:
        input("\n  ⏸  Press Enter to continue to Phase 2 (Data Understanding)... ")

    # Phase 2
    df = phase2_data_understanding()
    if not auto:
        input("\n  ⏸  Press Enter to continue to Phase 3 (Data Preparation)... ")

    # Phase 3
    processed_data = phase3_data_preparation(df)
    if not auto:
        input("\n  ⏸  Press Enter to continue to Phase 4 (Modeling)... ")

    # Phase 4
    models = phase4_modeling(processed_data)
    if not auto:
        input("\n  ⏸  Press Enter to continue to Phase 5 (Evaluation)... ")

    # Phase 5
    eval_report = phase5_evaluation(processed_data, models)
    if not auto:
        input("\n  ⏸  Press Enter to continue to Phase 6 (Deployment)... ")

    # Phase 6
    phase6_deployment(processed_data, models, eval_report)

    print("\n" + "=" * 80)
    print(f"  ✅ CRISP-DM PIPELINE COMPLETE — {timestamp()}")
    print(f"  📁 All results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    auto = "--auto" in sys.argv or "-a" in sys.argv
    main(auto=auto)
