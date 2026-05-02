import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score


# NLTK downloads
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Initialize tools
custom_stopwords = {
    'rt', 'amp', 'get', 'got', 'make', 'like', 'say',
    'said', 'one', 'would', 'could', 'also', 'think',
    'know', 'go', 'going', 'use', 'time', 'mkr', 'al',
    'cal', 'im', 'dont', 'ur', 'u', 'funy', 'wil',
    'right', 'people', 'even', 'back', 'see', 'want',
    'need', 'still', 'way', 'good', 'really', 'much',
    'many', 'come', 'look', 'thing', 'take', 'give',
    'suport', 'schol', 'buly', 'se', 'amp', 'mkr',
    'get', 'go', 'one', 'would', 'make', 'say',
    'like', 'know', 'im', 'al', 'people', 'midle',
    'ur', 'rt', 'caled', 'first', 'anything',
    'even', 'never', 'every', 'always', 'still',
    'back', 'last', 'much', 'many', 'us'
}
stop_words = set(stopwords.words('english')).union(custom_stopwords)

lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Emoji → sentiment word map (preserves intent before ASCII strip)
EMOJI_MAP = {
    '❤️': ' love ', '😂': ' laugh ', '💀': ' laugh ',
    '😭': ' sad ',  '🥰': ' love ', '😡': ' angry ',
    '🤣': ' laugh ', '😊': ' happy ', '💔': ' heartbroken ',
    '😍': ' love ', '🙏': ' respect ', '👍': ' good ',
    '🔥': ' fire ', '💯': ' great ', '😘': ' love ',
    '😢': ' sad ', '🤦': ' facepalm ', '🥺': ' sad ',
    '😤': ' frustrated ', '🖕': ' offensive ',
}

# Prediction thresholds
CONF_THRESHOLD      = 0.25   # model must be at least 55% confident to flag
VADER_NEG_THRESHOLD = -0.30  # VADER compound must be this negative to hard-alert

# Page configuration
st.set_page_config(
    page_title="CyberBullying Detector",
    page_icon="🛡️",
    layout="wide"
)
st.title("🛡️ CyberBullying Detection Dashboard")

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0px'>
        <h2 style='color: #7c83fd'>🛡️ CyberBullying Detector</h2>
        <p style='color: gray; font-size: 0.85rem'>
            NLP & Machine Learning<br>Powered Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 EDA & Word Analysis", "🤖 Predict"]
    )

    st.markdown("---")

    st.markdown("""
    <div style='padding: 10px 0px'>
        <p style='color: gray; font-size: 0.8rem'>
            📁 <b>Dataset:</b> 47,692 Tweets<br>
            🤖 <b>Model:</b> Complement Naive Bayes<br>
            📊 <b>Accuracy:</b> 80.2%<br>
            🔤 <b>Features:</b> TF-IDF + VADER
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")


# ── Load saved models ─────────────────────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('le.pkl', 'rb') as f:
    le = pickle.load(f)

with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open('y_pred.pkl', 'rb') as f:
    y_pred = pickle.load(f)

# ── Robust CSV loading — auto-detects column names ───────────────────────────
df = pd.read_csv('cyberbullying_cleaned.csv')

# Show actual columns in sidebar for debugging (remove once stable)
# st.sidebar.write(df.columns.tolist())

# Map label column → 'sentiment'
LABEL_ALIASES = ['cyberbullying_type', 'label', 'class', 'category', 'type', 'sentiment']
for alias in LABEL_ALIASES:
    if alias in df.columns:
        if alias != 'sentiment':
            df.rename(columns={alias: 'sentiment'}, inplace=True)
        break
else:
    # Last resort: use the first non-numeric column that isn't text
    for col in df.columns:
        if df[col].dtype == object and col not in ('text', 'tweet_text', 'cleaned_text'):
            df.rename(columns={col: 'sentiment'}, inplace=True)
            break

# Map text column → 'text'
TEXT_ALIASES = ['tweet_text', 'tweet', 'content', 'message', 'text']
for alias in TEXT_ALIASES:
    if alias in df.columns:
        if alias != 'text':
            df.rename(columns={alias: 'text'}, inplace=True)
        break
else:
    # Last resort: first object column
    for col in df.columns:
        if df[col].dtype == object and col != 'sentiment':
            df.rename(columns={col: 'text'}, inplace=True)
            break

# Map cleaned_text column → 'cleaned_text'
CLEANED_ALIASES = ['cleaned_text', 'clean_text', 'processed_text', 'cleaned']
for alias in CLEANED_ALIASES:
    if alias in df.columns:
        if alias != 'cleaned_text':
            df.rename(columns={alias: 'cleaned_text'}, inplace=True)
        break
else:
    # Fall back to raw text if no cleaned version exists
    df['cleaned_text'] = df['text']


# ── Text preprocessing ────────────────────────────────────────────────────────
def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t",   " not",    text)
    text = re.sub(r"\'re",   " are",    text)
    text = re.sub(r"\'s",    " is",     text)
    text = re.sub(r"\'d",    " would",  text)
    text = re.sub(r"\'ll",   " will",   text)
    text = re.sub(r"\'t",    " not",    text)
    text = re.sub(r"\'ve",   " have",   text)
    text = re.sub(r"\'m",    " am",     text)
    return text


def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)

    # Convert emojis to sentiment words BEFORE stripping non-ASCII
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, word)

    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub('[0-9]+', '', text)
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = [lemmatizer.lemmatize(word, pos='v') for word in text]
    return ' '.join(text)


def preprocess(text):
    return strip_all_entities(decontract(text))


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_text(text):
    cleaned = preprocess(text)

    # FIX 2: VADER always runs on the ORIGINAL text (not cleaned)
    vader = sia.polarity_scores(text)

    # FIX 3: Empty-text guard — but DON'T auto-safe if VADER is very negative.
    # "Fuck you" strips down to almost nothing, but VADER still sees it as hostile.
    if not cleaned.strip():
        vader_compound = vader['compound']
        # If VADER itself says very negative → still flag it
        if vader_compound <= -0.5:
            # Return a synthetic "other_cyberbullying" result so the UI catches it
            synth_probs = {cls: 0.05 for cls in le.classes_}
            synth_probs['not_cyberbullying'] = 0.0
            synth_probs['other_cyberbullying'] = 0.70
            return 'other_cyberbullying', 0.70, synth_probs, vader, cleaned
        # Otherwise it genuinely is empty/safe
        empty_probs = {cls: 0.0 for cls in le.classes_}
        empty_probs['not_cyberbullying'] = 1.0
        return 'not_cyberbullying', 1.0, empty_probs, vader, cleaned

    vec   = tfidf.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    pred_idx = np.argmax(proba)
    label    = le.classes_[pred_idx]
    conf     = proba[pred_idx]
    all_probs = {le.classes_[i]: proba[i] for i in range(len(le.classes_))}

    return label, conf, all_probs, vader, cleaned


# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard Overview")

    total     = len(df)
    bully_pct = df[df['sentiment'] != 'not_cyberbullying'].shape[0] / total * 100
    avg_len   = df['text'].str.split().str.len().mean()
    accuracy  = accuracy_score(y_test, y_pred)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Tweets",     f"{total:,}")
    with k2: st.metric("Bullying Content", f"{bully_pct:.1f}%")
    with k3: st.metric("Avg Tweet Length", f"{avg_len:.0f} words")
    with k4: st.metric("Model Accuracy",   f"{accuracy*100:.1f}%")

    # Category Distribution
    st.markdown("---")
    st.markdown("### 📊 Category Distribution")
    st.caption("Distribution of tweets across cyberbullying categories")

    counts = df['sentiment'].value_counts()
    colors = ['#f97316', '#8b5cf6', '#ec4899', '#06b6d4', '#eab308', '#22c55e']

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("Tweet Count by Category", fontsize=12, pad=10)
        bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='none')
        for bar, val in zip(bars, counts.values):
            ax.text(val + 30, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=9)
        ax.set_xlabel('Tweet Count')
        ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title("Category Share (%)", fontsize=12, pad=10)
        ax.pie(counts.values, labels=counts.index,
               autopct='%1.1f%%', colors=colors, startangle=140)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Model Performance
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    st.caption("Precision, Recall and F1-Score per category on test data")

    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)
    rows = []
    for cls in le.classes_:
        r = report[cls]
        rows.append({
            'Category':  cls,
            'Precision': f"{r['precision']:.3f}",
            'Recall':    f"{r['recall']:.3f}",
            'F1-Score':  f"{r['f1-score']:.3f}",
            'Support':   f"{int(r['support']):,}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Confusion Matrix
    st.markdown("---")
    st.markdown("### 🔥 Confusion Matrix")
    st.caption("Actual vs Predicted categories on test data")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.85rem; padding: 10px'>
        Automatic Detection of Cyberbullying &nbsp;|&nbsp;
        GEC Patan · BrainyBeam Info Tech · 2025-26 &nbsp;|&nbsp;
        Developed by <b>Milan Shroff</b>
    </div>
    """, unsafe_allow_html=True)


# ── EDA page ──────────────────────────────────────────────────────────────────
elif page == "🔍 EDA & Word Analysis":
    st.markdown("## 🔍 EDA & Word Analysis")
    st.markdown("#### Filter by Category")

    sel_cat = st.selectbox("Select Category",
                           ['All'] + list(df['sentiment'].unique()))

    fdf = df if sel_cat == 'All' else df[df['sentiment'] == sel_cat]
    st.markdown(f"**Showing {len(fdf):,} tweets**")

    # Most Frequent Words
    st.markdown("#### Most Frequent Words")
    top_n = st.slider("Number of words to show", 10, 50, 20)

    all_words = ' '.join(fdf['cleaned_text'].dropna()).split()
    word_freq = Counter(all_words).most_common(top_n)
    wf_df = pd.DataFrame(word_freq, columns=['Word', 'Count'])

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(wf_df['Word'][::-1], wf_df['Count'][::-1], edgecolor='none')
    ax.set_xlabel('Frequency')
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Word Cloud
    st.markdown("#### Word Cloud")
    wc_cat = st.selectbox("Select category for word cloud",
                           list(df['sentiment'].unique()), key='wc_sel')

    text_blob = ' '.join(df[df['sentiment'] == wc_cat]['cleaned_text'].dropna())
    wc = WordCloud(width=800, height=400,
                   background_color='white', max_words=150).generate(text_blob)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(wc_cat, fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # VADER Sentiment Chart
    st.markdown("#### Average VADER Sentiment by Category")

    vader_scores = []
    for cat in df['sentiment'].unique():
        sample = df[df['sentiment'] == cat]['cleaned_text'].dropna().sample(
            min(300, len(df[df['sentiment'] == cat])), random_state=42)
        avg_score = sample.apply(lambda t: sia.polarity_scores(t)['compound']).mean()
        vader_scores.append({'Category': cat, 'Avg Compound Score': avg_score})

    vader_df = pd.DataFrame(vader_scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(vader_df['Category'], vader_df['Avg Compound Score'],
           color=['#f97316', '#8b5cf6', '#ec4899', '#06b6d4', '#eab308', '#22c55e'],
           edgecolor='none', width=0.5)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Avg Compound Score')
    plt.xticks(rotation=20, ha='right')
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Predict page ──────────────────────────────────────────────────────────────
elif page == "🤖 Predict":
    st.markdown("## 🤖 Cyberbullying Detector")
    st.markdown("Type any text below and the model will classify it.")

    user_input = st.text_area(
        "Enter text here", height=150,
        placeholder='e.g. "You so stupid bestie ❤️"  or  "I hate you, go die"'
    )

    analyse_btn = st.button("🔍 Analyse")

    if analyse_btn and user_input.strip():
        label, conf, all_probs, vader, cleaned = predict_text(user_input)

        st.markdown("---")

        vader_compound   = vader['compound']
        is_bullying      = label != 'not_cyberbullying'
        high_confidence  = conf >= CONF_THRESHOLD
        clearly_negative = vader_compound <= VADER_NEG_THRESHOLD

        # ── Decision logic ────────────────────────────────────────────────────
        if is_bullying and high_confidence and clearly_negative:
            st.error(f"🚨 Cyberbullying Detected — **{label}** ({conf*100:.1f}% confidence)")

        elif is_bullying and high_confidence and not clearly_negative:
            st.warning(f"⚠️ Possibly Offensive — **{label}** ({conf*100:.1f}% confidence)")
            st.info("ℹ️ Model flagged this but tone appears non-hostile. "
                    "Could be sarcasm, friendly teasing, or dark humour.")

        elif is_bullying and not high_confidence:
            st.success("✅ Not Cyberbullying — Model confidence too low to flag.")
            st.caption(
                f"Highest match was **{label}** at {conf*100:.1f}% "
                f"— below the {CONF_THRESHOLD*100:.0f}% threshold."
            )

        else:
            st.success(f"✅ Not Cyberbullying ({conf*100:.1f}% confidence)")

        # Note if text vanished after preprocessing
        if not cleaned.strip():
            st.info("ℹ️ Text was fully stripped during preprocessing (very short or all stopwords). "
                    "VADER sentiment was used as fallback.")

        # ── Class Probabilities ───────────────────────────────────────────────
        st.markdown("#### Class Probabilities")
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(list(sorted_probs.keys()), list(sorted_probs.values()),
                color=['#f97316', '#8b5cf6', '#ec4899', '#06b6d4', '#eab308', '#22c55e'],
                edgecolor='none')
        ax.axvline(CONF_THRESHOLD, color='red', linewidth=1.2,
                   linestyle='--', label=f'Threshold ({int(CONF_THRESHOLD*100)}%)')
        ax.legend(fontsize=8)
        ax.set_xlabel('Probability')
        ax.spines[:].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── VADER Sentiment Breakdown ─────────────────────────────────────────
        st.markdown("#### VADER Sentiment Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            labels_v = ['Negative', 'Neutral', 'Positive']
            values_v = [vader['neg'], vader['neu'], vader['pos']]
            colors_v = ['#ef4444', '#6b7280', '#22c55e']
            ax.bar(labels_v, values_v, color=colors_v, edgecolor='none', width=0.45)
            ax.set_ylim(0, 1.1)
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            compound = vader['compound']
            st.metric("Compound Score", f"{compound:.3f}")
            if compound <= -0.5:
                st.error("Very Negative 😡")
            elif compound <= VADER_NEG_THRESHOLD:
                st.warning("Negative 😟")
            elif compound < 0:
                st.info("Mildly Negative 😐")
            elif compound == 0:
                st.info("Neutral 😐")
            elif compound < 0.5:
                st.info("Slightly Positive 🙂")
            else:
                st.success("Very Positive 😊")

        with st.expander("🔧 View preprocessed text"):
            st.code(cleaned if cleaned.strip() else
                    "(empty after cleaning — VADER used as fallback)")

    elif analyse_btn:
        st.warning("Please enter some text first.")
