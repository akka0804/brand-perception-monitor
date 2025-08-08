# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px

# -----------------------------
# Utility: Mock data generator
# -----------------------------
def get_mock_mentions(n=40):
    platforms = ["Twitter", "Reddit", "Google News", "YouTube", "Instagram"]
    seed_samples = [
        # positive
        "LeapScholar helped me secure a scholarship abroad!",
        "Thanks to LeapScholar, my UK student visa got approved in record time! 🎉",
        "The SOP tips from LeapScholar actually got me shortlisted. Highly recommend.",
        "LeapScholar's webinars are highly insightful and well structured.",
        "Got my dream admit to University of Toronto! Thank you LeapScholar ❤️",
        "LeapScholar truly cares about student success — excellent counselors.",
        "Fantastic platform for overseas education guidance!",
        "Counselors were friendly and highly professional.",
        "LeapScholar's advice on SOP writing is top-notch.",
        # neutral / informational
        "LeapScholar launches new AI tool to assist students in visa documentation.",
        "Has anyone used LeapScholar for Australia admissions? My experience has been smooth so far.",
        "Watched LeapScholar's video on US visa interviews — helpful for first-timers.",
        "LeapScholar’s content on reels is motivating for aspiring students.",
        "LeapScholar has opened a new center in Mumbai.",
        # negative
        "Terrible service, I would not recommend LeapScholar.",
        "LeapScholar promised quick replies but I’ve been waiting for 5 days. Disappointed.",
        "Too many spam emails from LeapScholar lately. Anyone else facing this?",
        "Poor communication, not worth the time.",
        "LeapScholar just feels like a money-making business.",
        "Had a bad counseling session today, felt rushed and ignored.",
        "The support team responds very late — frustrating experience.",
        "Worst experience, complete waste of money.",
        "LeapScholar app needs a lot of improvement; keeps crashing."
    ]

    data = []
    today = datetime.utcnow().date()
    for i in range(n):
        mention = random.choice(seed_samples)
        platform = random.choice(platforms)
        # Random date in last 30 days
        rand_days = random.randint(0, 29)
        dt = datetime.combine(today - timedelta(days=rand_days), datetime.min.time()) + timedelta(hours=random.randint(6, 23))
        data.append({"platform": platform, "mention": mention, "datetime": dt})
    df = pd.DataFrame(data).sort_values("datetime", ascending=False).reset_index(drop=True)
    return df

# -----------------------------
# Sentiment analysis (VADER)
# -----------------------------
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(df):
    df = df.copy()
    df["sentiment_score"] = df["mention"].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    df["sentiment_label"] = df["sentiment_score"].apply(lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral"))
    return df

# -----------------------------
# Rule-based emotion detection (lightweight)
# -----------------------------
EMOTION_KEYWORDS = {
    "joy": ["thank", "thanks", "happy", "great", "congrats", "excited", "🎉", "❤️", "fantastic", "helpful", "insightful", "love", "recommend"],
    "anger": ["terrible", "worst", "frustrat", "angry", "spam", "scam", "money-making", "rude"],
    "sadness": ["disappoint", "sad", "regret", "unhappy", "waste"],
    "fear": ["scared", "afraid", "worried", "concern"],
    "surprise": ["surpris", "wow", "unexpected", "shock"],
}

def detect_emotion(text):
    t = str(text).lower()
    scores = {k: 0 for k in EMOTION_KEYWORDS.keys()}
    for emo, kws in EMOTION_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[emo] += 1
    # choose top emoji or neutral
    if all(v == 0 for v in scores.values()):
        return "neutral"
    return max(scores, key=scores.get)

def add_emotions(df):
    df = df.copy()
    df["emotion"] = df["mention"].apply(detect_emotion)
    return df

# -----------------------------
# Topic extraction with TF-IDF
# -----------------------------
def extract_topics(df, top_n=8):
    docs = df["mention"].astype(str).tolist()
    if len(docs) == 0:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
    X = vectorizer.fit_transform(docs)
    # sum tf-idf for each term across documents
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_idx = scores.argsort()[::-1][:top_n]
    top_terms = list(zip(terms[top_idx], scores[top_idx]))
    return top_terms

# -----------------------------
# Page setup & theme toggle
# -----------------------------
st.set_page_config(page_title="Brand Perception Monitor", layout="wide", initial_sidebar_state="expanded")

if "page" not in st.session_state:
    st.session_state.page = "home"

# Theme selection
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

# Blue -> Purple gradient colors
GRADIENT_DARK = "linear-gradient(135deg, #07103a 0%, #2a0f6b 50%, #5b2bb5 100%)"
GRADIENT_LIGHT = "linear-gradient(135deg, #e6f0ff 0%, #e8d9ff 50%, #f2e6ff 100%)"

if theme == "Dark":
    page_bg = GRADIENT_DARK
    text_color = "#E6EDF3"
    card_bg = "rgba(255,255,255,0.03)"
    accent = "#8A63FF"   # purple accent
    positive = "#2DD4BF"
    negative = "#FF6B6B"
    neutral_clr = "#A3A3A3"
else:
    page_bg = GRADIENT_LIGHT
    text_color = "#0b1320"
    card_bg = "rgba(0,0,0,0.04)"
    accent = "#4B6CFF"   # bluish accent
    positive = "#0ea5a4"
    negative = "#ef4444"
    neutral_clr = "#6b7280"

# Inject CSS (full-page gradient + remove default white containers)
st.markdown(
    f"""
    <style>
    /* full page gradient */
    .stApp {{
        background: {page_bg};
        background-attachment: fixed;
    }}
    /* Make the main container transparent so gradient shows through */
    .block-container {{
        background: transparent;
        color: {text_color};
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    /* Streamlit cards and elements tweak */
    .st-expander > .st-expander__container {{
        background: {card_bg} !important;
        border-radius: 12px;
        padding: 0.8rem;
    }}
    .css-1d391kg {{}}
    /* Buttons */
    .stButton>button {{
        background: {accent} !important;
        color: white !important;
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
    }}
    /* Make table headers readable */
    .stDataFrame table thead tr th {{
        color: {text_color} !important;
    }}
    /* metric styling */
    .metric-card {{
        background: {card_bg};
        padding: 14px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.04);
    }}
    /* mention cards */
    .mention-card {{
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        background: rgba(255,255,255,0.02);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Home / Landing page
# -----------------------------
if st.session_state.page == "home":
    # header area
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<div style='font-size:28px;font-weight:800;color:{text_color}'>📊 Brand Perception Monitor</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{text_color};opacity:0.85;margin-top:6px'>Quick brand intelligence for LeapScholar — sentiment, topics & alerts.</div>", unsafe_allow_html=True)
    with col2:
        # replaceable front image (we use an inline svg-like placeholder for offline safety)
        svg_placeholder = f"""
<div style='width:100%;height:160px;border-radius:14px;
background:linear-gradient(90deg, rgba(74,144,226,0.12), rgba(136,77,255,0.12));
display:flex;align-items:center;justify-content:center;'>
    <h2 style='color:#884dff;font-weight:700;'>Brand Perception Monitor</h2>
</div>
"""

        st.markdown(svg_placeholder, unsafe_allow_html=True)


    st.markdown(f"<div style='text-align:center'><button onclick=\"window.dispatchEvent(new Event('streamlit:runFunction'))\" class='stButton'>🚀 Let's Monitor</button></div>", unsafe_allow_html=True)

    # The above fake button triggers nothing by itself; use a normal st.button as fallback:
    if st.button("🚀 Let's Monitor"):
        st.session_state.page = "monitor"

    st.markdown("---")
    st.markdown("<div style='color:{0};opacity:0.9'><b>What this demo includes</b></div>".format(text_color), unsafe_allow_html=True)
    st.markdown(
        """
        - Multi-platform mock mentions (Twitter, Reddit, News, YouTube, Instagram)\n
        - VADER sentiment + simple emotion detection\n
        - Topic extraction (TF-IDF)\n
        - Time-series sentiment trend and top mentions\n
        - CSV download + simulated alert when negatives spike
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Monitoring dashboard
# -----------------------------
elif st.session_state.page == "monitor":
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between'><div style='font-size:28px;font-weight:800;color:{text_color}'>📊 Brand Perception Monitor Dashboard</div><div></div></div>", unsafe_allow_html=True)
    st.markdown("<br>")

    # Load & process data
    df = get_mock_mentions(n=60)
    df = analyze_sentiment(df)
    df = add_emotions(df)

    # Sidebar filters
    with st.sidebar:
        st.header("Filters & Actions")
        platforms = ["All"] + sorted(df["platform"].unique().tolist())
        platform = st.selectbox("Platform", platforms, index=0)
        # date range
        min_date = df["datetime"].min().date()
        max_date = df["datetime"].max().date()
        date_range = st.date_input("Date range", [min_date, max_date])
        if st.button("Refresh data"):
            df = get_mock_mentions(n=60); df = analyze_sentiment(df); df = add_emotions(df)

        st.markdown("---")
        st.markdown("Download")
        st.download_button("Download CSV", df.to_csv(index=False), "brand_mentions.csv", "text/csv")
        st.markdown("---")
        st.write("Settings")
        neg_threshold = st.slider("Negative share alert threshold (%)", 5, 60, 20)

    # apply filters
    if platform != "All":
        df = df[df["platform"] == platform]
    # date filter
    start_date, end_date = date_range
    df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)]

    # KPIs
    col1, col2, col3, col4 = st.columns([1.2,1,1,1])
    with col1:
        st.markdown(f"<div class='metric-card'><div style='font-size:14px;color:{text_color};opacity:0.8'>Total Mentions</div><div style='font-size:22px;font-weight:700;color:{accent}'>{len(df)}</div></div>", unsafe_allow_html=True)
    with col2:
        avg_sent = df["sentiment_score"].mean() if len(df)>0 else 0
        avg_sent_display = ((avg_sent + 1) * 5)
        st.markdown(f"<div class='metric-card'><div style='font-size:14px;color:{text_color};opacity:0.8'>Avg Sentiment (0-10)</div><div style='font-size:22px;font-weight:700;color:{accent}'>{avg_sent_display:.1f}</div></div>", unsafe_allow_html=True)
    with col3:
        # brand health = weighted (positive% - negative%) scaled
        pos_pct = (df["sentiment_label"]=="Positive").mean() * 100 if len(df)>0 else 0
        neg_pct = (df["sentiment_label"]=="Negative").mean() * 100 if len(df)>0 else 0
        health = max(0, min(100, pos_pct - neg_pct + 50))  # arbitrary 0-100
        st.markdown(f"<div class='metric-card'><div style='font-size:14px;color:{text_color};opacity:0.8'>Brand Health</div><div style='font-size:22px;font-weight:700;color:{accent}'>{health:.0f}%</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div style='font-size:14px;color:{text_color};opacity:0.8'>Top Emotion</div><div style='font-size:18px;font-weight:700;color:{accent}'>{df['emotion'].mode().iloc[0] if len(df)>0 else '—'}</div></div>", unsafe_allow_html=True)

    st.markdown("")

    # Sentiment distribution (plotly)
    sentiment_counts = df["sentiment_label"].value_counts().reindex(["Positive","Neutral","Negative"]).fillna(0)
    sent_df = pd.DataFrame({"sentiment": sentiment_counts.index, "count": sentiment_counts.values})
    fig = px.pie(sent_df, names="sentiment", values="count", hole=0.45,
                 color="sentiment", color_discrete_map={"Positive": positive, "Neutral": neutral_clr, "Negative": negative})
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment over time
    ts = df.copy()
    if len(ts) > 0:
        ts["date"] = ts["datetime"].dt.date
        trend = ts.groupby(["date","sentiment_label"]).size().unstack(fill_value=0)
        trend = trend.reset_index().melt(id_vars="date", value_name="count", var_name="sentiment")
        fig2 = px.line(trend, x="date", y="count", color="sentiment",
                       color_discrete_map={"Positive": positive, "Neutral": neutral_clr, "Negative": negative},
                       markers=True)
        fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), legend=dict(orientation="h", y=-0.2))
        st.subheader("Sentiment trend")
        st.plotly_chart(fig2, use_container_width=True)

    # Topics
    st.subheader("Top topics (keywords)")
    top_terms = extract_topics(df, top_n=10)
    if top_terms:
        terms, scores = zip(*top_terms)
        topic_df = pd.DataFrame({"term": terms, "score": scores})
        fig3 = px.bar(topic_df, x="score", y="term", orientation="h", color="term", color_discrete_sequence=px.colors.sequential.Blues)
        fig3.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No significant topics found.")

    # Top positive & negative mentions
    st.subheader("Key conversations")
    pos = df[df["sentiment_label"]=="Positive"].nlargest(3, "sentiment_score")
    neg = df[df["sentiment_label"]=="Negative"].nsmallest(3, "sentiment_score")

    colp, coln = st.columns(2)
    with colp:
        st.markdown(f"<div style='font-weight:700;color:{text_color}'>🌟 Top Positive</div>", unsafe_allow_html=True)
        if len(pos)==0:
            st.info("No positive mentions.")
        else:
            for _, r in pos.iterrows():
                st.markdown(f"<div class='mention-card' style='border-left:4px solid {positive};'><b>[{r['platform']}]</b> {r['mention']}<div style='opacity:0.8;color:{text_color};font-size:12px'>Score: {r['sentiment_score']:.2f} · Emotion: {r['emotion']}</div></div>", unsafe_allow_html=True)
    with coln:
        st.markdown(f"<div style='font-weight:700;color:{text_color}'>🚨 Top Negative</div>", unsafe_allow_html=True)
        if len(neg)==0:
            st.info("No negative mentions.")
        else:
            for _, r in neg.iterrows():
                st.markdown(f"<div class='mention-card' style='border-left:4px solid {negative};'><b>[{r['platform']}]</b> {r['mention']}<div style='opacity:0.8;color:{text_color};font-size:12px'>Score: {r['sentiment_score']:.2f} · Emotion: {r['emotion']}</div></div>", unsafe_allow_html=True)

    # Raw data
    st.markdown("---")
    st.subheader("All mentions")
    st.dataframe(df[["datetime","platform","mention","sentiment_score","sentiment_label","emotion"]].sort_values("datetime", ascending=False), use_container_width=True)

    # Simulated alert when negative share spikes
    neg_share = (df["sentiment_label"] == "Negative").mean() * 100 if len(df)>0 else 0
    if neg_share >= neg_threshold:
        st.error(f"🚨 Negative mentions are {neg_share:.1f}% which is above your threshold of {neg_threshold}%. Suggest reviewing urgent items.")
        if st.button("Simulate: Send alert email to Marketing Head"):
            st.success("✅ Alert simulated (email would be sent in production).")

    st.markdown("<br><div style='text-align:center'><button onclick=\"window.location.reload();\" class='stButton'>Back to Home</button></div>", unsafe_allow_html=True)
    if st.button("Back to Home"):
        st.session_state.page = "home"