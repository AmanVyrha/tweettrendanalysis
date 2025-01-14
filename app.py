import tweepy
from textblob import TextBlob
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import defaultdict


bearer_token = "BEARER_TOKEN"

client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)


# Functions for Data Acquisition, Processing, and Analysis

def get_tweets(keyword, count=100, retries=3, retry_delay=5):
    tweets_data = []
    for attempt in range(retries):
        try:
            response = client.search_recent_tweets(
                query=f"{keyword} lang:en -is:retweet",
                max_results=count,
                tweet_fields=["created_at", "text", "public_metrics"],
                user_fields=["username", "public_metrics"],
                expansions=["author_id"]
            )

            users = {u["id"]: u for u in response.includes['users']}

            for tweet in response.data:
                analysis = TextBlob(tweet.text)
                sentiment = "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"

                user = users[tweet.author_id]

                tweets_data.append({
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "sentiment": sentiment,
                    "polarity": analysis.sentiment.polarity,
                    "retweet_count": tweet.public_metrics['retweet_count'],
                    "reply_count": tweet.public_metrics['reply_count'],
                    "like_count": tweet.public_metrics['like_count'],
                    "quote_count": tweet.public_metrics['quote_count'],
                    "user_name": user.username,
                    "user_followers_count": user.public_metrics['followers_count']
                })
            return pd.DataFrame(tweets_data)

        except tweepy.errors.TweepyException as e:
            st.error(f"An error occurred while fetching tweets: {e}")
            if attempt < retries - 1:
                st.warning(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                st.error("Failed to fetch tweets after multiple retries.")
                return pd.DataFrame()

    return pd.DataFrame(tweets_data)


def analyze_topics(df, num_topics=5, num_words=5):
    """
    Performs basic topic analysis using word counts.
    (Consider replacing with LDA or NMF for more advanced analysis)
    """
    from collections import Counter
    import re

    word_counts = Counter()
    for text in df["text"]:
        text = re.sub(r'[^\w\s]', '', text).lower()
        words = text.split()
        word_counts.update(words)

    topics = {}
    for i in range(num_topics):
        top_words = [word for word, count in word_counts.most_common(num_words)]
        topics[i] = top_words
        for word in top_words:
            del word_counts[word]  # Remove words to get different topics

    return topics


def calculate_top_influencers(df, num_influencers=5):
    """
    Identifies top influencers based on follower count and engagement.
    """
    if df.empty:
        return pd.DataFrame()

    df['engagement_score'] = df['retweet_count'] + df['like_count'] + df['reply_count'] + df['quote_count']

    influencer_df = df.groupby('user_name').agg({
        'user_followers_count': 'max',
        'engagement_score': 'sum'
    }).reset_index()

    influencer_df = influencer_df.sort_values(
        by=['user_followers_count', 'engagement_score'],
        ascending=False
    )

    return influencer_df.head(num_influencers)

# --------------------------------------------------------------------

# Streamlit Dashboard


st.set_page_config(page_title="Real-Time Social Media Dashboard", layout="wide")

st.title("Real-Time Social Media Insights Dashboard")

# --- Sidebar ---
st.sidebar.header("Settings")
keyword = st.sidebar.text_input("Enter keyword to track:", "Mia Khalifa")
refresh_rate = st.sidebar.slider("Refresh rate (seconds):", min_value=5, max_value=120, value=30)
num_tweets_to_fetch = st.sidebar.slider("Number of tweets to fetch per refresh:", min_value=10, max_value=100, value=50)

# --- Placeholder for Dynamic Content ---
placeholder = st.empty()

# --- Data Storage ---
data_buffer = defaultdict(list)
max_buffer_size = 1000  # Keep the last 1000 data points

# --- Real-time Data Fetching and Dashboard Update Loop ---
while True:
    df = get_tweets(keyword, count=num_tweets_to_fetch)

    if not df.empty:
        # --- Update Data Buffer ---
        for index, row in df.iterrows():
            for key in row.index:
                data_buffer[key].append(row[key])

            # --- Limit Buffer Size ---
            if len(data_buffer["created_at"]) > max_buffer_size:
                for key in data_buffer:
                    data_buffer[key].pop(0)  # Remove oldest data

        # --- Create DataFrame from Buffer ---
        buffer_df = pd.DataFrame(data_buffer)

        # --- Perform Analyses ---
        topics = analyze_topics(buffer_df, num_topics=3, num_words=5)
        top_influencers_df = calculate_top_influencers(buffer_df)
        sentiment_counts = buffer_df["sentiment"].value_counts()

        # --- Update Dashboard ---
        with placeholder.container():
            col1, col2 = st.columns(2)

            with col1:
                # --- Sentiment Over Time ---
                st.subheader("Sentiment Trend")
                fig, ax = plt.subplots()
                ax.plot(buffer_df["created_at"], buffer_df["polarity"])
                ax.set_xlabel("Time")
                ax.set_ylabel("Sentiment Polarity")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

                # --- Top Influencers ---
                st.subheader("Top Influencers")
                st.dataframe(top_influencers_df, height=200)

            with col2:
                # --- Sentiment Distribution ---
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots()
                ax.bar(sentiment_counts.index, sentiment_counts.values)
                st.pyplot(fig)

                # --- Top Topics ---
                st.subheader("Trending Topics")
                for topic_num, words in topics.items():
                    st.write(f"**Topic {topic_num + 1}:** {', '.join(words)}")

            # --- Recent Tweets (Optional) ---
            st.subheader("Recent Tweets")
            st.dataframe(buffer_df[["created_at", "text", "sentiment"]].tail(5), height=250)

    time.sleep(refresh_rate)
