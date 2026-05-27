"""
PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP
Task 04: Sentiment Analysis on Social Media
=============================================
Objective: Analyze and visualize sentiment patterns in social media
data to understand public opinion and attitudes toward specific topics
or brands.

Dataset: Twitter Entity Sentiment Analysis
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
(Sample simulated for offline use)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SAMPLE TWEET DATASET
# ─────────────────────────────────────────────
np.random.seed(42)

sample_tweets = {
    'Positive': [
        "I absolutely love this new product! It's amazing and works perfectly. #happy",
        "The customer service was fantastic, they solved my problem instantly! 😊",
        "Great experience with the app today, very smooth and intuitive! 👍",
        "This brand never disappoints me. Best quality products every time!",
        "Wow, just received my order and it looks even better than the photos! 🎉",
        "The new update is incredible, so many useful features added!",
        "Had an amazing experience at the store, staff was so helpful!",
        "Totally recommend this to everyone. Life-changing product! ✨",
        "Excellent quality and fast delivery. Will definitely order again!",
        "Best purchase I've made this year! Highly satisfied with everything.",
    ],
    'Negative': [
        "Terrible experience, my order never arrived and support ignored me. 😡",
        "This product is complete garbage, broke after one day. Never buying again!",
        "Worst customer service I've ever encountered. Absolute disaster!",
        "App keeps crashing and they refuse to fix it. Very frustrated!",
        "Total waste of money, quality is nowhere near what was advertised.",
        "Still waiting for my refund after 3 weeks. Completely unacceptable.",
        "The new update ruined everything. Please roll back this disaster!",
        "Charged twice for one order and customer service is useless. Awful!",
        "Product arrived damaged and company does not care. Very disappointing.",
        "Zero stars if I could. Worst brand experience ever.",
    ],
    'Neutral': [
        "Just received the package. Will update after I try it.",
        "The product is okay, nothing special but does the job.",
        "New update released. Some things changed, need time to get used to it.",
        "Customer service responded. Waiting to see how they resolve this.",
        "Ordered last week, expected delivery is tomorrow.",
        "Been using this for a month. Mixed feelings so far.",
        "The store is conveniently located near my home.",
        "Interesting new feature in the latest version.",
        "Product looks as described in the photos.",
        "Got a reply from the brand. Let's see what happens.",
    ],
    'Irrelevant': [
        "Monday mornings are hard no matter what.",
        "Just had coffee. Ready to start the day.",
        "Weekend plans anyone? 🎈",
        "Traffic today is unreal. Need a helicopter.",
        "Watching a great movie tonight. 🎬",
        "Reading a good book this evening.",
        "Can't believe how fast this year is going!",
        "Finally got some sleep last night. 😴",
        "Cooking pasta for dinner. Simple but delicious.",
        "Just went for a long walk. Refreshing!",
    ]
}

rows = []
entities = ['Apple', 'Google', 'Amazon', 'Tesla', 'Microsoft',
            'Samsung', 'Netflix', 'Meta', 'Twitter', 'Uber']

for sentiment, tweets in sample_tweets.items():
    for tweet in tweets:
        rows.append({
            'tweet_id': np.random.randint(100000, 999999),
            'entity': np.random.choice(entities),
            'sentiment': sentiment,
            'tweet_content': tweet
        })

# Expand to a larger synthetic dataset
base_df = pd.DataFrame(rows)
extra_rows = []
for _ in range(1970):
    base = base_df.sample(1).iloc[0]
    extra_rows.append({
        'tweet_id': np.random.randint(100000, 999999),
        'entity': np.random.choice(entities),
        'sentiment': base['sentiment'],
        'tweet_content': base['tweet_content']
    })

df = pd.concat([base_df, pd.DataFrame(extra_rows)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("=" * 55)
print("  PRODIGY INFOTECH — DS TASK 04: Sentiment Analysis")
print("=" * 55)
print(f"\nDataset Shape : {df.shape}")
print(f"\nSentiment Distribution:\n{df['sentiment'].value_counts()}")
print(f"\nEntity Distribution:\n{df['entity'].value_counts()}")

# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'@\w+', '', text)                  # Remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)             # Remove # from hashtags
    text = re.sub(r'[^\w\s]', '', text)               # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_tweet'] = df['tweet_content'].apply(clean_tweet)

# ─────────────────────────────────────────────
# 3. SIMPLE KEYWORD-BASED SENTIMENT SCORING
# ─────────────────────────────────────────────
positive_words = ['love', 'amazing', 'fantastic', 'great', 'excellent', 'awesome',
                  'happy', 'best', 'wonderful', 'perfect', 'recommend', 'incredible',
                  'smooth', 'helpful', 'satisfied', 'incredible', 'life-changing']
negative_words = ['terrible', 'garbage', 'worst', 'awful', 'disaster', 'frustrated',
                  'disappointing', 'useless', 'unacceptable', 'damaged', 'hate',
                  'broken', 'crashing', 'ignored', 'refuse', 'waste']

def sentiment_score(text):
    words = text.split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    return pos - neg

df['sentiment_score'] = df['cleaned_tweet'].apply(sentiment_score)

print(f"\n--- Sentiment Score Stats ---")
print(df.groupby('sentiment')['sentiment_score'].describe().round(2))

# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────
sent_colors = {
    'Positive': '#2ecc71',
    'Negative': '#e74c3c',
    'Neutral':  '#3498db',
    'Irrelevant': '#95a5a6'
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Twitter Sentiment Analysis\nProdigy Infotech DS Task 04",
             fontsize=15, fontweight='bold')

# Plot 1: Overall Sentiment Distribution (Pie)
ax1 = axes[0, 0]
sent_counts = df['sentiment'].value_counts()
ax1.pie(sent_counts, labels=sent_counts.index, autopct='%1.1f%%',
        colors=[sent_colors[s] for s in sent_counts.index],
        wedgeprops={'edgecolor':'white','linewidth':2}, startangle=90)
ax1.set_title("Overall Sentiment Distribution")

# Plot 2: Sentiment by Entity (stacked bar)
ax2 = axes[0, 1]
entity_sent = df.groupby(['entity','sentiment']).size().unstack(fill_value=0)
entity_sent_pct = entity_sent.div(entity_sent.sum(axis=1), axis=0) * 100
entity_sent_pct[[c for c in ['Positive','Neutral','Negative','Irrelevant'] if c in entity_sent_pct.columns]].plot(
    kind='bar', stacked=True, ax=ax2,
    color=[sent_colors[c] for c in ['Positive','Neutral','Negative','Irrelevant'] if c in entity_sent_pct.columns],
    edgecolor='white', linewidth=0.5)
ax2.set_title("Sentiment Distribution by Entity")
ax2.set_xlabel("Entity")
ax2.set_ylabel("Percentage (%)")
ax2.tick_params(axis='x', rotation=45)
ax2.legend(loc='upper right', fontsize=8)

# Plot 3: Sentiment Score Distribution
ax3 = axes[0, 2]
for sent in ['Positive', 'Negative', 'Neutral']:
    scores = df[df.sentiment==sent]['sentiment_score']
    ax3.hist(scores, bins=10, alpha=0.6, label=sent, color=sent_colors[sent], edgecolor='white')
ax3.set_title("Keyword Sentiment Score Distribution")
ax3.set_xlabel("Sentiment Score")
ax3.set_ylabel("Frequency")
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Top Entities by Tweet Count
ax4 = axes[1, 0]
entity_counts = df['entity'].value_counts()
ax4.bar(entity_counts.index, entity_counts.values, color='#8e44ad', edgecolor='white')
ax4.set_title("Tweet Volume by Entity")
ax4.set_xlabel("Entity")
ax4.set_ylabel("Tweet Count")
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Positive vs Negative per Entity
ax5 = axes[1, 1]
pos_neg = df[df.sentiment.isin(['Positive','Negative'])].groupby(['entity','sentiment']).size().unstack(fill_value=0)
x = np.arange(len(pos_neg))
w = 0.35
ax5.bar(x - w/2, pos_neg.get('Positive', 0), w, label='Positive', color='#2ecc71', edgecolor='white')
ax5.bar(x + w/2, pos_neg.get('Negative', 0), w, label='Negative', color='#e74c3c', edgecolor='white')
ax5.set_xticks(x)
ax5.set_xticklabels(pos_neg.index, rotation=45, ha='right')
ax5.set_title("Positive vs Negative per Entity")
ax5.set_ylabel("Count")
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Word frequency for Positive tweets
ax6 = axes[1, 2]
pos_text = ' '.join(df[df.sentiment=='Positive']['cleaned_tweet'].tolist())
stop = {'the','a','an','is','it','in','on','to','of','and','i','my','this','was','for',
        'with','just','so','at','me','you','be','are','have','they','that','not','but'}
words = [w for w in pos_text.split() if w not in stop and len(w) > 2]
top_words = Counter(words).most_common(12)
ax6.barh([w[0] for w in reversed(top_words)], [w[1] for w in reversed(top_words)],
         color='#2ecc71', edgecolor='white')
ax6.set_title("Top Words in Positive Tweets")
ax6.set_xlabel("Frequency")
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/PRODIGY_DS_04_sentiment.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved: PRODIGY_DS_04_sentiment.png")

# ─────────────────────────────────────────────
# 5. SUMMARY INSIGHTS
# ─────────────────────────────────────────────
print("\n--- Key Insights ---")
most_pos = df[df.sentiment=='Positive']['entity'].value_counts().idxmax()
most_neg = df[df.sentiment=='Negative']['entity'].value_counts().idxmax()
print(f"Entity with most Positive sentiment : {most_pos}")
print(f"Entity with most Negative sentiment : {most_neg}")
print(f"Overall Positive tweet share        : {(df.sentiment=='Positive').mean()*100:.1f}%")
print(f"Overall Negative tweet share        : {(df.sentiment=='Negative').mean()*100:.1f}%")
print("\n✅ Task 04 Complete!")
