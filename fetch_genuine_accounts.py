from dotenv import load_dotenv
from datetime import datetime
import os
import pandas as pd
import praw
import pickle

# Load environment variables from .env file
load_dotenv()

# Reddit API credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
USER_AGENT = os.getenv('USER_AGENT')

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Load the dataset
df = pd.read_csv('/Users/prachisinha/Minor-Project/suspicious_accounts_dataset.csv')

# Summary of the dataset
summary = {
    'total_posts': len(df),
    'unique_users': df['username'].nunique(),
    'unique_subreddits': df['subreddit'].nunique(),
    'total_images': df[df['is_image'] == True].shape[0],
    'total_non_images': df[df['is_image'] == False].shape[0]
}

# Dictionary of the count of number of posts for each subreddit
subreddit_post_count = df['subreddit'].value_counts().to_dict()

# Dictionary of the count of number of posts for each user
user_post_count = df['username'].value_counts().to_dict()

# Print the summary
for key, value in summary.items():
    print(f"{key}: {value}")

# Print the dictionaries
print("\nSubreddit Post Count:")
print(subreddit_post_count)

print("\nUser Post Count:")
print(user_post_count)

'''
# Store the dictionaries for later use
with open('/Users/prachisinha/Minor-Project/subreddit_post_count.pkl', 'wb') as f:
    pickle.dump(subreddit_post_count, f)

with open('/Users/prachisinha/Minor-Project/user_post_count.pkl', 'wb') as f:
    pickle.dump(user_post_count, f)
'''

# Fetch posts from Reddit
genuine_posts = []

for subreddit, count in subreddit_post_count.items():
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        for submission in subreddit_obj.new(limit=count):
            genuine_posts.append({
                'username': submission.author.name if submission.author else 'deleted',
                'title': submission.title,
                'text': submission.selftext,
                'created_utc': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'url': submission.url,
                'subreddit': subreddit,
                'is_image': submission.url.endswith(('jpg', 'jpeg', 'png', 'gif'))
            })
    except Exception as e: # Handle exceptions
        print(f"Error fetching posts from subreddit {subreddit}: {e}")

# Convert the list of genuine posts to a DataFrame
genuine_df = pd.DataFrame(genuine_posts)

# Save the genuine posts to a CSV file
genuine_df.to_csv('/Users/prachisinha/Minor-Project/genuine_accounts_dataset.csv', index=False)