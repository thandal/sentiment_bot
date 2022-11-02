#!/usr/bin/python3

import numpy as np
import pickle
from scipy.special import softmax
from time import sleep, time
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import tweepy

from SECRETS import *


SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
sentiment_config = AutoConfig.from_pretrained(SENTIMENT_MODEL)


client = tweepy.Client(
  bearer_token=BEARER_TOKEN,
  access_token=ACCESS_TOKEN,
  access_token_secret=ACCESS_TOKEN_SECRET,
  consumer_key=API_KEY,
  consumer_secret=API_KEY_SECRET,
  return_type=dict)


def preprocess_tweet(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text).replace("\n", " ").strip()

def score_sentiment(text):
    preprocessed_text = preprocess_tweet(text)
    encoded_input = sentiment_tokenizer(preprocessed_text, return_tensors='pt')
    output = sentiment_model(**encoded_input)
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    return scores

def average_sentiment_scores(tweets):
    avg_scores = np.zeros(3)
    for tweet in tweets['data']:
        text = tweet['text']
        preprocessed_text = preprocess_tweet(text)
        avg_scores += score_sentiment(preprocessed_text)
    avg_scores = avg_scores / len(tweets['data'])
    return avg_scores


try: newest_tweet_id = pickle.load(open('newest_tweet_id.pkl', 'rb'))
except: newest_tweet_id = None

last_check_times = {}
while 1:
    try:
        new_tweets = client.get_users_mentions(BIBLIEST_USER_ID, since_id=newest_tweet_id)
    except:
        # Hope this is a transient error, wait and try again.
        sleep(30)
        continue
    print(new_tweets)
    if new_tweets['meta']['result_count'] == 0:
        sleep(10)
        continue
    newest_tweet_id = new_tweets['meta']['newest_id']
    pickle.dump(newest_tweet_id, open('newest_tweet_id.pkl', 'wb'))
    for tweet in new_tweets['data']:
        text = tweet['text']
        if ' check ' in text.lower():
            target = text.split(' ')[-1]
            if target.startswith('@'):
                print('checking', target)
                if target in last_check_times and (time() - last_check_times[target]) < 3600:
                    response = client.create_tweet(text=f"boop BLIK...\nRecently checked {target}, please wait a little", in_reply_to_tweet_id=tweet['id']) 
                    continue
                last_check_times[target] = time()
                try:
                    response = client.get_user(username=target[1:])
                    target_id = response['data']['id']
                    target_name = response['data']['name']
                    target_tweets = client.get_users_tweets(target_id, exclude='retweets', max_results=100) # limited to 100??
                    scores = average_sentiment_scores(target_tweets)
                    response = client.create_tweet(text=f"boop beep...\n{target_name}({target})'s recent tweets are\n{round(scores[0]*100)}% NEGATIVE 😡\n{round(scores[1]*100)}% NEUTRAL 😐\n{round(scores[2]*100)}% POSITIVE 😀", in_reply_to_tweet_id=tweet['id']) 
                except:
                    response = client.create_tweet(text=f"boop BONK...\nHad an error processing {target}", in_reply_to_tweet_id=tweet['id']) 
