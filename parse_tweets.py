# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:11:56 2021

@author: Shadow
"""

import pandas as pd
import numpy as np

from sentiment_classifier import *
from tokenizer import *

from sklearn.metrics import classification_report
from collections import Counter
from tqdm import tqdm

def parse_tweets():
    
    test_data = pd.read_csv('Bitcoin_tweets.csv')
    
    test_data = test_data.dropna()
    
    num_follower = test_data['user_followers'].tolist()
    avg_followers = np.mean(num_follower)
    std_followers = np.std(num_follower)
    print('Average # of Followers: ', avg_followers)
    print('Standard Deviation # Followers: ', np.std(num_follower))
    
    threshold = round(avg_followers+std_followers)
    
    text = []
    date = []
    for i in tqdm(range(len(test_data))):
        
        example = test_data.iloc[i,:].tolist()
        
        followers = example[4]
        if followers > threshold:
            text.append(example[9])
            date.append(example[8])
            
    new_data = pd.DataFrame(data={'sentence':text, 'date':date})
    
    new_data.to_pickle('raw_tweet_data.pkl')
    
def get_tweet_sentiment():
    test_data = pd.read_pickle('raw_tweet_data.pkl')
    print()
    print('Number of examples in raw tweet data: ', len(test_data))
    
    text = test_data['sentence'].tolist()
    
    
    tokenizer_checkpoint = "facebook/muppet-roberta-base"
    model_checkpoint = 'reddit_sentiment_classifier'
    sanity_check_dataset = 'Reddit_Data_Test.csv'
    
    
    text = pd.DataFrame(text)
    text.columns = ['sentence']
    text['sentence'] = text['sentence'].astype(str)
    
    sanity_check_data = pd.read_csv(sanity_check_dataset)
    sanity_check_data.columns = ['sentence', 'label']
    
    sanity_check_data['sentence'] = sanity_check_data['sentence'].astype(str)
    sanity_check_data['label'] =  sanity_check_data['label'].astype(int)
    
    #had to use smaller maximum length because of memory constraints
    max_length = 96
    tokenizer = SentTokenizer(tokenizer_checkpoint, max_length, reddit_eval=True)
    
    sanity_check_dataset = tokenizer.encode_data(sanity_check_data)
    test_dataset = tokenizer.encode_data(text)
    
    model = Lit_SequenceClassification('reddit_sentiment_classifier')
    
    sanity_check_preds, ground_truths = model_testing(model, sanity_check_dataset)
    
    cr = classification_report(y_true=ground_truths, y_pred = sanity_check_preds, output_dict = False)
    
    print()
    print('Sanity Check on Reddit Test Data: ')
    print(cr)
    
    test_preds = model_prediction(model, test_dataset)
    
    processed_twitter_data = pd.DataFrame(data = {'sentiment': test_preds, 'date': test_data['date'].tolist()})
    print()
    print('Number of examples in processed tweet data: ', len(processed_twitter_data))
    print()
    print(processed_twitter_data.head())
    processed_twitter_data.to_pickle('processed_tweets.pkl')
    
if __name__ == "__main__":
    
    #we would set to True  if you had the 1GB Bitcoin Tweet data in your local directory
    use_bitcoin_csv = False
    if use_bitcoin_csv == True:
        parse_tweets()
        get_tweet_sentiment()
    else:
        get_tweet_sentiment()


