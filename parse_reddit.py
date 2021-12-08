# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:41:37 2021

@author: Shadow
"""

import praw 
import pandas as pd

from sentiment_classifier import *
from tokenizer import *

from sklearn.metrics import classification_report
from collections import Counter

def main(subreddit_id = 'CryptoCurrency'):
    reddit = praw.Reddit(client_id = "gM2kmUwONgKK1Hw77Jentw", #peronal use script
                        client_secret = "4WQSyS6-07wzeEbNv5oy7AO_7KZIOw", #secret token
                        usernme = "Skunkathor13", #profile username
                        password = "CPEFinal2021", #profile password
                        user_agent = "USERAGENT")
    
    subreddit_id = 'CryptoCurrency'
    #subreddit_id = 'etherium'
    
    
    subreddit = reddit.subreddit(subreddit_id)
    top_posts = subreddit.top(limit=10)
    text = []
    
    print()
    print("Let's take a look at what today's top posts look like:")
    for i,submission in enumerate(top_posts):
        
        print()
        print('Submission #: ', i+1)
        print('Title: {}, ups: {}, downs: {}, upvote ratio: {}, score: {}'.format(submission.title, submission.ups, submission.downs, submission.upvote_ratio, submission.score))
        text.append(submission.title)
    
    #tokenizer_checkpoint = 'bert-base-uncased'
    #tokenizer_checkpoint = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer_checkpoint = "facebook/muppet-roberta-base"
    
    model_checkpoint = 'reddit_sentiment_classifier'
    sanity_check_dataset = 'Reddit_Data_Test.csv'
    
    
    text = pd.DataFrame(text)
    text.columns = ['sentence']
    text['sentence'] = text['sentence'].astype(str)
    
    #the snity checkpoint is just to confirm that we are starting from a good checkpoint before parsing the reddit data
    sanity_check_data = pd.read_csv(sanity_check_dataset)
    sanity_check_data.columns = ['sentence', 'label']
    
    sanity_check_data['sentence'] = sanity_check_data['sentence'].astype(str)
    sanity_check_data['label'] =  sanity_check_data['label'].astype(int)
    
    
    max_length = 192
    tokenizer = SentTokenizer(tokenizer_checkpoint, max_length, reddit_eval=True)
    
    sanity_check_dataset = tokenizer.encode_data(sanity_check_data)
    test_dataset = tokenizer.encode_data(text)
    
    model = Lit_SequenceClassification('reddit_sentiment_classifier')
    
    sanity_check_preds, ground_truths = model_testing(model, sanity_check_dataset)
    
    cr = classification_report(y_true=ground_truths, y_pred = sanity_check_preds, output_dict = False)
    
    print()
    print("The sanity check is just to confirm that we are starting from a good checkpoint before parsing the reddit data")
    print('Sanity Check on Reddit Test Data: ')
    print(cr)
    
    test_preds = model_prediction(model, test_dataset)
    
    prediction_counter = Counter(test_preds)
    majority_vote = prediction_counter.most_common(1)[0][0]
    
    id2tag = {0:'negative', 1:'neutral', 2:'positive'}
    
    print('Prediction Counter: ', prediction_counter)
    print('Majority Vote: ', id2tag[majority_vote])
    
    print()
    print("Now let's take a look at the Reddit posts aswell as the predicted sentiment:")
    for pred, title in zip(test_preds, text['sentence'].tolist()):
        print()
        print('Predicted Sentiment: ', id2tag[pred])
        print('Original Text: ', title)
        

if __name__ == "__main__":
    main()

