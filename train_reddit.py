# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:52:41 2021

@author: Shadow
"""

import torch 
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from tokenizer import *
from sentiment_classifier import *

def main():
    
    #tokenizer_checkpoint = 'bert-base-uncased'
    #tokenizer_checkpoint = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    tokenizer_checkpoint = "facebook/muppet-roberta-base"
    model_checkpoint = "best_phrasebank_model"
    training_dataset = 'Reddit_Data.csv'
    testing_dataset = 'Reddit_Data_Test.csv'
    
    
    #label 2 correspnds to positive sentiment 
    #label 1 is neutral 
    #label 0 is negative 
    
    reddit_train_data, reddit_test_data = pd.read_csv(training_dataset), pd.read_csv(testing_dataset)
    reddit_train_data.columns, reddit_test_data.columns = ['sentence', 'label'], ['sentence', 'label'] 
    
    reddit_train_data['sentence'], reddit_test_data['sentence'] = reddit_train_data['sentence'].astype(str), reddit_test_data['sentence'].astype(str)
    reddit_train_data['label'], reddit_test_data['label'] = reddit_train_data['label'].astype(int), reddit_test_data['label'].astype(int)
    
    #shuffling the training data
    reddit_train_data = reddit_train_data.sample(frac=1)
    
    #train data has 400  examples, validation has 100
    train_data, val_data = reddit_train_data.iloc[:400, :], reddit_train_data.iloc[400:, :]
    
    
    #need to find the average length of the sequences
    total_avg = sum( map(len, list(train_data['sentence'])) ) / len(train_data['sentence'])
    print('Avg. sentence length: ', total_avg)
    
    #tokenizer hyper-param
    max_length = 192
    
    #loading tokenizer
    tokenizer = SentTokenizer(tokenizer_checkpoint, max_length, reddit_eval=True)
    
    #tokenizing the data
    train_dataset = tokenizer.encode_data(train_data)
    val_dataset = tokenizer.encode_data(val_data)
    test_dataset = tokenizer.encode_data(reddit_test_data)
    
    #loading the model
    model = Lit_SequenceClassification(model_checkpoint, save_fp = 'reddit_sentiment_classifier')
    
    preds, ground_truths = model_testing(model, test_dataset)
    
    cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)
    
    print()
    print('Reddit Data Base Report: ')
    print(cr)
    
    model = train_LitModel(model, train_dataset, val_dataset, epochs=15, batch_size=8, patience = 3, num_gpu=1)
        
     
    preds, ground_truths = model_testing(model, test_dataset)
    
    cr = classification_report(y_true=ground_truths, y_pred = preds, output_dict = False)
    
    print()
    print('Reddit Data Finetune Report: ')
    print(cr)
    
    #need to save the model again for reasons adressed here: https://github.com/huggingface/transformers/issues/8272
    model.save_model()
    

   


if __name__ == "__main__":
    main()
    