README-Sentiment Analysis 

Below is a description of each of the files in this directory:

Data:
Reddit_Data.csv => Manually curated dataset. Contains 500 examples (400 train/100 val)
Reddit_Data_test.csv => Manually curated dataset. Contains 100 testing examples
raw_tweet_data.pkl => Raw Bitcoin Tweets used for Deep-RL evaluation
processed_tweet_data.pkl => Processed Bitcoin tweets used for Deep-RL evaluation

Helper Files:
sentiment_classifier.py => Contains all the code for the Sentiment Classifier. Contains the model, training, and evaluation functions
tokenizer.py => Contains the code the tokenizer as well as the custom torch dataset

Training Files:
train_financial_phrasebank.py => Trains the model on the Financial Phrasebank Dataset
train_reddit.py => Loads the model fine-tuned on the Financial Phrasebank Dataset and then further finetunes it on our custom Reddit Data

Parsing Files:
parse_reddit.py => Once the models are trained, you can run this file and return the sentiment to today's top 10 posts on r/CryptoCurrency. This file also showcases the model's performance and would be used for live, day-to-day trading
parse_tweets.py => Loads in the Bitcoin Tweets CSV file and processes the tweets for the Deep_RL system.


To Run the Above Files:
1. Run train_financial_phrasebank.py from command line
2. Run train_reddit.py from command line
3. Run parse_reddit.py from command line to showcase system (My reddit Developer Password and Credentials are Provided)