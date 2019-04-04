import os
import json
from datetime import date
import string
import re
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords  
from nltk.stem import PorterStemmer
from nltk import pos_tag




# =============================================================================
#  USER-BASED FEATURES
# =============================================================================

def is_verified(tweet):
    if (tweet['user']['verified'] == True):
        user_verified = 1
    else:
        user_verified = 0
    return user_verified

def has_description(tweet):
    if tweet['user']['description'] is None:
        user_description = 0
    else:
        user_description = 1
    return user_description

def followers_count(tweet):
    num_followers = tweet['user']['followers_count']
    return num_followers

def friends_count(tweet):
    num_friends = tweet['user']['friends_count']
    return num_friends

def statuses_count(tweet):
    num_statuses = tweet['user']['statuses_count']
    return num_statuses

# =============================================================================
# PROPAGATION-BASED FEATURES
# =============================================================================

def time_span(tweet):
    charM_intM = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5,"Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
    posting_date = tweet['created_at']
    registration_date = tweet['user']['created_at']
    
    p_day = int(posting_date[8:10])
    p_month = charM_intM[posting_date[4:7]]
    p_year = int(posting_date[26:30])
    
    r_day = int(registration_date[8:10])
    r_month = charM_intM[registration_date[4:7]]
    r_year = int(registration_date[26:30])
    
    time_span = date(p_year, p_month, p_day) - date(r_year, r_month, r_day)
    
    return time_span.days

def retweet_count(tweet):
    num_retweet = tweet['retweet_count']
    return num_retweet

def favorite_count(tweet):
    num_favorite = tweet['favorite_count']
    return num_favorite



# =============================================================================
# CONTENT-BASED FEATURES
# =============================================================================

def question_mark(tweet):
    parts = tweet['text'].split('?')
    if len(parts) < 2:
        occurrence = 0
    else:
        occurrence = 1
    return occurrence

def exclamation_mark(tweet):
    parts = tweet['text'].split('!')
    if len(parts) < 2:
        occurrence = 0
    else:
        occurrence = 1
    return occurrence

def contain_url(tweet):
    if len(tweet['entities']['urls']) < 1:
        have_url = 0
    else:
        have_url = 1
    return have_url

def contain_hashtag(tweet):
    if len(tweet['entities']['hashtags']) < 1:
        have_hashtag = 0
    else:
        have_hashtag = 1
    return have_hashtag


def clean_tweets(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
 
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
 
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
 
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
 
    tweets_clean = []    
    stopwords_english = stopwords.words('english')
    # Happy Emoticons
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
        ])
     
    # Sad Emoticons
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
        ])
    
    # all emoticons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
              word not in emoticons and # remove emoticons
                word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stemmer = PorterStemmer()
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
 
    return tweets_clean

def create_vocab_list():
    topic_path = ('charliehebdo/non-rumours','charliehebdo/rumours')
    dataset = []
    for path in topic_path:
        for filename in os.listdir(path):
            single_tweet = json.load(open('%s/%s/source-tweet/%s.json' % (path, filename, filename),'r'))
            single_text = single_tweet['text']
            cleaned_tweet = clean_tweets(single_text)
            dataset.append(cleaned_tweet)       
        vocab_set = set([])
        for tweet_tokens in dataset:
            vocab_set = vocab_set | set(tweet_tokens)
            vocab_list = list(vocab_set)
    return vocab_list


def bag_of_words2_vector(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec

def create_tagger_list(vocab_list):
    tagger_list = []
    vocab_tagged = pos_tag(vocab_list)
    for tagged_word in vocab_tagged:
        word, tag = tagged_word
        tagger_list.append(tag)
        tagger_list = list(set(tagger_list))
    return tagger_list

def pos_tagging(cleaned_tweet):
    tag_list = []
    tweet_tagged = pos_tag(cleaned_tweet)
    for tagged_word in tweet_tagged:
        word, tag = tagged_word
        tag_list.append(tag)
    return tag_list




dataset = []
topic_path = ('charliehebdo/non-rumours','charliehebdo/rumours')
vocab_list = create_vocab_list()
tagger_list = create_tagger_list(vocab_list)

for path in topic_path:
    
    for filename in os.listdir(path):
        tweet = json.load(open('%s/%s/source-tweet/%s.json' % (path, filename, filename),'r'))
        feature_single_tweet = []
        
        # User-based features
        feature_single_tweet.append(is_verified(tweet))
        feature_single_tweet.append(has_description(tweet))
        feature_single_tweet.append(followers_count(tweet))
        feature_single_tweet.append(friends_count(tweet))
        feature_single_tweet.append(statuses_count(tweet))
        
        # Propogation-based features
        feature_single_tweet.append(time_span(tweet))
        feature_single_tweet.append(retweet_count(tweet))
        feature_single_tweet.append(favorite_count(tweet))
        
        #Content-based features
        feature_single_tweet.append(question_mark(tweet))
        feature_single_tweet.append(exclamation_mark(tweet))
        feature_single_tweet.append(contain_url(tweet))
        feature_single_tweet.append(contain_hashtag(tweet))
        
        # Unigram bag of word vector
        cleaned_tweet = clean_tweets(tweet['text'])
        word_vector = bag_of_words2_vector(vocab_list, cleaned_tweet)
        feature_single_tweet += word_vector
        
        # Part of speech tagging
        tagged_tweet = pos_tagging(cleaned_tweet)
        tag_vector = bag_of_words2_vector(tagger_list, tagged_tweet)
        feature_single_tweet += tag_vector
        
        # Tweet labelling
        if path == 'charliehebdo/non-rumours':
            feature_single_tweet.append(0)
        else:
            feature_single_tweet.append(1)
            
        dataset.append(feature_single_tweet)

df = pd.DataFrame(dataset)
df.to_csv('dataset.csv')

