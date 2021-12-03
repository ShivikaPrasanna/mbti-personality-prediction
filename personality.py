'''
Developed by Shivika Prasanna on 11/05/2021.
Last updated on 11/27/2021.
Predicts personality based on tweets.
Working code.
To run (using conda env): $ conda activate personality
                        $ python3 Code/personality.py -d <dataset> 
Dataset: /Users/shivikaprasanna/Desktop/Mizzou_Academics/Semesters/Sem4/CMP_SC\ 8725-01/Project/Data/mbti_1.csv
'''
import os, sys, csv
import string, re
import argparse

import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns
import string

import nltk
from nltk.classify import NaiveBayesClassifier
nltk.download('stopwords')
nltk.download('punkt')

print("Python version: ", sys.version)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="dataset in .csv format")
ap.add_argument("-t", "--test", required=True, help="test path of person")
ap.add_argument("-n", "--name", required=True, help="name of the person")
args = vars(ap.parse_args())

dataset_path = str(args['dataset'])
test_file_path = str(args['test'])
name = str(args['name'])

df = pd.read_csv(dataset_path)

# Dataset understanding
print("Dataset preview: ", df.head(10))
print("Size of the dataset: ", df.shape)

print("Preview of posts: ", df.iloc[0,1].split('|||'))
print("Length of posts: ", len(df.iloc[1,1].split('|||')))

types = np.unique(df.type.values)
print("List of {} unique types: {}".format(len(types), types))

total_types = df.groupby(['type']).count()
print("Total count for each personality: ", total_types)

# Visualizing the counts of the types
count_types = df['type'].value_counts()
plt.figure(figsize = (12,4))
plt.bar(np.array(total_types.index), height = total_types['posts'],)
plt.xlabel('Types', size = 12)
plt.ylabel('Number of posts', size = 12)
plt.title('Total posts for each personality type')
plt.savefig('types_counts.png')

# Bag-of-words model creation: Split posts on personality and add to a Pandas Series
all_posts = pd.DataFrame()
for i in types:
    tmp1 = df[df['type'] == i]['posts']
    tmp2 = []
    for j in tmp1:
        tmp2 += j.split('|||')
    tmp3 = pd.Series(tmp2)
    all_posts[i] = tmp3
print("Preview of posts grouped by the personality type: ", all_posts.tail())

# Tokenize words
stop_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
def filtered_bag_of_words(words):
    words = nltk.word_tokenize(words)
    return {
        word:1 for word in words \
        if not word in stop_words
    }

print("Bag of words for type INTJ: ", filtered_bag_of_words(all_posts['INTJ'].iloc[1]))

# Array of features
features = []
for i in types:
    tmp1 = all_posts[i]
    tmp1 = tmp1.dropna()
    features += [[(filtered_bag_of_words(j), i) for j in tmp1]]

# Splitting posts proportionally and creating a train-test set of 80:20 split
split_data = []
for i in range(16):
    split_data += [len(features[i]) * 0.8]
split_data = np.array(split_data, dtype = int)

print("Count of the split data: ", split_data)

train=[]
for i in range(16):
    train += features[i][:split_data[i]]

# Testing NB model
nb_classifier = NaiveBayesClassifier.train(train)

# Naive Bayes model accuracy
print("Naive Bayes Classifier accuracy on train set is: ", nltk.classify.util.accuracy(nb_classifier, train)*100)

test=[]
for i in range(16):
    test += features[i][split_data[i]:]

print("Naive Bayes Classifier accuracy on test set is: ", nltk.classify.util.accuracy(nb_classifier, test)*100)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("The efficiency of the model is {}, which is quite low, when we use the dataset segregated on personalities only. Instead of selecting all 16 types of personalities, a better approach would be to split the data on the 4 axes that builds the Myers Briggs personality. In other words, the 4 axes are Introversion (I) – Extroversion (E), Intuition (N) – Sensing (S), Thinking (T) – Feeling (F), Judging (J) – Perceiving (P). A classifier model would be built for each of the axes so we can predict the personality by combining 1 type from each of the axes.".format(nltk.classify.util.accuracy(nb_classifier, test)*100))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Changing approach to create models to predict 1 personality from each axis
# Starting with Introvert-Extrovert (IE)
features = []
for i in types:
    tmp1 = all_posts[i]
    tmp1 = tmp1.dropna()
    if('I' in i):
        features += [[(filtered_bag_of_words(j), 'introvert') for j in tmp1]]
    if('E' in i):
        features += [[(filtered_bag_of_words(j), 'extrovert') for j in tmp1]]

train = []
for i in range(16):
    train += features[i][:split_data[i]]

introvert_extrovert_model = NaiveBayesClassifier.train(train)

test = []
for i in range(16):
    test += features[i][split_data[i]:]

train_ie_model_acc = nltk.classify.util.accuracy(introvert_extrovert_model, train)*100
test_ie_model_acc = nltk.classify.util.accuracy(introvert_extrovert_model, test)*100
print("IE: Train accuracy = {}, \nTest accuracy = {} ".format(train_ie_model_acc,  test_ie_model_acc))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Second is Intuition-Sensing (NS)
features = []
for i in types:
    tmp1 = all_posts[i]
    tmp1 = tmp1.dropna()
    if('N' in i):
        features += [[(filtered_bag_of_words(j), 'intuition') for j in tmp1]]
    if('S' in i):
        features += [[(filtered_bag_of_words(j), 'sensing') for j in tmp1]]

train = []
for i in range(16):
    train += features[i][:split_data[i]]

intuition_sensing_model = NaiveBayesClassifier.train(train)

test = []
for i in range(16):
    test += features[i][split_data[i]:]

train_ns_model_acc = nltk.classify.util.accuracy(intuition_sensing_model, train)*100
test_ns_model_acc = nltk.classify.util.accuracy(intuition_sensing_model, test)*100
print("NS: Train accuracy = {}, \nTest accuracy = {} ".format(train_ns_model_acc, test_ns_model_acc))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Third is Thinking–Feeling (TF)
features = []
for i in types:
    tmp1 = all_posts[i]
    tmp1 = tmp1.dropna()
    if('T' in i):
        features += [[(filtered_bag_of_words(j), 'thinking') for j in tmp1]]
    if('F' in i):
        features += [[(filtered_bag_of_words(j), 'feeling') for j in tmp1]]

train = []
for i in range(16):
    train += features[i][:split_data[i]]

thinking_feeling_model = NaiveBayesClassifier.train(train)

test = []
for i in range(16):
    test += features[i][split_data[i]:]

train_tf_model_acc = nltk.classify.util.accuracy(thinking_feeling_model, train)*100
test_tf_model_acc = nltk.classify.util.accuracy(thinking_feeling_model, test)*100
print("TF: Train accuracy = {}, \nTest accuracy = {} ".format(train_tf_model_acc, test_tf_model_acc))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Fourth is Judging–Perceiving (JP)
features = []
for i in types:
    tmp1 = all_posts[i]
    tmp1 = tmp1.dropna()
    if('J' in i):
        features += [[(filtered_bag_of_words(j), 'judging') for j in tmp1]]
    if('P' in i):
        features += [[(filtered_bag_of_words(j), 'perceiving') for j in tmp1]]

train = []
for i in range(16):
    train += features[i][:split_data[i]]

judging_perceiving_model = NaiveBayesClassifier.train(train)

test = []
for i in range(16):
    test += features[i][split_data[i]:]

train_jp_model_acc = nltk.classify.util.accuracy(judging_perceiving_model, train)*100
test_jp_model_acc = nltk.classify.util.accuracy(judging_perceiving_model, test)*100
print("JP: Train accuracy = {}, \nTest accuracy = {} ".format(train_jp_model_acc, test_jp_model_acc))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

temp = {'train' : [train_ie_model_acc, train_ns_model_acc, train_tf_model_acc, train_jp_model_acc], 'test' : [test_ie_model_acc, test_ns_model_acc, test_tf_model_acc, test_jp_model_acc]}
test_train_acc_results = pd.DataFrame.from_dict(temp, orient='index', columns=['Introvert - Extrovert', 'Intuition - Sensing', 'Thinking - Feeling', 'Judging - Perceiving'])
print("Train vs Test accuracies: ", test_train_acc_results)

labels = np.array(test_train_acc_results.columns)

training = test_train_acc_results.loc['train']
ind = np.arange(4)
width = 0.4
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, training, width, color='red')

testing = test_train_acc_results.loc['test']
rects2 = ax.bar(ind+width, testing, width, color='yellow')

fig.set_size_inches(12, 6)
fig.savefig('test_train_acc_results.png', dpi=200)

ax.set_xlabel('Model Classifying Trait', size = 18)
ax.set_ylabel('Accuracy Percent (%)', size = 18)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(labels)
ax.legend((rects1[0], rects2[0]), ('Tested on a known dataframe', 'Tested on an unknown dataframe'))

# Creating a function to use the above models to predict personalities
def MBTI(input):
    tokens = filtered_bag_of_words(input)
    i_e = introvert_extrovert_model.classify(tokens)
    i_s = intuition_sensing_model.classify(tokens)
    t_f = thinking_feeling_model.classify(tokens)
    j_p = judging_perceiving_model.classify(tokens)
    
    p_axis = ''
    
    if (i_e == 'introvert'):
        p_axis += 'I'
    if (i_e == 'extrovert'):
        p_axis += 'E'
    if (i_s == 'intuition'):
        p_axis += 'N'
    if (i_s == 'sensing'):
        p_axis += 'S'
    if (t_f == 'thinking'):
        p_axis += 'T'
    if (t_f == 'feeling'):
        p_axis += 'F'
    if (j_p == 'judging'):
        p_axis += 'J'
    if (j_p == 'perceiving'):
        p_axis += 'P'
    return (p_axis)

def predictMBTIPersonality(input, name):
    traits=[]
    p_ax = []

    axis1 = pd.DataFrame([0,0,0,0], ['I', 'N', 'T', 'J'], ['count'])
    axis2 = pd.DataFrame([0,0,0,0], ['E', 'S', 'F', 'P'], ['count'])

    for i in input:
        p_ax += [MBTI(i)]
    
    for i in p_ax:
        for j in ['I', 'N', 'T', 'J']:
            if (j in i):
                axis1.loc[j] += 1
        for j in ['E', 'S', 'F', 'P']:
            if (j in i):
                axis2.loc[j] += 1
    
    axis1 = axis1.T
    axis1 = axis1 * 100/len(input)
    axis2 = axis2.T
    axis2 = axis2 * 100/len(input)

    predictedTrait = ''

    for i, j in zip(axis1, axis2):
        trait_char = max(axis1[i][0], axis2[j][0])
        if (axis1[i][0] == trait_char):
            predictedTrait += i
        if (axis2[j][0] == trait_char):
            predictedTrait += j

    traits += [predictedTrait]

    labels = np.array(test_train_acc_results.columns)

    intj = axis1.loc['count']
    index = np.arange(4)
    width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(index, intj, width, color='#0d729c')
    
    esfp = axis2.loc['count']
    rects2 = ax.bar(index, esfp, width, color='#f68657')

    fig.set_size_inches(10, 7)

    ax.set_xlabel('MBTI Trait', size = 18)
    ax.set_ylabel('Trait percent(%)', size = 18)
    ax.set_xticks(index + width/2)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 105, step= 10))
    ax.set_title(name+'\'s personality is ' + predictedTrait, size = 20)
    plt.grid(True)

    fig.savefig(name+'_results.png', dpi=200)

    return (traits)

# Testing the above functions with tweets of famous personalities scraped from Twitter using Twarc
if test_file_path.endswith('.jsonl'):
    with open(test_file_path, 'r') as json_file:
        json_list = list(json_file)
        final_tweets = []

        for json_str in json_list:
            res = json.loads(json_str)
            tweet = res['full_text']
            final_tweets.append(tweet)

    predicted_trait = predictMBTIPersonality(final_tweets, name)

if test_file_path.endswith('.csv'):
    df = pd.read_csv(test_file_path)
    if 'tweet' in df.columns:
        predicted_trait = predictMBTIPersonality(df['tweet'], name)
    elif 'content' in df.columns:
        predicted_trait = predictMBTIPersonality(df['content'], name)
    elif 'text' in df.columns:
        predicted_trait = predictMBTIPersonality(df['text'], name)
if test_file_path.endswith('.txt'):
    writings = open(test_file_path, encoding='unicode_escape')
    text = writings.readlines()
    sentence = text[0].split('|||')
    predicted_trait = predictMBTIPersonality(sentence, name)

print('Trait is: ', predicted_trait)





