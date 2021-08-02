# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 16:10:30 2021

@author: pschw
"""
import numpy as np
import pandas as pd
from utils import * 

path_in = "C:/Users/pschw/Dropbox/Columbia/Courses/NLP for Python/Project/code/NLP_final_project/"

full_data = open_pickle(path_in, "full_data.pkl")

country_set = full_data.iloc[:,[1,2,3,6]]
country_set.columns = ["country","hdi_cat","hdi","sw_dict"]

def import_lex(path, file_name):
    '''
    to import text file and process to list
    '''
    file = open(path + file_name, 'r')
    the_str = file.read()
    the_lex = the_str.split()
    file.close()
    return the_lex

pw = import_lex(path_in,"positive-words.txt")
nw = import_lex(path_in,"negative-words.txt")
def gen_senti(sentence):
    import re
    #clean sentence and convert into array 
    #(I keep hypen and +- as well as ' since they appear in some words
    clean_sentence = re.sub('[^A-Za-z\+\-\']+', ' ', sentence)
    my_ar = clean_sentence.lower().split()
    #initiate count variable
    pc=0; nc=0
    #for every word in the sentence that matches with the lexicon,
    #count + 1
    for word in my_ar:
        if len(set([word]).intersection(set(pw))) == 1:
            pc = pc + 1
        elif len(set([word]).intersection(set(nw))) == 1:
            nc = nc - 1
        else:
            continue
    #calculate the score if any positive or negative words otherwise produce NA
    if (abs(pc)+abs(nc)) != 0:
        S = (pc+nc) / (abs(pc)+abs(nc))
    else:
        S = np.nan
    return S

country_set["simple_senti"] = country_set.sw_dict.apply(gen_senti)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def vader_cmpd(text):
    cmpd_score = sid.polarity_scores(text)['compound']
    return cmpd_score
sid = SentimentIntensityAnalyzer()

country_set["vader"] = country_set.sw_dict.apply(vader_cmpd)

# set(country_set.hdi_cat)
# country_set.simple_senti[country_set.hdi_cat=="Very high"].mean(skipna=True)
# my_pd = my_pd.append({"Word":word, "Frequency":word_cnt},ignore_index=True)

def cat_mean(mean_col, cat_col):
    mean_stats = pd.DataFrame()
    for word in set(cat_col):
        mean = mean_col[cat_col==word].mean(skipna=True)
        median = mean_col[cat_col==word].median(skipna=True)
        sd = mean_col[cat_col==word].std(skipna=True)
        mean_stats = mean_stats.append({"Mean":mean, "Median":median,
                                        "Sigma":sd},ignore_index=True)
    mean_stats.index = set(cat_col)
    return mean_stats

stats_simple_sent = cat_mean(country_set.simple_senti, country_set.hdi_cat)
stats_vader = cat_mean(country_set.vader, country_set.hdi_cat)

stats_simple_sent.to_csv(path_in + "stats_simple_senti.csv")
stats_vader.to_csv(path_in + "stats_vader.csv")
