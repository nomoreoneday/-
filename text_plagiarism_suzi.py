import re
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
from pprint import pprint
import os

#加载停用词
with open('chinese_stopwords.txt','r',encoding = 'utf-8') as file:
	stopwords = [i[:-1] for i in file.readlines()]

#数据加载
news = pd.read_csv('sqlResults.csv',encoding ='gb18030')
print(news.shape)
print(news.head(5))