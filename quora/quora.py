#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from embedding import GoogleNegativeNewsEmbeddingWrapper
import os
import re
import pudb
# Any results you write to the current directory are saved as output.


# Perform some basic analysis on the prior distributions of positives and negatives. 


def print_class_distributions(df): 
    """ Calculates the number of samples as well as the class imbalance. 
    
    Args: 
        df (pd.DataFrame) : a DataFrame with one column having the header "target" with a int datatype. 
    
    Returns: 
        None
    """
    count - len(df)
    positives = len(df[df.target == 1])
    print(f"{(positives/count )*100}% of the {count} samples are positives")
    
def plot_classed_sample_length(df, bins=30): 
    """ Plots the histograms of the question text lengths for positive and negative samples on the same axis
    
    Args: 
        df (pd.DataFrame) : A data frame that has a column labelled "target" which is a binomial value. 
        
    Returns: 
        None
    """
    df["question_length"] = df["question_text"].apply(lambda x : len(x.split(" ")))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos_bins = max(df[df.target == 1].question_length) - min(df[df.target == 1].question_length)
    neg_bins = max(df[df.target == 0].question_length) - min(df[df.target == 0].question_length)
    _ = ax.hist(df[df.target == 1].question_length, bins=pos_bins, density=True, label="Positives")
    _ = ax.hist(df[df.target == 0].question_length, bins=neg_bins, density=True, color="red", alpha=0.3, label="Negative")
    ax.legend()
    


# In[3]:


# df = process_quora_csv("../input/train.csv")
#plot_classed_sample_length(df)


# The above prior distributions show a strong distribution over the number of words in the question. Positive (in the sense that it is insincere) samples tend to be longer whilst sincere questions are shorter in nature. 
# 
# Using the Google News negative vector embedding, clean the data, tokenise and embed the training data. Save the results in chunks. 

# In[ ]:


# embed_shape = (50, 300)
def pad_vector(self, x, length): 
        """Either pads the vector to a size of self.pad or removes the excess.
        
        Args: 
            x (np.array) : row vector 
            length (int) : Length to pad vector to. 
            
        Returns: 
            (np.array)   : A row vector of length 50. 
        """
        if len(x) < length: 
            return np.concatenate((x, np.zeros((x.shape[0], length - len(x)))), axis=1)
        else: 
            return x[0:length, 0]


def labelled_data_generator(embedder, chunk_size, input_file): 
    def generator(): 
        for df in pd.read_csv(input_file, chunksize=chunk_size):
            for _, row  in df.iterrows():
                question = embedder.tokenise_and_embed_text(row["question_text"])
                yield (tf.convert_to_tensor(question), tf.convert_to_tensor(row["target"]))
    
    return generator



embedder = GoogleNegativeNewsEmbeddingWrapper("../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin", word_length = 300, question_length =50)
chunk_size = 1000
input_file = "../input/train.csv"

list(labelled_data_generator(embedder, chunk_size, input_file)())
