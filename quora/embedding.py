#!/usr/bin/env python
# coding: utf-8


import numpy as np
import gensim.models.keyedvectors as word2vec
import re


class EmbeddingWrapper(object): 
    
    def __init__(self, embedding_file, word_length=300, question_length=50): 
        self.word_length = word_length
        self.question_length = question_length
        self.embedding = self.open_embedding_file(embedding_file)
        
    
    def clean_text(self, text): 
        """ Cleans a given text so that it is optimal for the embedding type

        Args: 
            text (str) : An unprocessed text 

        Returns: 
                (str) : Text cleaned optimally for the embedding
        """
        return ''

    def tokenise_and_embed_text(self, text, clean_func=None): 
        """Tokenises and embeds the given text, performing the inputted cleaning if passed in.

        Args: 
            text (str)                 : An unprocessed text
            clean_func(func(str)->str) : A function that cleans the inputted text. If None, the inputted text is not cleaned. 

        Returns: 
            (np.array) : an ndarray with embedding column for each word in the text. The shape of this is the number of words in the text by the embedding size. 
        """
        if clean_func is not None: 
            text = clean_func(text)

        vectors = []
        for word in text.split(): 
            try: 
                vectors.append(self.embedding[word][:])
            except KeyError as e:
                #TODO use n-gram model as well for unseen words. 
                vectors.append(np.ones((self.word_length)))

        for p in range(self.question_length-len(text.split())): 
            vectors.append(np.zeros((self.word_length)))
        return np.stack(vectors)

    def embed_dataset(self, dataset): 
        """ Embeds the dataset with the given embedding. 

        Args: 
            dataset (pd.DataFrame) : 

        Returns: 
            (np.array) : an ndarray of embeddings for each data point in the dataset. The order is preserved. 
        """
        return np.stack([self.tokenise_and_embed_text(point, clean_func = self.clean_text) for point in dataset])
  
class GoogleNegativeNewsEmbeddingWrapper(EmbeddingWrapper): 

    def open_embedding_file(self, embedding_file): 
        """ Opens and returns the embedding file

        Returns: 
            A embedding dictionary in the form: 
        """
        word2vecDict = word2vec.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        embeddings_index = dict()
        for word in word2vecDict.wv.vocab:
            embeddings_index[word] = word2vecDict.word_vec(word)

        return embeddings_index


    def clean_text(self, text): 
        """ Cleans a given text so that it is optimal for the embedding type

        Args: 
            text (str) : An unprocessed text 

        Returns: 
                (str) : Text cleaned optimally for the embedding
        """
        return self.fix_prepositions(self.fix_numbers(self.clean_punctuation(text)))


    def clean_punctuation(self, x): 
        """ Removes problematic punctation from the text so that more words will be in the embedding. 
        Args: 
            x (str) : Any text string with or without punctuation. 

        Returns: 
                (str) : Text with problematic punctation removed.
        """
        x = x.replace("&", " & ")
        for punct in "/-'":
            x = x.replace(punct, ' ')
        for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
            x = x.replace(punct, '')
        return x

    def fix_numbers(self, x): 
        """ Fixes problems with numbers in the text being hashed out.

        Args: 
            x (str) : Any text string. 

        Returns: 
                (str) : Text with hashed number values removed. 
        """
        for i in range(1,6):
            x = re.sub(f'[0-9]{{{i},}}', i*'#', x)
        return x

    def fix_prepositions(self, x): 
        """ Remove prepositions that won't be in the embedding. 

        Args: 
            x (str) : Any text string. 

        Returns: 
            (str)   : A string without prepositions

        """
        for prep in ["and", "to", "a", "of"]: 
            x = x.replace(prep, "")
        return x


