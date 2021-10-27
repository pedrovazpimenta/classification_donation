#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
__author__ = "Pedro Vaz Pimenta"
#-------------------------------------------------------------------------------
# in this script, we take a large data frame file and label the rows according
# to a model of semantic proximation of words, choosing tags from another table;
# for it to run, it will also need a word vector file. The output is the same
# data frame from the start joined with rank columns containing the tags
# 
# for more details and resources, check the notebook file
#-------------------------------------------------------------------------------
import pandas as pd             
import re
import numpy as np
import nltk
from spellchecker import SpellChecker
from nltk.corpus import stopwords
#-------------------------------------------------------------------------------
nltk.download('stopwords') 
#-------------------------------------------------------------------------------
# relevant urls; the first two are too big for the repository, so you'll need
# to download them to run the script, only the tags.csv file is provided
url_1 = 'donations.csv'     #https://milliondollarlist.org/
url_2 = 'glove.6B.300d.txt' #https://nlp.stanford.edu/projects/glove/
url_3 = 'tags.csv'          #tags .csv file
url_4 = 'outputtable.csv'   #output .csv file
#-------------------------------------------------------------------------------
# functions

def replace_nan(s):
    # NaN or str -> str, '' if NaN
    if str(s) == 'nan':
        return ''
    else:
        return s
    
def replace_disaster(s):
    # 'yes' -> 'disaster'
    if s == 'yes':
        return 'disaster'
    else:
        return s    

def str_to_set_of_words(s):
    # str -> set
    s = s.lower()                               #we need lower case only
    s = re.sub(r'[0-9]', ' ', s)                #replace any number by ' '
    s = re.sub(r'\W', ' ', s)                   #replace useless symbols by ' '
    s = s.split()                               #split the string
    ret = [word for word in s if len(word) >2]  #get rid of small words
    return set(ret)                             #returns a set, so no repetition

def remove_stop_words(set_of_words):
    # set -> set
    # remove stopwords from the set
    return set_of_words.difference(stopwords.words('English'))

def correct_text(set_of_words, correction):
    ''' set , dict -> set
        correct words from set according to dict
        we also remove new stopwords that may have appeared
    '''
    set_of_words = remove_stop_words(set_of_words)
    iterating = set_of_words.copy()
    for word in iterating:
        if word in correction:
            corrected_word = correction[word]
            corrected_word = corrected_word.split() 
            corrected_word = set(corrected_word)
            set_of_words.union(corrected_word)
            set_of_words.remove(word)
    return set_of_words 

def correction_tags(set_of_words):
    # set -> set , small manual corrections + remove stopwords                  
    set_of_words = remove_stop_words(set_of_words)   
    iterating = set_of_words.copy()
    for word in iterating:                           
        if word == 'imrpoving':
            set_of_words.add('improving')
            set_of_words.remove('imrpoving')
        elif word == 'informationon':
            set_of_words.add('information')
            set_of_words.remove('informationon')
        elif word == 'meatâ':
            set_of_words.add('meat')
            set_of_words.remove('meatâ')
        elif word == 'valuesâ':
            set_of_words.add('values')
            set_of_words.remove('valuesâ')
    return set_of_words

def cos_angle(vec_1 , vec_2):
    # np.array(float) , np.array(float) -> float , cos(angle between vectors)
    ve_1 = vec_1/np.linalg.norm(vec_1)
    ve_2 = vec_2/np.linalg.norm(vec_2)
    return np.dot(ve_1 ,ve_2) 

def set_distance(set_1 , set_2 , word_distances):
    ''' set , set -> float
        given two sets, calculate the "distance" between every combination of
        words possible, square them keeping the signal and return the mean of
        the values    
    '''
    dist = 0
    for word_1 in set_1:
        for word_2 in set_2:
            d = word_distances[word_1][word_2]
            if d == 0:
                pass
            else:
                dist += (d**3)/d
    n = len(set_1)*len(set_2)
    if n == 0:
        return 0
    else:
        return dist/n 
    
def dist_ranks(set_0 , tags , tags_dict , word_distances):
    ''' set , pd.DataFrame, dict -> list
        get a set of words, a data frame containing tags and the dictionary that
        converts tags into a set of words, returning the best 5 matches. One 
        could change the parameter size for a bigger or smaller rank    
    '''
    size = 5
    distances_tags = {}
    distances = []
    for tag in tags['tags']:
        d = set_distance(set_0 , tags_dict[tag] , word_distances)
        distances_tags[tag] = d
        distances += [d]
    distances.sort(reverse = True)
    ret = []
    done = set({})
    for i in range(size):
        for d in distances_tags:
            if distances_tags[d] == distances[i]:
                if d in done:
                    pass
                else:
                    ret += [[d , distances[i]]]
                    break
    return ret

def chooose_index(l, i , j):
    # list , int , int -> list element
    return l[i][j]

def to_int(s):
    # string -> int , convert '$5,000,000.00' to 5000000
    list_numbers = re.findall('[0-9]+', s)
    number = ''.join(list_numbers)
    return int(int(number)/10)
    
#-------------------------------------------------------------------------------

def main():

    print('Started.')
    
    donations_df = pd.read_csv(url_1, 
                                encoding = "ISO-8859-1")
    restrictions = ['Dollars',
                    'Recipient', 
                    'Recipient Notes', 
                    'Recipient Subsector', 
                    'Recipient Sub Group', 
                    'Gift Notes', 
                    'Gift Purpose',
                    'Disaster']

    donations_df_r = donations_df[restrictions] 

    print('donations.cvs: opened and filtered')

    for column in donations_df_r.columns:   
        donations_df_r[column] = donations_df_r[column].apply(replace_nan)
        
    donations_df_r['Disaster'] = donations_df_r[column].apply(replace_disaster)

    print('NaN values replaced and disaster status added.')

    donations_df_r['Joined'] = (donations_df_r["Recipient"].astype(str)+' '
        +donations_df_r['Recipient Notes'].astype(str)+' '
        +donations_df_r['Recipient Subsector'].astype(str)+' '
        +donations_df_r['Recipient Sub Group'].astype(str)+' '
        +donations_df_r['Gift Notes'].astype(str)+' '
        +donations_df_r['Gift Purpose'].astype(str))
                                
    joined_df = donations_df_r[['Dollars','Joined']]

    print('Strings joined.')
    
    joined_df['Sets of words'] = joined_df['Joined'].apply(str_to_set_of_words)
    joined_df['Sets of words'] = joined_df['Sets of words'].apply(
        remove_stop_words)
    
    print('Stopwords removed and sets of words created.')
    
    words = set({})
    for set_words in joined_df['Sets of words']:
        words = words.union(set_words)
        
    print('Set of words created. Loading word2vec')
    
    word2vec = {}
    with open(url_2, 
              encoding="utf8") as infile:
        for line in infile:
            lst = []
            line = str(line)
            lst = line.split()
            dictkey = lst.pop(0)
            floats = np.array(lst)
            word2vec[dictkey] = floats.astype(np.float64)
            
    print('Word2vec loaded')
    
    wordsout = set({})
    for word in words:
        if not (word in word2vec):
            wordsout.add(word)
            
    print('Spell checking.')
    
    spell = SpellChecker()
    correction = {}
    misspelled = spell.unknown(wordsout)
    for word in misspelled:
        correction[word] = spell.correction(word)
    
    print('Correcting the words.')
    
    joined_df['Sets of words'] = joined_df['Sets of words'].apply(lambda x :
        correct_text(x , correction))
    
    print('Automatic correction done, now doing manual correction.')
    
    words = set({})
    
    for set_words in joined_df['Sets of words']:
        words = words.union(set_words)
    wordsout = set({})
    
    for word in words:
        if not (word in word2vec):
            wordsout.add(word)
            
    correction['abour'] = 'about'
    correction['acces'] = 'access'
    correction['acti'] = 'act'
    correction['actr'] = 'act'
    correction['childern'] = 'children'
    correction['cleen'] = 'clean'
    correction['childr'] = 'children'
    correction['citywards'] = 'city wards'
    correction['comefor'] = 'come for'
    correction['contr'] = 'contribution'
    correction['contrib'] = 'contribution'
    correction['cultur'] = 'culture'
    correction['demostrate'] = 'demonstrate'
    correction['engagment'] = 'engagement'
    correction['peop'] = 'people'
    correction['peopl'] = 'people'
    correction['schoo'] = 'school'
    correction['soci'] = 'social'
    
    joined_df['Sets of words'] = joined_df['Sets of words'].apply(lambda x :
        correct_text(x , correction))
    
    print('Correction done. Now loading and correcting tags.')
        
    tags = pd.read_csv(url_3, 
                                encoding = "ISO-8859-1")
    tags['tags_sets'] = tags['tags'].apply(str_to_set_of_words)
        
    tags['tags_sets'] = tags['tags_sets'].apply(correction_tags)
    tags_dict = {}
    
    for tag in tags['tags']:
        tags_dict[tag] = correction_tags(str_to_set_of_words(tag))
      
    print('Generating distances matrix.')  
            
    word_tags = set({})
    for word_set in tags['tags_sets']:
        word_tags = word_tags.union(word_set)
    word_distances = {}
    
    for word_1 in words:
        for word_2 in word_tags:
            try:
                try:
                    word_distances[word_1][word_2] = cos_angle(
                        word2vec[word_1] , word2vec[word_2])
                except:
                    word_distances[word_1] = {word_2 : cos_angle(
                        word2vec[word_1] , word2vec[word_2])}
            except:
                try:
                    word_distances[word_1][word_2] = 0
                except:
                    word_distances[word_1] = {word_2 : 0}
                    
    print('Ranking.')  
                    
    joined_df['ranks'] = joined_df['Sets of words'].apply(lambda x :
        dist_ranks(x , tags , tags_dict, word_distances))
    
    print('Ranking done, generating output.')  
    
    filter_to_columns = ['Dollars', 'Sets of words']
    
    for i in range(5):
        filter_to_columns += [f'rank {i+1} tag']
        joined_df[f'rank {i+1} tag'] = joined_df['ranks'].apply(lambda x : 
            chooose_index(x, i , 0))
        filter_to_columns += [f'rank {i+1} distance']
        joined_df[f'rank {i+1} distance'] = joined_df['ranks'].apply(lambda x : 
            chooose_index(x, i , 1))
        
    df_final_clean = joined_df[filter_to_columns].copy()
    
    df_final_clean['Dollars'] = df_final_clean['Dollars'].apply(to_int)
    to_join = pd.read_csv(url_1, 
                                encoding = "ISO-8859-1")
    dff = df_final_clean.copy()
    dff = dff.rename(columns = {'Dollars' : 'Dollars Int'})
    to_save = to_join.join(dff)
    to_save.to_csv(url_4, index = False)
    
    print('All done.') 
    
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
