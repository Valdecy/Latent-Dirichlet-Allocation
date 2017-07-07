############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Text Mining

# Citation: 
# PEREIRA, V. (2017). Project: LDA - Latent Dirichlet Allocation, File: Python-DM-Text Mining-01.py, GitHub repository: <https://github.com/Valdecy/LDA - Latent Dirichlet Allocation>

############################################################################

# pip install stop-words

# Installing Required Libraries
import numpy  as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from random import randint

# Function: lda_tm
def lda_tm(document = [], K = 2, alpha = 0.12, eta = 0.01, iterations = 5000, dtm_matrix = False, dtm_bin_matrix = False, dtm_tf_matrix = False, dtm_tfidf_matrix = False, co_occurrence_matrix = False, correl_matrix = False):
   
    ################ Part 1 - Start of Function #############################
    tokenizer = RegexpTokenizer(r'\w+')
    result_list = []
    
    # English Stopwords
    stop_words_en = get_stop_words('en')
   
    # Corpus
    corpus = []
    for i in document:
        tokens = tokenizer.tokenize(i.lower())
        tokens = [i for i in tokens if not i in stop_words_en]
        corpus.append(tokens)
    
    # Corpus ID
    corpus_id = []
    for i in document:
        tokens = tokenizer.tokenize(i.lower())
        tokens = [i for i in tokens if not i in stop_words_en]
        corpus_id.append(tokens)
    
    # Unique Words
    uniqueWords = []
    for j in range(0, len(corpus)): 
        for i in corpus[j]:
            if not i in uniqueWords:
                uniqueWords.append(i)
       
    # Corpus ID for Unique Words   
    for j in range(0, len(corpus)): 
        for i in range(0, len(uniqueWords)):
            for k in range(0, len(corpus[j])): 
                if uniqueWords[i] == corpus[j][k]:
                    corpus_id[j][k]  = i  
    
    # Topic Assignment
    topic_assignment = []
    for i in document:
        tokens = tokenizer.tokenize(i.lower())
        tokens = [i for i in tokens if not i in stop_words_en]
        topic_assignment.append(tokens)
    
    # dtm
    if dtm_matrix == True or dtm_bin_matrix == True or dtm_tf_matrix == True or dtm_tfidf_matrix == True:
        dtm = np.zeros(shape = (len(corpus), len(uniqueWords)))   
        for j in range(0, len(corpus)): 
            for i in range(0, len(uniqueWords)):
                for k in range(0, len(corpus[j])): 
                    if uniqueWords[i] == corpus[j][k]:
                        dtm[j][i]  = dtm[j][i] + 1
        dtm_pd = pd.DataFrame(dtm, columns = uniqueWords)
        result_list.append(dtm_pd)
    
    # dtm_bin
    if dtm_bin_matrix == True:
        dtm_bin = np.zeros(shape = (len(corpus), len(uniqueWords)))  
        for i in range(0, len(corpus)): 
            for j in range(0, len(uniqueWords)):
                if dtm[i,j] > 0:
                    dtm_bin[i,j] = 1
        dtm_bin_pd = pd.DataFrame(dtm_bin, columns = uniqueWords)
        result_list.append(dtm_bin_pd)
    
    # dtm_tf
    if dtm_tf_matrix == True:
        dtm_tf = np.zeros(shape = (len(corpus), len(uniqueWords))) 
        for i in range(0, len(corpus)): 
            for j in range(0, len(uniqueWords)):
                if dtm[i,j] > 0:
                    dtm_tf[i,j] = dtm[i,j]/dtm[i,].sum()
        dtm_tf_pd = pd.DataFrame(dtm_tf, columns = uniqueWords)
        result_list.append(dtm_tf_pd)
    
    # dtm_tfidf
    if dtm_tfidf_matrix == True:
        idf  = np.zeros(shape = (1, len(uniqueWords)))  
        for i in range(0, len(uniqueWords)):
            idf[0,i] = np.log10(dtm.shape[0]/(dtm[:,i]>0).sum())
        dtm_tfidf = np.zeros(shape = (len(corpus), len(uniqueWords)))
        for i in range(0, len(corpus)): 
            for j in range(0, len(uniqueWords)):
                dtm_tfidf[i,j] = dtm_tf[i,j]*idf[0,j]
        dtm_tfidf_pd = pd.DataFrame(dtm_tfidf, columns = uniqueWords)
        result_list.append(dtm_tfidf_pd)
    
    # Co-occurrence Matrix
    if co_occurrence_matrix == True:
        co_occurrence = np.dot(dtm_bin.T,dtm_bin)
        co_occurrence_pd = pd.DataFrame(co_occurrence, columns = uniqueWords, index = uniqueWords)
        result_list.append(co_occurrence_pd)
    
    # Correlation Matrix
    if correl_matrix == True:
        correl = np.zeros(shape = (len(uniqueWords), len(uniqueWords)))
        for i in range(0, correl.shape[0]): 
            for j in range(i, correl.shape[1]):
                correl[i,j] = np.corrcoef(dtm_bin[:,i], dtm_bin[:,j])[0,1]
        correl_pd = pd.DataFrame(correl, columns = uniqueWords, index = uniqueWords)
        result_list.append(correl_pd) 
   
    # LDA Initialization
    for i in range(0, len(topic_assignment)): 
        for j in range(0, len(topic_assignment[i])): 
            topic_assignment[i][j]  = randint(0, K-1)
    
    cdt = np.zeros(shape = (len(topic_assignment), K))
    for i in range(0, len(topic_assignment)): 
        for j in range(0, len(topic_assignment[i])): 
            for m in range(0, K): 
                if topic_assignment[i][j] == m:
                    cdt[i][m]  = cdt[i][m] + 1
    
    cwt = np.zeros(shape = (K,  len(uniqueWords)))
    for i in range(0, len(corpus)): 
        for j in range(0, len(uniqueWords)):
            for m in range(0, len(corpus[i])):
                if uniqueWords[j] == corpus[i][m]:
                    for n in range(0, K):
                        if topic_assignment[i][m] == n:
                            cwt[n][j]  = cwt[n][j] + 1 
    
    # LDA Algorithm
    for i in range(0, iterations): 
        for d in range(0, len(corpus)):
            for w in range(0, len(corpus[d])):
                t0 = topic_assignment[d][w]
                wid = corpus_id[d][w]
                cdt[d,t0] = cdt[d,t0] - 1 
                cwt[t0,wid] = cwt[t0,wid] - 1
                p_z = ((cwt[:,wid] + eta) / (np.sum((cwt), axis = 1) + len(corpus) * eta)) * ((cdt[d,] + alpha) / (sum(cdt[d,]) + K * alpha )) 
                z = np.sum(p_z)
                p_z_ac = np.add.accumulate(p_z/z)   
                u = np.random.random_sample()
                for m in range(0, K):
                    if u <= p_z_ac[m]:
                        t1 = m
                        break
                topic_assignment[d][w] = t1 
                cdt[d,t1] = cdt[d,t1] + 1 
                cwt[t1,wid] = cwt[t1,wid] + 1
                print('iteration:', i)
        
    theta = (cdt + alpha)
    for i in range(0, len(theta)): 
        for j in range(0, K):
            theta[i,j] = theta[i,j]/np.sum(theta, axis = 1)[i]
    
    result_list.append(theta)
        
    phi = (cwt + eta)
    d_phi = np.sum(phi, axis = 1)
    for i in range(0, K): 
        for j in range(0, len(phi.T)):
            phi[i,j] = phi[i,j]/d_phi[i]
     
    phi_pd = pd.DataFrame(phi.T, index = uniqueWords)
    result_list.append(phi_pd)
    
    return result_list

    ############### End of Function ##############

######################## Part 2 - Usage ####################################

# Documents
doc_1 = "Data Mining is a technique. Data Mining is my first favourite technique."
doc_2 = "Data Mining is a technique. Data Mining is my second favourite technique."
doc_3 = "Data Mining is a technique. Data Mining is my third favourite technique."
doc_4 = "Data Mining is a technique. Data Mining is my fourth favourite technique."
doc_5 = "On Friday, I will play the guitar."
doc_6 = "On Saturday, I will play the guitar."
doc_7 = "On Sunday, I will play the guitar."
doc_8 = "On Monday, I will play the guitar."
doc_9 = "Very good! Very good indeed! How can I thank you?"

# Compile Documents
docs = [doc_1, doc_2, doc_3, doc_4, doc_5, doc_6, doc_7, doc_8, doc_9]

# Call Function
lda = lda_tm(document = docs, K = 3, alpha = 0.12, eta = 0.01, iterations = 2500, co_occurrence_matrix = True)

########################## End of Code #####################################
