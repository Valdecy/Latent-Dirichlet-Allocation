# Latent-Dirichlet-Allocation
Latent Dirichlet Allocation (LDA) function with Collapsed Gibbs Sampling. The function preprocess the corpus and returns: 1) the documents per topic probabilities and 2) the term per topic probabilities. It also computes the dtm, binary dtm, tf dtm and tf-idf dtm if required.

* K = The total number of topics. Default: 2
* alpha = Dirichlet prior. Default: 0.12, 
* eta = Dirichlet prior. Default: 0.01, 
* iterations = The total number of iterations. Default: 5000, 
* dtm_matrix = Computes the dtm if True. Default: False, 
* dtm_bin_matrix = Computes the binary dtm if True. Default: False, 
* dtm_tf_matrix = Computes the tf dtm if True. Default: False, 
* dtm_tfidf_matrix = Computes the tf-idf dtm if True. Default: False, 
* co_occurrence_matrix = Computes the co-occurence matrix if True. Default: False,
* correl_matrix = Computes the terms correlation matrix if True. Default: False
