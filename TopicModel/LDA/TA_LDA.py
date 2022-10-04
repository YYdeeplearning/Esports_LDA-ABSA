"""
Old code running LDA on sequential machine.

Preprocessing:
For each dataset, we apply standard preprocessing techniques, such as tokenization
and removal of numbers and punctuation marks. We also filter out stop words,
i.e., words with document frequency above 70%, as well as standard stop words
from a list. Additionally, we remove low-frequency words, i.e., words that appear
in less than a certain number of documents (30 documents for UN debates, 100
documents for the SCIENCE corpus, and 10 documents for the ACL dataset). We
use 85% randomly chosen documents for training, 10% for testing, and 5% for
validation, and we remove one-word documents from the validation and test sets.

In particular, we run 5 epochs of LDA followed by 120
epochs of D-LDA. For D-LDA, we use RMSProp (Tieleman and Hinton, 2012) to set
the step size, setting the learning rate to 0.05 for the mean parameters and to 0.005
for the variance parameters
"""

import gensim.corpora as corpora
from gensim.models import CoherenceModel, TfidfModel
from gensim.models import LdaModel, LdaMulticore
from gensim.models.phrases import Phrases, Phraser

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import multiprocessing as mp
from multiprocessing import Pool, Process, Value, Array

import os
import time
import psutil

from preprocessing import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "9"

import numpy as np
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this

cwd = os.getcwd()


def read_mda(year):
    return joblib.load('./outputs/PMDA_All' + str(year) + '.pkl')


def estimate_lda(num_topics=40, cores=4, bigram=False):
    """
    estimate LDA model for all years of data
    :param dt: list of MDA texts, a column of big dataframe like sec_dt
    :param max_topics:
    :param min_topics:
    :param test: test the LDA model
    :param multi_core: number of cores
    :param lemma: lemmatizer
    :return: list sentence, report_id, corpus, id_word, bigram_mod, model_list, coherence_list, perplexity_list
    """
    t1 = time.time()
    logging.info('Number of CPUs: %d' % mp.cpu_count())
    start = 1997
    stop = 2019
    # Run single thread
    if cores < 2:
        list_mdas = []
        for year in range(start, stop):
            list_mdas = list_mdas + read_mda(year)
    else:
        # Run multithreads
        with Pool(processes=cores) as pool:
            list_mdas = pool.starmap(read_mda, zip(range(start, stop)))
        list_mdas = [j for i in list_mdas for j in i]
    t2 = time.time()
    logging.info('All text data from %d to %d are loaded in full in %.3f ' % (start, stop, (t2 - t1) / 3600))

    # process = psutil.Process(os.getpid())
    # logging.info('Current mem usage: %.3f after loading data.' % (process.memory_info().rss / (1024.0 ** 3)))  # in GB

    # Estimate
    corp_train, corp_test, dict_all, lda_train, coherence, perplexity = \
        cross_validate_LDA(texts=list_mdas, num_topics=num_topics, bi_gram=bigram, multi_core=cores)
    t2 = time.time()
    logging.info('Total computing time: %.3f', (t2 - t1) / 3600)
    filename = 'All_MDAs_' + str(num_topics) + 'bigram_' + str(bigram)

    # Save model list
    # joblib.dump(corp_train, './outputs/lda/corp_train_' + filename + '.pkl')
    # joblib.dump(corp_test, './outputs/lda/corp_test_' + filename + '.pkl')
    # joblib.dump(dict_all, './outputs/dict_all_' + filename + '.pkl')
    joblib.dump(lda_train, './outputs/lda/lda_train_' + filename + '.pkl')
    joblib.dump(coherence, './outputs/lda/coherence_value_' + filename + '.pkl')
    joblib.dump(perplexity, './outputs/lda/perplexity_value_' + filename + '.pkl')


def cross_validate_LDA(texts, num_topics, bi_gram=False, multi_core=0):
    """
    Compute coherence and hold-out perplexity for test data
    1. In the first case you're trying to figure out how well you model "explains" unseen data.
    2. In the second one you evaluate the 'perceptual quality' of your topics,
    which aren't supposed to change during a test stage.

    The `LdaModel.bound()` method computes a lower bound
    on perplexity, based on a supplied corpus (~of held-out documents).
    This is the method used in Hoffman&Blei&Bach in their "Online Learning
    for LDA" NIPS article. https://groups.google.com/forum/#!topic/gensim/LM619SB57zM

    Perplexity:
    1. https://stats.stackexchange.com/questions/305846/how-should-perplexity-of-lda-behave-as-value-of-the-latent-variable-k-increases?rq=1
    2. https://stats.stackexchange.com/questions/273355/why-does-lower-perplexity-indicate-better-generalization-performance?rq=1
    Coherence Model:
    1. https://radimrehurek.com/gensim/models/coherencemodel.html
    2. https://stats.stackexchange.com/questions/375062/how-does-topic-coherence-score-in-lda-intuitively-makes-sense?rq=1
    3. https://stats.stackexchange.com/questions/322809/inferring-the-number-of-topics-for-gensims-lda-perplexity-cm-aic-and-bic?rq=1
    Topic stability
    1. https://stats.stackexchange.com/questions/63026/topic-stability-in-topic-models?rq=1
    Parameters:
    ----------
    dictionary: Gensim dictionary
    corpus: Gensim corpus
    texts: List of input texts
    limit: Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """

    # Set training parameters.
    passes = 10
    iterations = 100
    decay = 0.5
    offset = 64
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    RANDOM_STATE = 42
    logging.info("Estimating LDA with " + str(num_topics)+" topics...")

    # n-gram
    if bi_gram:
        try:
            bigram = Phraser.load('./outputs/bigram_all_SEC10K_phraser')
            logging.info('Loaded bigram at: ./outputs/bigram_all_SEC10K_phraser')
        except:
            min_count = 1000  # set to int(len(texts) / 10) results in 163 bigram including lot of website links with
            # filter_extremes(no_below=10, no_above=0.7)
            bi_phrases = Phrases(texts, min_count=min_count, threshold=10)
            bi_phrases.save('./outputs/bigram_all_SEC10K_phrases')
            bigram = Phraser(bi_phrases)
            bigram.save('./outputs/bigram_all_SEC10K_phraser')
            logging.info('Saved bigram at: ./outputs/bigram_all_SEC10K_phraser')
        texts = bigram[texts]
    # Compute hold-out perplexity score
    cutoff = 0.8
    c_train = texts[0:int(cutoff * len(texts))]
    c_test = texts[int(cutoff * len(texts)):-1]

    # Creating dictionary will take 38 minutes
    try:
        dict_all = corpora.Dictionary.load('./outputs/dict_all_bigram-'+str(bi_gram)+'_SEC10K')
    except:
        dict_all = corpora.Dictionary(texts)
        dict_all.save('./outputs/dict_all_bigram-'+str(bi_gram)+'_SEC10K')

    # Filter extreme tokens
    dict_all.filter_extremes(no_below=10, no_above=0.9)

    # bag-of-word
    t3 = time.time()
    if multi_core > 1:
        with Pool(processes=multi_core) as pool:
            corp_train = pool.map(dict_all.doc2bow, c_train)
        with Pool(processes=multi_core) as pool:
            corp_test = pool.map(dict_all.doc2bow, c_test)
    else:
        # This non-parallel implementation takes 38 minutes for all SEC 10K
        corp_train = [dict_all.doc2bow(sent) for sent in c_train]
        corp_test = [dict_all.doc2bow(sent) for sent in c_test]
    logging.info('BoW in %.3f ' % ((time.time() - t3) / 3600))

    if multi_core > 0:
        chunksize = int(np.ceil(len(corp_train)/(multi_core*10)))*10
    else:
        chunksize = 1000

    logging.info("Compute perplexity on hold-out corpus with (k=%d)" % num_topics)
    t3 = time.time()
    if multi_core <= 1:
        lda_train = LdaModel(
                                    corp_train, id2word=dict_all, num_topics=num_topics, chunksize=chunksize,
                                    passes=passes, iterations=iterations, random_state=RANDOM_STATE,
                                    eval_every=eval_every, minimum_probability=0,
                                    alpha='auto',  # shown to be better than symmetric in most cases
                                    decay=decay, offset=offset,  # best params from Hoffman paper,
                                 )
    if multi_core > 1:
        # Cannot allocate memory if using VPCC, use UV instead
        lda_train = LdaMulticore(
                                    corp_train, id2word=dict_all, num_topics=num_topics, chunksize=chunksize,
                                    passes=passes, iterations=iterations, random_state=RANDOM_STATE,
                                    eval_every=eval_every, minimum_probability=0,
                                    # alpha='asymmetric',  # shown to be better than symmetric in most cases
                                    decay=decay, offset=offset,  # best params from Hoffman paper,
                                    workers=multi_core
                                 )

    t4 = time.time()
    logging.info('Time to train LDA: %.3f', (t4 - t3) / 3600)

    t3 = time.time()
    cm = CoherenceModel(model=lda_train, corpus=corp_train, coherence='u_mass').get_coherence()
    """ IF USING corp_test to compute coherence
    Traceback (most recent call last):
    File "TA_LDA_Explorer.py", line 356, in <module>
    list_coh.append(CoherenceModel(model=lda_train, corpus=corp_test, coherence='u_mass').get_coherence())
    File "/work/s1720411/py35/lib/python3.6/site-packages/gensim/models/coherencemodel.py", line 435, in get_coherence
    confirmed_measures = self.get_coherence_per_topic()
    File "/work/s1720411/py35/lib/python3.6/site-packages/gensim/models/coherencemodel.py", line 425, in 
    get_coherence_per_topic
    return measure.conf(segmented_topics, self._accumulator, **kwargs)
    File "/work/s1720411/py35/lib/python3.6/site-packages/gensim/topic_coherence/direct_confirmation_measure.py", 
    line 71, in log_conditional_probability
    m_lc_i = np.log(((co_occur_count / num_docs) + EPSILON) / (w_star_count / num_docs))
    ZeroDivisionError: float division by zero

    """
    logging.info('Cohenrence score: %.3f' % cm)
    t4 = time.time()
    logging.info('Time to compute coherence: %.3f', (t4 - t3) / 3600)

    t3 = time.time()
    total_docs = int(len(corp_test)/2)
    perplex = lda_train.log_perplexity(corp_test, total_docs=total_docs)
    t4 = time.time()
    logging.info('Time to compute perplexity: %.3f', (t4 - t3) / 3600)
    return corp_train, corp_test, dict_all, lda_train, cm, perplex


if __name__ == '__main__':
    t1 = time.time()
    num_topics = int(sys.argv[1])
    logging.info('Estimating LDA with %d topics...' % num_topics)
    cores = int(sys.argv[2])
    logging.info('Estimating LDA using %d cores...' % cores)
    estimate_lda(num_topics, cores, True)
