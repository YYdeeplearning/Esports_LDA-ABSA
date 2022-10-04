#-*-coding:utf-8 -*-

"""
##############

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
from gensim.models import CoherenceModel
from gensim.models import LdaModel, LdaMulticore
from gensim.corpora import MmCorpus, Dictionary
from gensim import corpora
from gensim.models import TfidfModel
import multiprocessing as mp
import os
import sys
import time
import numpy as np
from preprocessing import *
import logging
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
nltk.download('stopwords')

t1 = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
matplotlib.use('Agg')

game = sys.argv[1]
cwd = os.getcwd()


def dea_review(game):
    dta = pd.read_csv('./Esports_data/Collected_Review/' + game + '_csv.csv', index_col=0, engine='python')
    dta['review'] = dta['review'].apply(str)
    dta['review'] = dta['review'].apply(lambda x: x.lower())

    dta['Preview'] = dta['review'].apply((lambda x: remove_stopwords(x, lemma='spacy')))

    id2word = corpora.Dictionary(dta['Preview'])
    corpus = [id2word.doc2bow(mda) for mda in dta['Preview']]  # we consider all MDA is a document here

    # filter extreme, low-value words
    tfidf = TfidfModel(corpus, id2word=id2word)

    # filter low value words
    low_value = 0.025
    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = []  # reinitialize to be safe. You can skip this.
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        new_bow = [b for b in bow if b[0] not in low_value_words]
        # reassign
        corpus[i] = new_bow

    MmCorpus.serialize('./outputs/corpus_' + game + '_Preview.mm', corpus)
    joblib.dump(id2word, './outputs/dict_' + game + '_Preview.pkl')
    # 'F' mean filtered with tfidf

    e1 = time.time()
    esp = (e1 - t1) / 3600
    print('Process time: %.3f hour' % esp)


def read_review(game):
    corpus = MmCorpus('./outputs/corpus_' + game + '_Preview.mm')
    id2word = joblib.load('./outputs/dict_' + game + '_Preview.pkl')
    return corpus, id2word


def estimate_lda(num_topics=40, cores=4, game=game):
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
    logging.info('Number of CPUs: %d' % mp.cpu_count())
    dea_review(game)
    corpus, id2word = read_review(game)


    # Estimate
    corp_train, corp_test, dict_all, lda_train, coherence, perplexity = \
        cross_validate_LDA(corpus=corpus, id2word=id2word, num_topics=num_topics, multi_core=cores)
    t2 = time.time()
    logging.info('Total computing time: %.3f', (t2 - t1) / 3600)
    filename = game + '_' + str(num_topics)

    # Save model list
    # joblib.dump(corp_train, './outputs/lda/corp_train_' + filename + '.pkl')
    # joblib.dump(corp_test, './outputs/lda/corp_test_' + filename + '.pkl')
    # joblib.dump(dict_all, './outputs/dict_all_' + filename + '.pkl')
    joblib.dump(lda_train, './outputs/lda/lda_train_' + filename + '.pkl')
    joblib.dump(coherence, './outputs/lda/coherence_value_' + filename + '.pkl')
    joblib.dump(perplexity, './outputs/lda/perplexity_value_' + filename + '.pkl')


def cross_validate_LDA(corpus, id2word, num_topics, multi_core=0):
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

    # Compute hold-out perplexity score
    cutoff = 0.8
    corp_train = corpus[0:int(cutoff * len(corpus))]
    corp_test = corpus[int(cutoff * len(corpus)):-1]

    # Creating dictionary
    dict_all = Dictionary.from_corpus(corpus, id2word)

    if multi_core > 0:
        chunksize = int(np.ceil(len(corp_train)/(multi_core*10)))*10
    else:
        chunksize = 1000

    logging.info("Compute perplexity on hold-out corpus with (k=%d)" % num_topics)
    t3 = time.time()
    if multi_core <= 1:
        lda_train = LdaModel(corp_train, id2word=dict_all, num_topics=num_topics, chunksize=chunksize,
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

    logging.info('Cohenrence score: %.3f' % cm)
    t4 = time.time()
    logging.info('Time to compute coherence: %.3f', (t4 - t3) / 3600)

    t3 = time.time()
    total_docs = int(len(corp_test)/2)
    perplex = lda_train.log_perplexity(corp_test, total_docs=total_docs)
    t4 = time.time()
    logging.info('Time to compute perplexity: %.3f', (t4 - t3) / 3600)
    return corp_train, corp_test, dict_all, lda_train, cm, perplex


t0 = time.time()



for num_topics in range(3, 101):
    logging.info('Estimating LDA with %d topics...' % num_topics)
    cores = mp.cpu_count()
    logging.info('Estimating LDA using %d cores...' % cores)
    estimate_lda(num_topics, cores, game)

coherence_list = []
perplexity_list = []

for i in range(3, 101):
    coherence_list.append(joblib.load('./outputs/lda/coherence_value_' + game + '_' + str(i) + '.pkl'))
    perplexity_list.append(joblib.load('./outputs/lda/perplexity_value_' + game + '_' + str(i) + '.pkl'))

plt.plot(range(3, 101), coherence_list)
plt.title('Coherence_' + game)
plt.savefig('./outputs/lda/Coherence_' + game+'.png')

plt.plot(range(3, 101), perplexity_list)
plt.title('Perplexity_' + game)
plt.savefig('./outputs/lda/Perplexity_' + game+'.png')

t20 = time.time()
logging.info('Total Time to train All LDA: %.3f', (t20 - t0) / 3600)