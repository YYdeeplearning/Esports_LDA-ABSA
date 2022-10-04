"""
This job should be submitted using SMALL queue with select=4, could acquire more memory, which is not noted in the
guide of VPCC
"""

import joblib
import pickle as pkl
from preprocessing import split_into_sentences, remove_stopwords

from gensim.models import CoherenceModel
from gensim.corpora import MmCorpus

from wordcloud import WordCloud
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style as style
import pyLDAvis
import pandas as pd
import numpy as np

from itertools import repeat

from itertools import chain
from operator import itemgetter

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import multiprocessing as mp
from multiprocessing import Pool, Process, Value, Array

import sys
import os
import psutil
# plt.rcParams["font.family"] = "Times New Roman"
color = ['chocolate', 'brown', 'darkgreen', 'darkcyan', 'darkkhaki', 'purple']
style.use('seaborn-paper') #sets the size of the charts
plt.rcParams["font.family"] = "serif"
plt.rcParams['grid.alpha'] = 0.8
plt.rc('grid', linestyle='dotted')
dpi = 500
memory = psutil.virtual_memory().total / (1024.0 ** 3)
logging.info('Number of available CPUs: %d' % mp.cpu_count())
logging.info('Portable CPUs count: %d' % psutil.cpu_count())
logging.info('Usable CPUs: %d' % len(os.sched_getaffinity(0)))
logging.info('Memory info: %.3f' % memory)

import time
t1 = time.time()


def read_mda(year):
    return joblib.load('./outputs/PMDA_All' + str(year) + '.pkl')


STAGE = int(sys.argv[1])

if STAGE == 1:
    # STAGE 1: Compute ch scores for all years
    for year in range(1997, 2019):
        corpus = joblib.load('./outputs/lda/ldamp_' + str(year) + '_corpus.pkl')
        id2word = joblib.load('./outputs/lda/ldamp_' + str(year) + '_id2word.pkl')

        model_list = joblib.load('./outputs/lda/ldamp_' + str(year) + '.pkl')
        dat = joblib.load('./outputs/lda/sec_dt_PMDA_F' + str(year) + '.pkl')
        dat = dat['PMDA']
        dat['sMDA'] = dat['MDA'].map(split_into_sentences)
        dat['LsMDA'] = dat['sMDA'].map(len)
        # Coherence score
        ch_scores = []
        start = time.time()
        for i in range(len(model_list)):
            ch_scores.append(CoherenceModel(model_list[i], corpus=corpus, coherence='u_mass').get_coherence())
        stop = time.time()
        logging.info('Time to compute Coherence Scores: %.3f' % ((stop-start)/3600))

        x = range(model_list[0].num_topics, model_list[-1].num_topics+1)
        plt.plot(x, ch_scores)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.grid()
        plt.savefig('./outputs/plot/topic_coherence_'+str(year)+'.png')
        plt.close()

        # Wordcloud
        lda = model_list[20]
        lda_corpus = lda[corpus]
        wc = WordCloud(background_color="white", stopwords={'december', 'year'})  #, max_words=1000, mask=alice_mask)
        for t in range(lda.num_topics):
            plt.figure()
            plt.imshow(wc.fit_words(dict(lda.show_topic(t, 30))))
            plt.axis("off")
            plt.title("Topic #" + str(t))
            plt.savefig('./outputs/plot/'+str(year) + "_Topic_" + str(t)+'.png')
            plt.show()
            plt.close()

if STAGE == 2:
    # STAGE 2: Having observe the optimal number of topics for 2016 is 35, we proceed with this year
    # to examine the topic distribution
    # """
    # year = list(range(1997, 2019))
    # idx = [40, 35, 35, 35, 35, 30, 40, 30, 40, 30, 40, 35, 40, 40, 40, 35, 35, 35, 30, 35, 30, 40]
    year = list(range(2005, 2010))
    idx = [40, 30, 40, 35, 40]
    for i in range(len(year)):
        corpus = joblib.load('./outputs/lda/ldamp_' + str(year[i]) + '_corpus.pkl')
        dictionary = joblib.load('./outputs/lda/ldamp_' + str(year[i]) + '_id2word.pkl')
        model_list = joblib.load('./data/lda/ldamp_' + str(year[i]) + '.pkl')
        dat = joblib.load('./data/lda/sec_dt_PMDA_F' + str(year[i]) + '.pkl')
        dat['sMDA'] = dat['MDA'].map(split_into_sentences)
        dat['LsMDA'] = dat['sMDA'].map(len)
        dat['LsMDA'].sum()  # Around 2 million sentences
        # Example: Infer the topics of 31th document of 3rd company
        # dat['sMDA'][2][30]
        # model_list[25][dictionary.doc2bow(remove_stopwords(dat['sMDA'][2][30]))]

        # Generally, as the ITEM should consist of two sentences,
        # we do not infer the topic for first and last two sentences
        model = model_list[idx[i]-10]
        model_list = None

        # Since we want to infer topic for each sentence, we can not utilize the corpus created before to
        # infer the topic because the corpus was made from all MDA.
        # TODO: sent-LDA

        dat['TOPICS'] = dat['sMDA'].apply(lambda x: infer_topic(model, x))
        save_dat = dat[[col for col in dat.columns if col not in ['MDA', 'PMDA', 'PROCESSED_TEXT', 'TEXT_FILE']]]
        joblib.dump(save_dat, './data/lda/infer_topic_'+str(year[i])+'.pkl')

if STAGE == 3:
    # STAGE 3: HAVING ALL MDA in one estimation, compute coherence scores for difference LDA model
    ch = sys.argv[2]  # coherence measure u_mass, c_v, c_uci, or c_npmi
    cores = int(sys.argv[3])
    logging.info('Number of processes required: %d' % cores)

    def get_ch(m, c, ch, t=None):
        """
        compute coherence score
        :param m: lda model
        :param c: corpus
        :param ch: coherence measure u_mass, c_v, c_uci, or c_npmi
        :param t: texts for computing c_v, c_uci and c_npmi
        :return: coherence score
        """
        if ch == 'u_mass':
            return CoherenceModel(m, corpus=c, coherence=ch).get_coherence()
        else:
            return CoherenceModel(m, texts=t, coherence=ch).get_coherence()

    # Coherence score
    t1 = time.time()

    if ch == 'u_mass':
        with Pool(processes=cores) as pool:
            corpus = MmCorpus('./outputs/corpus_all.mm')
            logging.info('Corpus loaded!')
            id2word = joblib.load('./outputs/dictionary_all.pkl')
            logging.info('Dictionary loaded!')
            model_list = joblib.load('./outputs/lda/lda_mp_all_year_default_1039.pkl')  # lda_mp_all_year_10k5epoch1039
            logging.info('Model list loaded!')
            ch_scores = pool.starmap(get_ch, zip(model_list, repeat(corpus), repeat(ch)))
        model_list1 = None

        t2 = time.time()
        logging.info(ch_scores)
        logging.info('Time to compute Coherence Scores: %.3f' % ((t2-t1)/3600))

    else:  # c_v, c_uci, or c_npmi
        corpus = None
        logging.info('Number of CPUs: %d\n' % mp.cpu_count())
        # start = 1997
        # stop = 2019
        # # Run multithreads
        # with Pool(processes=cores) as pool:
        #     process = psutil.Process(os.getpid())
        #     logging.info('Current mem usage: %.3f before loading data.' % (process.memory_info().rss / (1024.0 ** 3)))  # in GB
        #     logging.info('Loading text data in each process...')
        #     list_mdas = pool.starmap(read_mda, zip(range(start, stop)))
        # logging.info('Size of the shared texts: %3f' % (1000000000*sys.getsizeof(list_mdas)))
        # texts = Array(list_mdas, lock=False)
        # t2 = time.time()
        # logging.info('All text data from %d to %d are loaded in %.3f \n' % (start, stop, (t2 - t1) / 3600))
        # process = psutil.Process(os.getpid())
        # logging.info('Current mem usage: %.3f after loading data.' % (process.memory_info().rss/(1024.0 ** 3)))  # in GB
        process = psutil.Process(os.getpid())
        logging.info('Current mem usage: %.3f before loading data.' % (process.memory_info().rss / (1024.0 ** 3)))  # in GB
        logging.info('Loading text data in each process...')
        texts = joblib.load('./outputs/lda/All_MDAs.pkl')
        # TODO: pool based on fork() creating the same process which inherit all elements hence require big memory.
        # Considering using shared memory with Array and spawning process manually to keep track of memory usage
        with Pool(processes=cores) as pool:
            model_list = joblib.load('./outputs/lda/lda_mp_all_year_default_1039.pkl')  # lda_mp_all_year_10k5epoch1039
            # joblib.dump(texts, './data/lda/PMDA_All.pkl')
            # logging.info('New data saved!')
            ch_scores = pool.starmap(get_ch, zip(model_list, repeat(corpus), repeat(ch), texts))
        model_list = None

        t2 = time.time()
        # ch_scores_c = ch_scores1+ch_scores2+ch_scores3
        logging.info('Coherence scores:\n')
        logging.info(ch_scores)
        logging.info('Time to compute Coherence Scores: %.3f' % ((t2 - t1) / 3600))
        joblib.dump(ch_scores, './outputs/lda/ALL_MDAS_coherence_scores_1039_' + ch + '.pkl')
        #
        # x = range(10, 100)
        # plt.figure(figsize=(16, 10), dpi=80)
        # plt.plot(x, ch_scores_c, color='tab:red')
        # plt.xlabel("Number of Topics")
        # plt.ylabel("Coherence Score")
        # plt.grid(axis='both', alpha=.3)
        # # Remove borders
        # plt.gca().spines["top"].set_alpha(0.0)
        # plt.gca().spines["bottom"].set_alpha(0.3)
        # plt.gca().spines["right"].set_alpha(0.0)
        # plt.gca().spines["left"].set_alpha(0.3)
        # plt.savefig('./outputs/plot/ALL_MDAs_topic_coherence_' + ch + '.png')
        # plt.close()
        # joblib.dump(ch_scores1, './outputs/lda/coherence039'+ch+'.pkl')

if STAGE == 4:
    # STAGE 4: Infer topic for all years
    def infer_topic(m, smda):
        """
        This function infer the topics for each sentence in a MDA
        :param m: the LDA model
        :param smda: an MDA document organized as sentences
        :return: a list of topics each for all sentences in MDA. The topics here are inferred with maximum probability
        """
        return [max(m[id2word.doc2bow(remove_stopwords(sent))], key=itemgetter(1)) for sent in smda]

    def infer_topic_mdas(y, m):
        """
        This function infer the topics for each MDA
        :param m: the LDA model
        :param y: year
        :return: a matrix (n_sentence x nb_topics) of topics for each MDA.
        The topics here are inferred func infer_topic above
        """
        dat = joblib.load('./data/lda/infer_topic_' + str(y) + '.pkl')
        dat['TOPICS'] = [infer_topic(m, smda) for smda in dat['sMDA']]
        return dat[['TOPICS']]


    cores = int(sys.argv[2])
    # model_list = joblib.load('./outputs/lda/lda_mp_all_year_default_1039.pkl')  # lda_mp_all_year_10k5epoch1039
    nb_topics = 40
    # model = model_list[nb_topics-10]
    # joblib.dump(model, './data/lda/lda30.pkl')
    # model_list = None
    model = joblib.load('./data/lda/lda30.pkl')

    # corpus = MmCorpus('./outputs/corpus_all.mm')
    id2word = joblib.load('./outputs/dictionary_all.pkl')

    with Pool(processes=cores) as pool:
        inferred_mdas = pool.starmap(infer_topic_mdas, zip(range(1997, 2018), repeat(model)))

    logging.info(len(inferred_mdas))
    logging.info(inferred_mdas[1:3])
    joblib.dump(inferred_mdas, './data/lda/ALL_MDAs_infered_' + str(nb_topics) + '.pkl')

if STAGE == 5:
    # STAGE 5: Wordcloud and excel
    nb_topics = int(sys.argv[1])
    filename = 'lda'+str(nb_topics)+'bigram'
    lda = joblib.load('./outputs/lda/lda_train_All_MDAs_30bigram_True.pkl')  # lda_mp_all_year_10k5epoch1039

    # Wordcloud
    sw = {'december', 'year', 'revenue'}
    # sw = {''}
    wc = WordCloud(stopwords=sw, background_color="white")  # , max_words=1000, mask=alice_mask)
    # Stop word will be ignored for fit_words function below
    topics = []
    t_words = []
    for t in range(lda.num_topics):
        topics.extend([t])
        words = lda.show_topic(t, 50)
        t_words.append(list(words))
        plt.figure()
        dictt = dict(words)
        dictt = {i: dictt[i] for i in dictt if i not in list(sw)}
        plt.imshow(wc.fit_words(dictt))
        plt.axis("off")
        plt.title("Topic " + str(t))
        plt.savefig('./wordcloud/Topic_' + str(t) + filename +'.png')
        plt.show()
        plt.close()
    dictt = {'Topic': topics, 'top_words': t_words}
    df = pd.DataFrame(dictt)
    df.to_csv('./wordcloud/LDA_top_words_topic_'+filename+'.csv')

if STAGE == 6:
    # STAGE 6: Create Topic and merge with sentic data
    nb_topics = 30
    inferred_mdas = joblib.load('./data/lda/ALL_MDAs_infered_' + str(nb_topics) + '.pkl')


    def count_topic(t_list, n):
        """
        This function count the number of sentences that are inferred to a topic in a list of inferred topics for all
        sentences in an MDA
        :param t_list: list of inferred topics
        :param n: the topic id
        :return: count
        """
        counts = np.sum([1 for p in t_list if p[0]==n])
        return counts


    # Create Topic count dataframe
    d = pd.DataFrame(columns=['TOPIC'+str(i) for i in range(nb_topics)])
    for df in inferred_mdas:
        df_lda = pd.DataFrame()
        for i in range(nb_topics):
            df_lda['TOPIC'+str(i)] = df['TOPICS'].apply(lambda x: count_topic(x, i))
        d = d.append(df_lda)

    logging.info(d.columns)
    logging.info(d.shape)
    d.to_csv('./data/lda/infer_count_test.csv')

if STAGE == 7:
    """
    Read the saved perplexity and coherence values and plot
    """
    list_per = []
    list_coh = []
    for i in range(5, 105, 5):
        fn = glob.glob('./outputs/lda/perplexity_value_All_MDAs_'+str(i)+'bigram_True.pkl')[0]
        list_per.append(joblib.load(fn))
    plt.plot(range(5, 105, 5), list_per)
    plt.grid()
    plt.savefig('./outputs/lda/perplexity.png')
    plt.close()

    for i in range(5, 105, 5):
        fn = glob.glob('./outputs/lda/coherence_value_All_MDAs_' + str(i) + 'bigram_True.pkl')[0]
        list_coh.append(joblib.load(fn))
    plt.plot(range(5, 105, 5), list_coh)
    plt.grid()
    plt.savefig('./outputs/lda/cohenrence.png')
    plt.close()

if STAGE == 8:
    """
       Read the saved lda models, compute perplexity and coherence values for test corpus and plot
    """
    corp_train = joblib.load('./outputs/lda/corp_train_All_MDAs.pkl')
    corp_test = joblib.load('./outputs/lda/corp_test_All_MDAs.pkl')
    total_docs = int(len(corp_test) / 2)
    list_per = []
    list_coh_train = []
    list_coh_test = []
    list_coh_all = []
    for i in range(5, 105, 5):
        fn_mod = glob.glob('./outputs/lda/lda_train_All_MDAs_' + str(i) + '_*.pkl')[0]
        lda_train = joblib.load(fn_mod)
        list_coh_train.append(CoherenceModel(model=lda_train, corpus=corp_train, coherence='u_mass').get_coherence())
        list_coh_test.append(CoherenceModel(model=lda_train, corpus=corp_test, coherence='u_mass').get_coherence())
        corpus_all = corp_train+corp_test
        try:
            list_coh_all.append(CoherenceModel(model=lda_train, corpus=corpus_all, coherence='u_mass').get_coherence())
        except:
            print('Can not estimate the coherence value for all corpus!')
        # list_per.append(lda_train.log_perplexity(corp_test, total_docs=total_docs))
    joblib.dump(list_coh_train, './outputs/lda/list_coherence_train.pkl')
    joblib.dump(list_coh_test, './outputs/lda/list_coherence_test.pkl')
    joblib.dump(list_coh_all, './outputs/lda/list_coherence_all.pkl')
    # joblib.dump(list_per, './outputs/lda/list_perplexity.pkl')
    # plt.plot(range(5, 105, 5), list_per)
    # plt.savefig('./outputs/lda/perplexity.png')
    # plt.close()
    plt.plot(range(5, 105, 5), list_coh_train)
    plt.savefig('./outputs/lda/cohenrence_train.png')
    plt.close()
    plt.plot(range(5, 105, 5), list_coh_test)
    plt.savefig('./outputs/lda/cohenrence_test.png')
    plt.close()
    plt.plot(range(5, 105, 5), list_coh_all)
    plt.savefig('./outputs/lda/cohenrence_all.png')

t2 = time.time()
esp = (t2-t1)/3600
logging.info('Total running time: %3f' % esp)