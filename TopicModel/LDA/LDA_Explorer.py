"""
This job should be submitted using SMALL queue with select=4, could acquire more memory, which is not noted in the
guide of VPCC
"""

import joblib
import pickle as pkl
from utils.preprocessing import split_into_sentences, remove_stopwords

from gensim.models import CoherenceModel
from gensim.corpora import MmCorpus

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


def read_mda(year):
    return joblib.load('./outputs/PMDA_All' + str(year) + '.pkl')


STAGE = int(sys.argv[1])


def infer_topic_sent():
    # STAGE 2: Having observe the optimal number of topics is 30, we proceed
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
        # Example: Infer the topics of 31st sentence of 3rd company
        # dat['sMDA'][2][30]
        # model_list[25][dictionary.doc2bow(remove_stopwords(dat['sMDA'][2][30]))]

        # Generally, as the ITEM should consist of two sentences,
        # we do not infer the topic for first and last two sentences
        model = model_list[idx[i]-10]
        model_list = None

        # Since we want to infer topic for each sentence, we can not utilize the corpus created before to
        # infer the topic because the corpus was made from all MDA.
        # TODO: sent-LDA

        dat['TOPICS'] = dat['sMDA'].apply(lambda x: infer_topic_mda(model, x))
        save_dat = dat[[col for col in dat.columns if col not in ['MDA', 'PMDA', 'PROCESSED_TEXT', 'TEXT_FILE']]]
        joblib.dump(save_dat, './data/lda/infer_topic_'+str(year[i])+'.pkl')


def infer_topic_mda(m, x):
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


def merge_data(nb_topics, path_to_infered_data):
    # STAGE 6: Create Topic vector and merge with financial data
    inferred_mdas = joblib.load(path_to_infered_data)

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


