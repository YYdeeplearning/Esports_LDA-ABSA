"""
This code simply combines all pickled PMDAs and serializes them. It should be run after TA_MDA_ALL.py
Output:
- Serialized corpus, dictionary
"""
import time
from preprocessing import *
import joblib

from gensim import corpora
from gensim.models import TfidfModel
from gensim.corpora import MmCorpus

import multiprocessing as mp
from multiprocessing import Pool


def read_mda(year):
    return joblib.load('./outputs/PMDA_All' + str(year) + '.pkl')


if __name__ == '__main__':
    print(mp.cpu_count())
    cores = 22
    start = 1997
    stop = 2018

    # Run multithreads
    with Pool(processes=cores) as pool:
        res = pool.starmap(read_mda, zip(range(start, stop)))

    joblib.dump(res, './outputs/PMDA_ALL_corpus.pkl')
#
# # Create and save corpus and dict
# id2word = corpora.Dictionary(sec_dt_mda)
# corpus = [id2word.doc2bow(sent) for sent in sec_dt_mda]
#
# # filter extreme, low-value words
# tfidf = TfidfModel(corpus, id2word=id2word)
#
# # filter low value words
# low_value = 0.025
# for i in range(0, len(corpus)):
#     bow = corpus[i]
#     low_value_words = [id for id, value in tfidf[bow] if value < low_value]
#     new_bow = [b for b in bow if b[0] not in low_value_words]
#     # reassign
#     corpus[i] = new_bow
#
# MmCorpus.serialize('./outputs/PMDA_All_corpus.mm', corpus)
# joblib.dump(id2word, './outputs/PMDA_All_dict.pkl')
# joblib.dump(len_mda, './outputs/PMDA_All_len.pkl')
# joblib.dump(sec_dt_mda, './outputs/PMDA_All_corpus.pkl')
#
# e1 = time.time()
# esp = (e1-t1)/3600
# print('Process time: %.3f hour' % esp)


# Test data
# corpus = MmCorpus('./outputs/1PMDA_All_corpus.mm')
# id2word = joblib.load('./outputs/1PMDA_All_dict.pkl')
# len_mda = joblib.load('./outputs/1PMDA_All_len.pkl')
# sec_dt_mda = joblib.load('./outputs/1PMDA_All.pkl')