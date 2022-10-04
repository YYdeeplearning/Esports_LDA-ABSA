"""
This code combine all MDAs and could be run for all years.
Output:
The PMDA of each year
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import time
import joblib
# from preprocessing import *

from gensim import corpora
from gensim.models import TfidfModel
from gensim.corpora import MmCorpus

import sys
t1 = time.time()

sec_dt_mda = []  # All MDAs concatenate for all years
len_mda = []  # list of (Length of all MDAs of i year), use for time slice
start = int(sys.argv[1])
stop = int(sys.argv[2])
for year in range(start, stop+1):
    # Load data
    sec_dt = joblib.load('./data/lda/sec_dt_PMDA_F' + str(year) + '.pkl')
    print('Processed: ', year)
    sec_dt = sec_dt['PMDA']
    sec_dt_mda.extend(sec_dt)
    len_mda.append(len(sec_dt))
    sec_dt = None
    print('Next... \n')
joblib.dump(len_mda, './outputs/PMDA_All_len_'+str(start)+'_'+str(stop)+'.pkl')
joblib.dump(sec_dt_mda, './outputs/PMDA_All_'+str(start)+'_'+str(stop)+'.pkl')

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


e1 = time.time()
esp = (e1-t1)/3600
print('Process time: %.3f hour' % esp)


# Test data
# corpus = MmCorpus('./outputs/1PMDA_All_corpus.mm')
# id2word = joblib.load('./outputs/1PMDA_All_dict.pkl')
# len_mda = joblib.load('./outputs/1PMDA_All_len.pkl')
# sec_dt_mda = joblib.load('./outputs/1PMDA_All.pkl')
# l=[]
# for year in range(1997, 2019):
#     # Load data
#     l.extend(joblib.load('./PMDA_All_len_'+str(year)+'.pkl'))
