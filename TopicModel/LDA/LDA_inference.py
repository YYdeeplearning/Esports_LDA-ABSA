
"""
This code is used for parallel inferences for LDA models
"""

from TA_LDA import *
import sys
import joblib
from wordcloud import WordCloud
import logging
from itertools import chain
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
year = int(sys.argv[1])
min_topics = int(sys.argv[2])
max_topics = int(sys.argv[3])
test = sys.argv[4]
multi_core = int(sys.argv[5])
lemma = sys.argv[6]

# year = 2003
# min_topics = 5
# max_topics = 10
# test = 'True'
# multi_core = 5
# lemma = 'spacy'


# sec_dt = joblib.load('./outputs/sec_dt_'+str(year)+'.pkl')
# sec_dt = sec_dt['MDA']
# # len(sec_dt)
# # Modeling
# MDAs, corpus, id2word, model_list, model_list_train, coherence_list, perplexity_list, filename = \
#     estimate_lda(sec_dt, year, max_topics, min_topics, test, multi_core, lemma)

# sents = joblib.load('./outputs/lda/MDA_' + str(year) + '.pkl')
# r_id = joblib.load('./outputs/lda/reportID_' + str(year) + '.pkl')
# corpus = joblib.load('./outputs/lda/corpus_' + filename + '.pkl')
# id2word = joblib.load('./outputs/lda/id2word_' + filename + '.pkl')
# model_list = joblib.load('./outputs/lda/' + filename + 'model_list.pkl')
# model_list_train = joblib.load('./outputs/lda/' + filename + 'model_list_train.pkl')
# coherence_list = joblib.load('./outputs/lda/' + filename + 'coherence_values.pkl')
# perplexity_list = joblib.load('./outputs/lda/' + filename + 'perplexity_values.pkl')
# dat = joblib.load('./outputs/lda/MDA_in-house_30_300.77_2008.pkl')
# model_list = joblib.load('./outputs/lda/in-house_30_300.77_2008model_list.pkl')
# lda = model_list[0]
# corpus = joblib.load('./outputs/lda/corpus_in-house_30_300.77_2008.pkl')
# lda_corpus = lda[corpus]
#
#
# # First way
# scores = list(chain(*[[score for topic_id,score in topic] \
#                       for topic in [doc for doc in lda_corpus]]))
#
#
# #threshold
# threshold = sum(scores)/len(scores)
# threshold
#
#
# #cluster1
# cluster1 = [j for i, j in zip(lda_corpus, dat) if i[0][1] > threshold]
# #cluster2
# cluster2 = [j for i, j in zip(lda_corpus, dat) if i[1][1] > threshold]
#
#
# # Other way
# lda_corpus = [max(prob, key=lambda y:y[1]) for prob in lda[corpus] ]
# playlists = [[] for i in range(lda.num_topics)]
#
# for i, x in enumerate(lda_corpus):
#     playlists[x[0]].append(dat[i])
#
# # Wordcloud
# import matplotlib.pyplot as plt
# for t in range(lda.num_topics):
#     plt.figure()
#     plt.imshow(WordCloud().fit_words(dict(lda.show_topic(t, 30))))
#     plt.axis("off")
#     plt.title("Topic #" + str(t))
#     plt.show()