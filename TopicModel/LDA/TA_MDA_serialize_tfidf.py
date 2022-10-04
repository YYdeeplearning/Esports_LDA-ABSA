"""
This code should be run following the MDA_extract. It loads the sec_dt data frame and:
- filters with more stopwords
- filters low value words with tf-idf
and it should be submit for only one year to use the multi node login

Output:
- new sec_dt dataframe with one more column 'PMDA' - processed MDA
- Serialized corpus, dictionary
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import time
from preprocessing import *

from gensim import corpora
from gensim.models import TfidfModel
from gensim.corpora import MmCorpus
t1 = time.time()

year = int(sys.argv[1])

# Load data
sec_dt = joblib.load('./data/sec_dt_' + str(year) + '.pkl')
sec_dt['PMDA'] = sec_dt['MDA'].apply((lambda x: remove_stopwords(x, lemma='spacy')))
joblib.dump(sec_dt, './outputs/sec_dt_PMDA_F' + str(year) + '.pkl')

# Create and save corpus and dict
id2word = corpora.Dictionary(sec_dt['PMDA'])
corpus = [id2word.doc2bow(mda) for mda in sec_dt['PMDA']]  # we consider all MDA is a document here
# TODO: sentence level LDA

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

MmCorpus.serialize('./outputs/corpus_PMDA_F' + str(year) + '.mm', corpus)
joblib.dump(id2word, './outputs/dict_PMDA_F' + str(year) + '.pkl')
# 'F' mean filtered with tfidf

e1 = time.time()
esp = (e1-t1)/3600
print('Process time: %.3f hour' % esp)