#-*-coding:utf-8 -*-

from preprocessing import *
from gensim import corpora
from gensim.models import TfidfModel
from gensim.corpora import MmCorpus
import time
t1 = time.time()

game = sys.argv[1]

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

MmCorpus.serialize('./outputs/corpus_'+game+'_Preview.mm', corpus)
joblib.dump(id2word, './outputs/dict_'+game+'_Preview.pkl')
# 'F' mean filtered with tfidf

e1 = time.time()
esp = (e1-t1)/3600
print('Process time: %.3f hour' % esp)