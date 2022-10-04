"""
Main code for running online-learning LDA gensim with serialized corpus using 32 threads of VPCC.
Should be run after TA_MDA_serialize_tfidf.py
"""
import joblib
from itertools import repeat

import multiprocessing as mp
from multiprocessing import Pool
import time
import sys

from gensim.models import LdaModel, TfidfModel
from gensim import corpora
from gensim.corpora import MmCorpus

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def lda_mp(corpus, id2word, num_topics):
    chunksize = 1000
    passes = 10
    iterations = 100
    decay = 0.5
    offset = 64
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    RANDOM_STATE = 42
    return LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize,
                    alpha='asymmetric', eta='auto', minimum_probability=0,
                    iterations=iterations, num_topics=num_topics,
                    passes=passes, eval_every=eval_every,
                    decay=decay, offset=offset, random_state=RANDOM_STATE
                    )


if __name__ == '__main__':
    print(mp.cpu_count())
    game = sys.argv[1]
    cores = int(sys.argv[2])
    min_topics = int(sys.argv[3])
    max_topics = int(sys.argv[4])
    with Pool(processes=cores) as pool:
        # ######1st approach: load the original corpus vector######
        # MDAs = joblib.load('./data/sec_dt_PMDA_' + str(year) + '.pkl')
        # MDAs = MDAs['PMDA']
        # # MDAs = joblib.load('./data/test1kMDA.pkl')
        # id2word = corpora.Dictionary(MDAs)
        # corpus = [id2word.doc2bow(sent) for sent in MDAs]
        # MDAs = None

        # ######2nd approach: load the mm corpus vector######
        corpus = MmCorpus('./outputs/corpus_PMDA_' + str(year) + '.mm')
        id2word = joblib.load('./outputs/dict_PMDA_' + str(year) + '.pkl')

        start_time = time.time()
        # start worker processes
        res = pool.starmap(lda_mp, zip(repeat(corpus), repeat(id2word), range(min_topics, max_topics+1)))
        # list(zip(itertools.repeat(0),range(1,10),itertools.repeat(12)))
        # multiple_results = [pool.apply_async(f, (corpus, id2word, i, chunksize)) for i in range(10, 18)]
        # multiple_results = [pool.apply_async(os.getpid, ()) for i in range(8)]
        # print([res.get(timeout=1) for res in multiple_results])
    # joblib.dump(multiple_results, './outputs/lda/testmp.pkl')
    joblib.dump(res, './outputs/lda/ldamp_'+game+'_'+str(min_topics)+'_'+str(max_topics)+'.pkl')
    joblib.dump(id2word, './outputs/lda/ldamp_'+game+'_'+str(min_topics)+'_'+str(max_topics)+'_id2word.pkl')
    joblib.dump(corpus, './outputs/lda/ldamp_'+game+'_'+str(min_topics)+'_'+str(max_topics)+'_corpus.pkl')
    # exiting the 'with'-block has stopped the pool
    esp = (time.time()-start_time)/3600
    print("USING %d CORES FOR RUNNING ONLINE-LEARNING LDA.\n", cores)
    print("FILLING YEARS: %d \n", year)
    print("MIN TOPIC: %d\n", min_topics)
    print("MAX TOPIC: %d\n", max_topics)
    print("TIME: %.3f hours." % esp)



# res = joblib.load('./outputs/lda/testmp.pkl')