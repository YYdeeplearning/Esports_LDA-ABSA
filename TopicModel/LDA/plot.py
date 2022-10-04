import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from gensim.corpora import MmCorpus, Dictionary
from wordcloud import WordCloud
import sys

def plot_wc(model, game):
    sw = ['fuck', 'shit', 'cyka', 'blyat']
    sw = set(sw)
    corpus, id2word = read_review(game)
    lda = model
    lda_corpus = lda[corpus]
    wc = WordCloud(background_color="white", stopwords=sw, collocations=False)  # , max_words=1000, mask=alice_mask)
    print('Plotting wordcloud for LDA model with: {} topics'.format(lda.num_topics))        
    for t in range(lda.num_topics):
        plt.figure()
        plt.imshow(wc.fit_words(dict(lda.show_topic(t, 30))))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.savefig('./outputs/lda/wordcloud_' + game + "_Topic_" + str(t) + '.png')
        # plt.show()
        plt.close()


def read_review(game):
    corpus = MmCorpus('./outputs/corpus_' + game + '_Preview.mm')
    id2word = joblib.load('./outputs/dict_' + game + '_Preview.pkl')
    return corpus, id2word


game = sys.argv[1]


''' perplexity '''
perplexity_list = []

for i in range(3, 101):
    perplexity_list.append(joblib.load('./outputs/lda/perplexity_value_'+game+'_'+str(i)+'.pkl'))

plt.plot(range(3, 101), perplexity_list)
plt.title('perplexity_' + game)
plt.savefig('./outputs/lda/Perplexity_' + game + '.png')
plt.close()

''' num_topics '''
coherence_list = []

for i in range(3, 101):
    coherence_list.append(joblib.load('./outputs/lda/coherence_value_'+game+'_'+str(i)+'.pkl'))

num_topics = coherence_list.index(min(coherence_list))+3

''' wordcloud '''

model = joblib.load('./outputs/lda/lda_train_'+game+'_'+str(num_topics)+'.pkl')  # lda_mp_all_year_10k5epoch1039
plot_wc(model, game)

''' coherence_score '''
    
plt.plot(range(3, 101), coherence_list)
plt.title('coherence_'+ game)
plt.savefig('./outputs/lda/Coherence_{}_{}.png'.format(game, num_topics))
plt.close()


''' wordcloud_corpus '''

wordcloud_list = []

for i in range(3, 101):
    model = joblib.load('./outputs/lda/lda_train_{}_{}.pkl'.format(game, i))
    wordcloud_list.append(model)

with open('./outputs/lda/{}_wordcloud_corpus.txt'.format(game),'w+',encoding='utf-8') as f:
    for lda_list in wordcloud_list:
        lda_dict = dict(lda_list.show_topic(0,50))
        f.write(str(lda_dict))
        f.write("\n\n")