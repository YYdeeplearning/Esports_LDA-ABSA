"""
This script will process the raw report text data:
Input: Raw text data as string
Functions:
- get_ftype: return report type: 10K, 10KSB, 10Q...
- get_cpr: return the conformed period of the report (the true date of report, not filling date)
- get_dic_list: return the dictionary list for stopwords removing, for financial texts
- Processing text data:
    + process_raw: return the main text, excluding the exhibitions of the input report
    + split_into_sentences: split the input main text to sentences, considering financial terms and decimal numbers.
    + create_item_list: create possible item list to find MDA in the report. MDA could be from item 7 to item 8 in 10K
                        or from item 6 to item 7 in 10KSB
    + extract_mda_sents: extract MDA from 10K
    + extract_mda_sents_ksb: extract MDA from 10KSB
    + find_loc: find possible locations (indexes in report) from two input location lists (beginning and end indexes)
    + test_extract_MDA: test the returned MDA, check for possible length, correct item positions...
"""
import joblib
import pickle as pkl
from nltk.corpus import stopwords
import pandas as pd
import sys
import re
import spacy
# from TA_tagger import ConsecutivePosTagger


# Dictionary list for stopwords removing
d_l = ['Auditor',
       'Currencies',
       'DatesandNumbers',
       'Generic',
       # 'GenericLong',
       'Geographic',
       'Names']


caps = "([A-Z])"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt|Corp|bldg|dept|Dept|mfg)[.]"
websites = "[.](com|net|org|io|gov|me|edu)"
digits = "([0-9])"
alphabets = "([A-Za-z])"


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


stop_words = stopwords.words('english')
# with open('../outputs/list_dict.pkl', 'rb') as fp:
#     stops = pkl.load(fp)

newStopWords = ['cs', 'paw', 'piece', 'tbh', 'man', 'install', 'thiccdude', 'gut', 'gucci', 'kid', 'b+2,1~1', 'chicken', 'lil', 'doo', 'yea', 'goog', 'dota', 'steam', 'lili', 'kappa', 'fortnitei', 'roblox',
                'ching', 'depression', 'lar', 'kool', 'goood', 'pump', 'hoe', 'life', 'league', 'crust', 'gam', 'nut', 'day', 'ᴛʜᴇ', 'nugget', 'game', 'mess', 'ok', 'hour', 'star', 'prox', 'ᴀɴᴅ','ʏᴏᴜ',
                'gaming', 'camper', 'second', 'yeet', 'ravioli', 'gamer', 'marshall', 'baby', 'way', 'gg', 'hahahaha', 'lit', 'realy', 'dinner', 'gud', 'minecraft',  
                'frame', 'guy', 'cancer', '-->uf+4', 'crate?where', 'dope', 'fortnite', 'mom', 'dogg', 'garbage', 'ig', 'gr8', '2kin', 'kuku', 'waste', 'hahahahahahahahahahahahahahahahahahahahahahahahahahahahahaha', 
                'bla', 'royale', 'pie', 'dis', 'system', 'world', 'lock', 'atm', 'thing', 'racistminute', 'coool', 'hate', 'time', 'trihard7', 'kak', 'gmae', 'aight', 'butter',  
                'veryvery', 'numba', 'chong', 'noice', 'regionlockchina', 'soul', 'legend', 'love', 'luv', 'boy', 'access', 'fight', 'addicting', 'gaben', 'trash', 'cheese', 'sandwich', 'ur',
                'bit', 'lot', 'obama', 'name', 'play', 'battleye', 'nmsl', 'word', 'refund', 'float', 'year', 'hole', 'girlfriend', 'valve', 'speech', 'alot', 'di',
                'suck', 'key', ':)', 'ggwp', 'yee', 'strike', 'gaem', 'crate', 'pubg', 'bob', 'glitch', 'bro', 'goodgame', 'god', 'fav', 'hwoarang', 'winner', '|||', 'tekken', 'hang', 'blablabla'
                'racist', 'ight', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'hahaha', 'gb', 'valorant', 'ffs', 'csgo', 'tf2', 'blyat', 'gang', '-25', 'cyka', 'lol', 'dog']
                


stop_words.extend(newStopWords)

def get_ftype(r):
    """
    Get report type
    :param r: input report text from LM
    :return: type of report: K, Q, ...
    """
    return r[r.find('_', r.find('QTR'))+1:r.find('_', r.find('_', r.find('QTR')) + 2)]


def get_cpr(t):
    """
    :param t: raw text data
    :return:
    - cpr: conformed period of report
    - p_t: the processed text data
    """
    # Conformed period of report
    try:
        cpr = pd.to_datetime(t[t.find('CONFORMED PERIOD OF REPORT:') + 28:
                             t.find('CONFORMED PERIOD OF REPORT:') + 36], format='%Y%m%d')
    except:
        cpr = 'MISSING'
    return cpr


def get_dic_list(d_l):
    """
    :param d_l: dictionary list, should be a list of name
    :return: a set of words
    """
    dict_list = []
    for d in d_l:
        dict = './data/StopWords_' + d + '.txt'
        try:
            with open(dict) as inputFileHandle:
                for line in inputFileHandle:
                    dict_list.append(line.split(None, 1)[0])  # add only first word
        except IOError:
            sys.stderr.write("Error: Could not open %s\n" % dict)
            sys.exit(-1)
    return set(dict_list)


def remove_stopwords(sentence, lemma=None):
    """
    stop_word is the combination of common stopwords from nltk and LM dictionary
    consider the capital stopword from LM, and remove the remaining punctuation
    :param sentence
    :param lemma:
                    None for using all words,
                    spacy for builtin SpaCy lemmatizer, and
                    in-house for custom lemmatizer

    :return: list of word
    """
    if lemma == 'spacy':
        sentence = lemmatization(sentence)
        return [word for word in sentence if word not in stop_words and len(word) > 1]
    elif lemma == 'in-house':
        tagger_svm = joblib.load('./outputs/tagger/tagger_svm.pkl')
        return [token[0] for token in list(tagger_svm.tag(sentence.split())) if token[1] in ['NN', 'NNP', 'NNS', 'NP']]
    else:
        if len(sentence.split()) >= 5:
            return [word for word in sentence.split() if word not in stop_words and len(word) > 1]


def lemmatization(sentence, allowed_postags=['NOUN', 'PROPN']):
    """https://spacy.io/api/annotation"""
    doc = nlp(sentence)
    texts_out = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return texts_out


def process_raw(t, stops):
    """
    :param t: raw text data
    :param stops: stopword lists, should be a set for speed
    :return:
    - p_t: the processed text data
    """
    t = t[(t.find('ITEM 1') + 8): t.find('<EX-31.1>')]
    p_t = ' '.join(word for word in t.split() if (len(word) > 2) and (word not in stops))
    return p_t


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = text.replace("/s/", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)

    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)  # Decimal
    text = re.sub("[.]" + digits, "<prd>\\1", text)  # Decimal .01%
    if "No. " in text: text = text.replace("No. ", "No<prd>")

    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    if "..." in text: text = text.replace("...", "<prd><prd><prd>")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s for s in sentences if len(s) > 4]
    return sentences


def create_item_list(number_item):
    """
    return the list of possible item flags to match for
    :param number_item: item 6 or item 7 to start with, depend on report type. 10KSB (item 6&7) or 10-K (item 7&8)
    :param name_item: possible name for item, including . or - or grammar mistakes
    :return: item_start, item_end for corresponding MDA in 10-K or 10KSB
    """

    item_start = dict()
    item_end = dict()

    item_start[1] = "item 7\. management's discussion and analysis"
    item_start[2] = "item 7\.management's discussion and analysis"
    item_start[3] = "item7\. management's discussion and analysis"
    item_start[4] = "item7\.management's discussion and analysis"
    item_start[5] = "item 7\. management discussion and analysis"
    item_start[6] = "item 7\.management discussion and analysis"
    item_start[7] = "item7\. management discussion and analysis"
    item_start[8] = "item7\.management discussion and analysis"
    item_start[9] = "item 7 management's discussion and analysis"
    item_start[10] = "item 7management's discussion and analysis"
    item_start[11] = "item7 management's discussion and analysis"
    item_start[12] = "item7management's discussion and analysis"
    item_start[13] = "item 7 management discussion and analysis"
    item_start[14] = "item 7management discussion and analysis"
    item_start[15] = "item7 management discussion and analysis"
    item_start[16] = "item7management discussion and analysis"
    item_start[17] = "item 7: management's discussion and analysis"
    item_start[18] = "item 7:management's discussion and analysis"
    item_start[19] = "item_start: management's discussion and analysis"
    item_start[20] = "item7:management's discussion and analysis"
    item_start[21] = "item 7: management discussion and analysis"
    item_start[22] = "item 7:management discussion and analysis"
    item_start[23] = "item7: management discussion and analysis"
    item_start[24] = "item7:management discussion and analysis"
    item_start[25] = "item 7\. plan of operation"
    item_start[26] = "item 7: plan of operation"
    item_start[27] = "item 7 - plan of operation"
    item_start[28] = "item 7- plan of operation"
    item_start[29] = "item 7 - management's discussion and analysis"
    item_start[30] = "item 7 - management discussion and analysis"
    item_start[31] = "item 7\. management s discussion and analysis"
    item_start[32] = "item 7 management s discussion and analysis"
    item_start[33] = "item 7 - management s discussion and analysis"
    item_start[34] = "item 7- management s discussion and analysis"
    item_start[35] = "item 7\. management's discussion analysis"
    item_start[36] = "item 7 - management's discussion analysis"
    item_start[37] = "item 7- management's discussion analysis"
    item_start[38] = "item 7 management's discussion analysis"
    item_start[39] = "item 7: management s discussion and analysis"
    item_start[40] = "item 7\. managements discussion and analysis"
    item_start[41] = "item 7\. management's plan of operation"
    item_start[42] = "item 7\. managements' discussion and analysis"
    item_start[43] = "item 7 \. management's discussion and analysis"
    item_start[44] = "item 7 -- management's discussion and analysis"
    # item_start[39] = "management's discussion and analysis"
    # item_start[40] = "management's discussion analysis"

    item_end[1] = "item 8\. financial statements"
    item_end[2] = "item 8\.financial statements"
    item_end[3] = "item8\. financial statements"
    item_end[4] = "item8\.financial statements"
    item_end[5] = "item 8 financial statements"
    item_end[6] = "item 8financial statements"
    item_end[7] = "item8 financial statements"
    item_end[8] = "item8financial statements"
    item_end[9] = "item 8a\. financial statements"
    item_end[10] = "item 8a\.financial statements"
    item_end[11] = "item8a\. financial statements"
    item_end[12] = "item8a\.financial statements"
    item_end[13] = "item 8a financial statements"
    item_end[14] = "item 8afinancial statements"
    item_end[15] = "item8a financial statements"
    item_end[16] = "item8afinancial statements"
    item_end[17] = "item 8\. consolidated financial statements"
    item_end[18] = "item 8\.consolidated financial statements"
    item_end[19] = "item8\. consolidated financial statements"
    item_end[20] = "item8\.consolidated financial statements"
    item_end[21] = "item 8 consolidated  financial statements"
    item_end[22] = "item 8consolidated financial statements"
    item_end[23] = "item8 consolidated  financial statements"
    item_end[24] = "item8consolidated financial statements"
    item_end[25] = "item 8a\. consolidated financial statements"
    item_end[26] = "item 8a\.consolidated financial statements"
    item_end[27] = "item8a\. consolidated financial statements"
    item_end[28] = "item8a\.consolidated financial statements"
    item_end[29] = "item 8a consolidated financial statements"
    item_end[30] = "item 8aconsolidated financial statements"
    item_end[31] = "item8a consolidated financial statements"
    item_end[32] = "item8aconsolidated financial statements"
    item_end[33] = "item 8\. audited financial statements"
    item_end[34] = "item 8\.audited financial statements"
    item_end[35] = "item8\. audited financial statements"
    item_end[36] = "item8\.audited financial statements"
    item_end[37] = "item 8 audited financial statements"
    item_end[38] = "item 8audited financial statements"
    item_end[39] = "item8 audited financial statements"
    item_end[40] = "item8audited financial statements"
    item_end[41] = "item 8: financial statements"
    item_end[42] = "item 8:financial statements"
    item_end[43] = "item8: financial statements"
    item_end[44] = "item8:financial statements"
    item_end[45] = "item 8: consolidated financial statements"
    item_end[46] = "item 8:consolidated financial statements"
    item_end[47] = "item8: consolidated financial statements"
    item_end[48] = "item8:consolidated financial statements"
    item_end[49] = "item 8 - financial statements"
    item_end[50] = "item 8- audited financial statements"
    item_end[51] = "item 8-financial statements"
    item_end[52] = "item 8 - index to financial statements"
    item_end[53] = "item 8. index to financial statements"
    item_end[54] = "item 8. index to consolidated financial statements"
    item_end[55] = "i tem 8. financial statements"
    item_end[56] = "i tem 8 . financial statements"
    item_end[57] = "item 8 -- financial statements"
    item_end[56] = "item 8 . financial statements"

    if number_item == 7:
        return item_start, item_end
    if number_item == 6:
        for i in range(1, len(item_start)+1):
            item_start[i] = item_start[i].replace('7', '6')
        for j in range(1, len(item_end)+1):
            item_end[j] = item_end[j].replace('8', '7')
        return item_start, item_end


def extract_mda_sents(s):
    item_7, item_8 = create_item_list(7)
    look = {" refer to ", " included in ", " contained in "}
    min_lookback = 50  # number of characters to look back in case of references
    min_word = 30  # number of minimum words in MDA

    locs_i7 = {}
    locs_i8 = {}

    lower_str = " ".join(s.lower().split())
    for j in range(1, len(item_7)+1):
        locs_i7[j] = []
        for m in re.finditer(item_7[j], lower_str):
            if not m:
                break
            else:
                substr1 = lower_str[m.start() - 20:m.start()]
                if not any(s in substr1 for s in look):
                    # print substr1
                    b = m.start()
                    locs_i7[j].append(b)

    list_i7 = []
    for value in locs_i7.values():
        for thing1 in value:
            list_i7.append(thing1)
    list_i7.sort()
    list_i7.append(len(lower_str))

    for j in range(1, len(item_8)+1):
        locs_i8[j] = []
        for m in re.finditer(item_8[j], lower_str):
            if not m:
                break
            else:
                substr1 = lower_str[(m.start() - min_lookback):m.start()]
                if not any(s in substr1 for s in look):
                    b = m.start()
                    locs_i8[j].append(b)
    list_i8 = []
    for value in locs_i8.values():
        for thing2 in value:
            list_i8.append(thing2)
    list_i8.sort()

    if list_i8 == []:
        # Before returning the KSB search, i8 could be omitted or i7 and i8 are wrongly numbered,
        # try to search for the next item. CAREFULLY check for item in table of contents,
        if len(list_i7) == 2:
            temp = s[list_i7[0]:len(s)].lower()
            list_i8.append(list_i7[0] + temp[250:len(temp)].find("item"))
            locations = find_loc(list_i7, list_i8)
        else:
            return extract_mda_sents_ksb(s)
    else:
        if list_i7 == []:
            return extract_mda_sents_ksb(s)
        else:
            locations = find_loc(list_i7, list_i8)

    if len(locations) == 0:
        if len(list_i7) > 1:
            return "uncommon_MDA"
        else:
            return extract_mda_sents_ksb(s)
    else:
        list_mdas = []
        count = 0
        for k0 in range(len(locations)):
            mda = s[locations[k0][0]:locations[k0][1]]
            if len(mda.split()) > min_word:
                count += 1
                list_mdas.append(mda)
        if count == 0:
            return "omitted_MDA"
        else:
            return max(list_mdas)
    
    
def extract_mda_sents_ksb(s):
    item_6, item_7 = create_item_list(6)

    look = {" refer to ", " included in ", " contained in "}
    min_lookback = 50  # number of characters to look back in case of references
    min_word = 30  # Minimum number of words in MDA
    locs_i6 = {}
    locs_i7 = {}

    lower_str = " ".join(s.lower().split())
    for j in range(1, len(item_6)+1):
        locs_i6[j] = []
        for m in re.finditer(item_6[j], lower_str):
            if not m:
                break
            else:
                lookback_text = lower_str[(m.start() - min_lookback):m.start()]
                if not any(s in lookback_text for s in look):
                    # print lookback_text
                    b = m.start()
                    locs_i6[j].append(b)

    list_i6 = []
    for value in locs_i6.values():
        for thing1 in value:
            list_i6.append(thing1)
    list_i6.sort()
    list_i6.append(len(lower_str))

    for j in range(1, len(item_7)+1):
        locs_i7[j] = []
        for m in re.finditer(item_7[j], lower_str):
            if not m:
                break
            else:
                lookback_text = lower_str[(m.start() - min_lookback):m.start()]
                if not any(s in lookback_text for s in look):
                    # print lookback_text
                    b = m.start()
                    locs_i7[j].append(b)
    list_i7 = []
    for value in locs_i7.values():
        for thing2 in value:
            list_i7.append(thing2)
    list_i7.sort()

    if list_i7 == []:
        # Before returning the NO_MDA, i7 could be omitted, try to search for the next item. CAREFULLY
        if len(list_i6) == 2:
            temp = s[list_i6[0]:len(s)].lower()
            list_i7.append(list_i6[0]+temp[250:len(temp)].find("item"))
            locations = find_loc(list_i6, list_i7)
        else:
            return "NO_MDA"
    else:
        if list_i6 == []:
            return "NO_MDA"
        else:
            locations = find_loc(list_i6, list_i7)

    if len(locations) == 0:
        if len(list_i6) > 1:
            return "uncommon_MDA"
        else:
            return 'NO_MDA'
    else:
        list_mdas = []
        count = 0
        for k0 in range(len(locations)):
            mda = s[locations[k0][0]:locations[k0][1]]
            if len(mda.split()) > min_word:
                count += 1
                list_mdas.append(mda)
        if count == 0:
            return "omitted_MDA"
        else:
            return max(list_mdas)


def find_loc(l1, l2):
    """
    find the location of interested text
    :param l1: list of start positions
    :param l2: list of end positions
    :return: range of position
    """
    locations = dict()
    for k0 in range(len(l1)):
        locations[k0] = []
        locations[k0].append(l1[k0])
    for k0 in range(len(locations)):
        for item in range(len(l2)):
            if locations[k0][0] <= l2[item]:
                locations[k0].append(l2[item])
                break
        if len(locations[k0]) == 1:
            del locations[k0]
    return locations


def test_extract_MDA(list_omit, dt, type):
    """
    test extract MDA function
    :param list_omit: the current list of test index
    :param dt: the list of reports
    :param type: omit or length
    :return: index and file name of problem report
    """
    for i in range(list_omit[-1] + 1, len(dt) - 1):
        report = dt[i]
        if type == 'length':
            if len(extract_mda_sents(report)) >= 100000 and (len(extract_mda_sents(report))/len(report) > 0.5):
                break
        if type == 'omit':
            # Check for other unusual MDA here
            if extract_mda_sents(report) == 'omitted_MDA':
                if len(dt[i]) >= 10000:
                    break
    print(i, dt[i][31:150])