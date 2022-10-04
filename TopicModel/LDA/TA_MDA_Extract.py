"""
This code reads and merges all filings in a particular year, and extracts the MDA.
year is defined in the start-stop range
Type of filings are defined in f_type variable
Output a data frame with following columns:
- CIK
- FORM_TYPE
- filing_DATE
- CONFORMED_DATE
- PROCESSED_TEXT: the contents on filings
- TEXT_FILE: the link of corresponding filing
- MDA: the MDA section of filing
- LMDA: length in number of characters of each MDA
"""

import zipfile
import matplotlib
matplotlib.use('Agg')

import time
from preprocessing import *


t1 = time.time()
start = int(sys.argv[1])
stop = int(sys.argv[2])
for year in range(start, stop+1):
    f_type = [
              '10-K',
              '10KSB',  # SME
              # '10-K405',  # was used to indicate that an officer or director of a company failed to file a Form 4
              # (or similar Form 3 or Form 5) on time. Form 4,
        #  or similar Form 3 or Form 5,
              # are used to disclose insider trading activity.
              # '10KSB40',
              # '10-KT',  # Transition of accounting period
              # '10KT405',
              '10-KSB']

    # stops = get_dic_list(d_l)

    # Load filings from zip files
    z_l = zipfile.ZipFile('./data/SEC/EDGAR_10X_C_' + str(year)+'.zip', 'r')  # zip files list
    r_l = z_l.namelist()  # report files list

    # find cik, filing date, and report date in the report file name
    cik = []
    f_date = []
    # More info: https://www.sec.gov/info/edgar/dissemination/appxb.txt
    r_type = []
    list_file = []
    list_cpr = []  # Conformed period of report
    list_pt = []  # processed reports
    for r in r_l:
        if get_ftype(r) in f_type:
            report = z_l.read(r).decode("utf-8")
            list_file.append(r)
            cik += [r[r.find('_', r.find('edgar_data_')+10)+1:r.find('_', r.find('edgar_data_')+11)]]
            # r_type += [r[r.find('_', r.find('10-X_C')+5)+1:r.find('_', r.find('_', r.find('10-X_C')+5)+1)]]
            r_type += [r[r.find('_', r.find('QTR'))+1:r.find('_', r.find('_', r.find('QTR')) + 2)]]
            f_date += [r[r.find('/', r.find('QTR'))+1:r.find('_', r.find('QTR'))]]
            cpr = get_cpr(report)
            list_cpr.append(cpr)
            list_pt.append(" ".join(report.split()))

    zippedList = list(zip(cik, r_type, f_date, list_cpr, list_pt, list_file))
    sec_dt = pd.DataFrame(zippedList,
                          columns=['CIK', 'FORM_TYPE', 'filing_DATE', 'CONFORMED_DATE', 'PROCESSED_TEXT', 'TEXT_FILE'])

    # Explore
    sec_dt.head()
    sec_dt['FORM_TYPE'].value_counts()

    # sec_dt.isna().sum()
    sec_dt = sec_dt.loc[~(sec_dt['filing_DATE'] == '')]
    sec_dt['CIK'] = sec_dt['CIK'].apply(int)
    sec_dt['filing_DATE'] = sec_dt['filing_DATE'].apply(int)

    # EXTRACT MDA AND COMPUTE LENGTH
    sec_dt['MDA'] = sec_dt['PROCESSED_TEXT'].map(extract_mda_sents)

    sec_dt['LMDA'] = sec_dt['MDA'].map(len)

    # save data:
    joblib.dump(sec_dt, './data/sec_dt_' + str(year) + '.pkl')
    #
    # # Load data
    # # sec_dt = joblib.load('./data/sec_dt_' + str(year) + '.pkl')
    # sec_dt['PMDA'] = sec_dt['MDA'].apply((lambda x: remove_stopwords(x, lemma='spacy')))
    # joblib.dump(sec_dt, './outputs/sec_dt_PMDA_' + str(year) + '.pkl')
    #
    # # Create and save corpus and dict
    # id2word = corpora.Dictionary(sec_dt['PMDA'])
    # corpus = [id2word.doc2bow(sent) for sent in sec_dt['PMDA']]
    #
    # MmCorpus.serialize('./outputs/corpus_PMDA_' + str(year) + '.mm', corpus)
    # joblib.dump(id2word, './outputs/dict_PMDA_' + str(year) + '.pkl')
    #
    # # Check percentage of each not-found MDA
    # # no_MDA - 6
    # # omitted_MDA - 11
    # # uncommon_MDA - 12
    # sec_dt.loc[sec_dt['LMDA'] <= 15, 'LMDA'].value_counts()*100/len(sec_dt)
    #
    # # Histogram of MDAs' length
    # plt.rcParams["font.family"] = "Times New Roman"
    # ax = sns.distplot(sec_dt.loc[sec_dt['LMDA'] > np.median(sec_dt['LMDA']), 'LMDA'],
    #               hist_kws={"linewidth": 3, "alpha": 0.55, "color": "g"})
    #
    # plt.yticks(ax.get_yticks(), ax.get_yticks() * 100)
    # plt.ylabel('Distribution [%]', fontsize=16)
    # plt.savefig('./outputs/stats/dist_len_MDA_'+str(year)+'.png')
    # plt.close()
    #
    # ax = sns.distplot(sec_dt['LMDA'],
    #                   hist_kws={"linewidth": 3, "alpha": 0.55, "color": "b"})
    #
    # plt.yticks(ax.get_yticks(), ax.get_yticks() * 100)
    # plt.ylabel('Distribution [%]', fontsize=16)
    # plt.savefig('./outputs/stats/dist_len_MDA_1' + str(year) + '.png')
    # plt.close()
    #
    # e1 = time.time()
    # print('Process time: %.3f hour'% (e1-t1)/3600)