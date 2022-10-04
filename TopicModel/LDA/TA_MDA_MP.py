"""
Run the remove_stopwords in MDA using multithreading mode, not work in the low memory systems!
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from preprocessing import *

import joblib
from itertools import repeat

import multiprocessing as mp
from multiprocessing import Pool

import time
import sys

t1 = time.time()

year = int(sys.argv[1])


# Load data
sec_dt = joblib.load('./data/sec_dt_' + str(year) + '.pkl')
sec_dt = sec_dt['MDA']

# Multiprocessing
with Pool(processes=22) as pool:
    res = pool.starmap(remove_stopwords, zip(sec_dt['MDA'], repeat('spacy')))

# Assign new column
sec_dt['PMDA'] = pd.Series(res).values
joblib.dump(res, './outputs/test_MP_PMDA_' + str(year) + '.pkl')
# Check percentage of each not-found MDA
# no_MDA - 6
# omitted_MDA - 11
# uncommon_MDA - 12
# sec_dt.loc[sec_dt['LMDA'] <= 15, 'LMDA'].value_counts()*100/len(sec_dt)
#
# # Histogram of MDAs' length
# plt.rcParams["font.family"] = "Times New Roman"
# ax = sns.distplot(sec_dt.loc[sec_dt['LMDA'] > np.median(sec_dt['LMDA']), 'LMDA'],
#                   hist_kws={"linewidth": 3, "alpha": 0.55, "color": "g"})
#
# plt.yticks(ax.get_yticks(), ax.get_yticks() * 100)
# plt.ylabel('Distribution [%]', fontsize=16)
# plt.savefig('./outputs/stats/dist_len_MDA_'+str(year)+'.png')
# plt.close()

e1 = time.time()
print('Process time: %.3f hour'% (e1-t1)/3600)