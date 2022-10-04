"""
This code first load the data created from TA_FA_EDA.py and extracts the following:
- All default firms in WRDS
- All MDAs related to default firms

"""

import time
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')

from plotting import *
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

plt.rcParams["font.family"] = "Calibri"
pd.set_option('use_inf_as_na', True)
f_type = [
          '10-KSB',
          '10KSB',  # SME
          # '10-K405',  # was used to indicate that an officer or director of a company failed to file a Form 4
          # (or similar Form 3 or Form 5) on time. Form 4, or similar Form 3 or Form 5,
          # are used to disclose insider trading activity.
          # '10KSB40',
          # '10-KT',  # Transition of accounting period
          # '10KT405',
          '10-K'
]

t1 = time.time()
# Load sample data and compute Z-Socre
dt = pd.read_csv('./full_sample.csv', index_col=0)
# dt.shape
dt['stalt'].value_counts()
dt = dt.loc[dt['stalt'] == 'TL', :]
dt.reset_index(drop=True, inplace=True)
dt['stalt'].value_counts()

# dt.shape
# dt.stalt
df = joblib.load('./data/senti/senti_topic_'+str(1997)+'.pkl')
df = df.loc[df['CIK'].isin(dt['cik']), :]
df.reset_index(drop=True, inplace=True)
for year in range(1998, 2019):
    # cores = int(sys.argv[2])
    df1 = joblib.load('./data/lda/infer_topic_' + str(year) + '.pkl')
    df1 = df1.loc[df1['CIK'].isin(dt['cik']), :]
    df1.reset_index(drop=True, inplace=True)
    print('Precessed %d' % year)
    df = df.append(df1)
    # sec_dt = sec_dt['sMDA'].copy()
    # sec_dt = None

joblib.dump(df, './data/smda_default_only.pkl')
print(df.shape)
print(df.columns)
print('Total sentences: %d' % (np.sum(df['sMDA'].map(len))))
print(df['sMDA'][0:10])
esp = (time.time() - t1)/3600
print('Finish in %.3f!'%esp)