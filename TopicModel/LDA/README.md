# Esports_LDA


From Dr. Nguyen Ba Hung


Please put PBS_run_LDA.sh series into parent directory

To run this script, you need to add two empty directory
'./Esports_data/';
'./outputs/lda/'

and rename your data file as "[Game_name]_csv.csv" into './Esports_data/Collected_Review/'


Package you need:

gensim
```
pip install --upgrade gensim
pip install python-Levenshtein(avoid warning)
```
spacy
```
pip install -U pip setuptools wheel
pip install -U spacy

python -m spacy download en_core_web_sm
```
scipy
```
pip install scipy
```
