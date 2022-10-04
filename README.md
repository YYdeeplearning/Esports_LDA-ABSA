# Topics & Sentiments Analysis on Esports

This project aims to analyze the reviews of four electronic sports (esports) games on Steam including TEKKEN7, Dota2, PUBG, and CS:GO, using a hybrid framework that combines topic modeling and sentiment analysis.

## Requirements for Esports_E2E-ABSA

* python 3.7.3
* pytorch 1.2.0
* transformers 4.1.1
* numpy 1.16.4
* tensorboardX 1.9
* tqdm 4.32.1

Esports_E2E-ABSA codes are mostly inspired by [**Li et al. 2019**](https://arxiv.org/abs/1910.00883)'s awesome work and borrowed from **BERT-E2E-ABSA** ([https://github.com/lixin4ever/BERT-E2E-ABSA](https://github.com/lixin4ever/BERT-E2E-ABSA)).

## Requirements for LDA

* python 3.7.3
* gensim 4.0.1
* spacy 3.0.6

  ```bash
  python -m spacy download en_core_web_sm
  ```

* scipy 1.6.2

## Dataset

We provide a preprocessed and annotation dataset including four esports games up to December 2021. You can find them accompany with model folders. The original data was collected by using **Steam API** ([https://partner.steamgames.com/doc/store/getreviews](https://partner.steamgames.com/doc/store/getreviews)).

## Environment

* OS: Ubuntu 20.04.2 LTS
* GPU: NVIDIA A40
* CUDA: 10.0
* cuDNN: v7.6.1
