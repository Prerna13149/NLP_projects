# NLP_projects

P1 Summary
In this project, I build a system for automatically classifying song lyrics comments by era. I performed the following steps:
- Performed basic text processing, tokenized input and converted it into a bag-of-words representation
- Built a machine learning classifier based on the generative model, using Naive Bayes
- Evaluated the classifiers and examined what it learned
- Built a machine learning classifier based on the discriminative model, using Perceptron
- Built a logistic regression classifier using PyTorch
- Implement techniques to improve your classifier

P2 Summary
This project focuses on sequence labeling with Hidden Markov Models and Deep Learning models. The target domain is part-of-speech 
tagging on English and Norwegian from the Universal Dependencies dataset. 
I performed the following steps:
- Performed basic preprocessing of the data
- Built a naive classifier that tags each word with its most common tag
- Implemented a Viterbi Tagger using Hidden Markov Model in PyTorch
- Built a Bi-LSTM deep learning model using PyTorch
- Built a Bi-LSTM_CRF model using the above components (Viterbi and Bi-LSTM)
- Implement techniques to improve your tagger

P3 Summary
In this problem set, I implemented a deep transition dependency parser in PyTorch. 
I performed the following tasks:
- Implemented an arc-standard transition-based dependency parser in PyTorch
- Implemented neural network components for choosing actions and combining stack elements
- Trained network to parse English and Norwegian sentences
- Implemented techniques to improve your parser

P4 Summary
In this project, I ventured into the challenging NLP task of coreference resolution. 
I performed the following tasks:
- Implemented a simple rule-based system that achieve results which are surprisingly difficult to beat.
- Got acquainted with the trickiness of evaluating coref systems, and the current solutions in the field.
- Experimented with two neural approaches for coref to be implemented in PyTorch:
      - A feedforward network that only looks at boolean mention-pair features
      - A fully-neural architecture with embeddings all the way down
- Got a glimpse at domain adaptation in the wild, by trying to run a system trained on news against a narrative corpus and vice-versa.
