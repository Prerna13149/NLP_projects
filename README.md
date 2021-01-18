# NLP_projects

P1 Summary <br />
In this project, I build a system for automatically classifying song lyrics comments by era. I performed the following steps: <br />
- Performed basic text processing, tokenized input and converted it into a bag-of-words representation <br />
- Built a machine learning classifier based on the generative model, using Naive Bayes <br />
- Evaluated the classifiers and examined what it learned <br />
- Built a machine learning classifier based on the discriminative model, using Perceptron <br />
- Built a logistic regression classifier using PyTorch <br />
- Implement techniques to improve your classifier <br />

P2 Summary <br />
This project focuses on sequence labeling with Hidden Markov Models and Deep Learning models. The target domain is part-of-speech <br />
tagging on English and Norwegian from the Universal Dependencies dataset. <br />
I performed the following steps: <br />
- Performed basic preprocessing of the data <br />
- Built a naive classifier that tags each word with its most common tag <br />
- Implemented a Viterbi Tagger using Hidden Markov Model in PyTorch <br />
- Built a Bi-LSTM deep learning model using PyTorch <br />
- Built a Bi-LSTM_CRF model using the above components (Viterbi and Bi-LSTM) <br />
- Implement techniques to improve your tagger <br />

P3 Summary <br />
In this problem set, I implemented a deep transition dependency parser in PyTorch. <br />
I performed the following tasks: <br />
- Implemented an arc-standard transition-based dependency parser in PyTorch <br />
- Implemented neural network components for choosing actions and combining stack elements <br />
- Trained network to parse English and Norwegian sentences <br />
- Implemented techniques to improve your parser <br />

P4 Summary <br />
In this project, I ventured into the challenging NLP task of coreference resolution. <br />
I performed the following tasks: <br />
- Implemented a simple rule-based system that achieve results which are surprisingly difficult to beat. <br />
- Got acquainted with the trickiness of evaluating coref systems, and the current solutions in the field. <br />
- Experimented with two neural approaches for coref to be implemented in PyTorch: <br />
      - A feedforward network that only looks at boolean mention-pair features <br />
      - A fully-neural architecture with embeddings all the way down <br />
- Got a glimpse at domain adaptation in the wild, by trying to run a system trained on news against a narrative corpus and vice-versa. <br />
