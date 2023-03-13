   Replication of  //Parikh, R., Dupuy, C., & Gupta, R. (2022). Canary Extraction in Natural Language Understanding Models. arXiv preprint arXiv:2203.13920. //

Target model: Ma, X., & Hovy, E. (2016). End-to-end sequence labeling via bi-directional lstm-cnns-crf. arXiv preprint arXiv:1603.01354.
	model code is originated from https://github.com/guillaumegenthial/tf_ner/tree/master/models/chars_conv_lstm_crf



1.
Follow the data/example and add your data to data/your_data for instance.

For name in {train, testa, testb}, create files {name}.words.txt and {name}.tags.txt that contain one sentence per line, each word / tag separated by space. I recommend using the IOBES tagging scheme.


2.
run canary_insertion in data directory with pattern name, n, R
ex) python canary_insertion.py call 4 10

3.
Create files vocab.words.txt, vocab.tags.txt and vocab.chars.txt that contain one lexeme per line.
run build_vocab.py with target directory name
ex) python build_vocb.py call-4-10

4.
Create a glove.npz file containing one array embeddings of shape (size_vocab_words, 300) using GloVe 840B vectors and np.savez_compressed.
run build_glove.py with target directory name
ex) python build_glove.py call-4-10

5.
Train model by running main.py in model dir.
ex) python main2.py call 4 10

6.
Analyze reulst with analyze_results.py.