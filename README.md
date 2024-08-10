# Estimator_Vectors
This library trains word embeddings on a corpus, along with context clue and subword embeddings jointly. The word embeddings are used for normal embeding tasks, and the context clue and subword embeddings are used for estimating out-of-vocabulary (OOV) word embeddings. This code is heavily based on the gensim code: https://radimrehurek.com/gensim/

For more information, see our IJCNN paper [Estimator Vectors: OOV Word Embeddings based on Subword and Context Clue Estimate](https://ieeexplore.ieee.org/abstract/document/9207711?casa_token=mwjGp1qnyZQAAAAA:Erjp2JT12fh0IIHja0z4Tfgv3HervlzxIPiD6Tv1dvaGulcuFJsRwa3aApMsgMebxcdzDq1P) by Raj Patel and Carlotta Domeniconi.

To use this code, you will need Cython and the gensim library. As mentioned earlier, this library is heavily based on gensim code, and a lot of the files pertain to them.

For usage, please refer to text8_example.py

The code is very rough right now, but I will improve it as time goes on.
