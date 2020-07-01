import math
import EstimatorVecs
from EstimatorVecs.models.EstimatorVecs import EstimatorVecs

import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec


print('Training')

#load corpus
sentences = word2vec.Text8Corpus('text8/text8')
	
#initialize and train model
model = EstimatorVecs(sentences, size=300, alpha=.025, window=5, max_vocab_size=None, min_count=5, sg=1, hs=0,negative=5, sample=.0001,iter=10, workers=8, cc_sampling=0, pairs=4, subword_type='ngram')


#save each piece
model.save('est_vecs_model')
model.word.save_word2vec_format('est_vecs_word')
model.cc.save_word2vec_format('est_vecs_context_clues')
model.sub.save_word2vec_format('est_vecs_sub')
	
print('training has finished')

#Usage

#Load Vectors
word_emb = gensim.models.KeyedVectors.load_word2vec_format('est_vecs_word', binary=False)
cc = gensim.models.KeyedVectors.load_word2vec_format('est_vecs_context_clues', binary=False)
subword = gensim.models.KeyedVectors.load_word2vec_format('est_vecs_sub', binary=False)



#Estimate OOV Word Embeddings
