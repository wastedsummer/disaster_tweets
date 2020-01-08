import torch
# import json
# import feedparser
import datetime
import os
import json
import pickle

import numpy as np
# import re
# import pickle
# import math_utils

# from _config import PATH
# from feed_utils import FeedEntry, clear_summary
from models import InferSent
import pandas



# now = datetime.datetime.now()




# load infersent model
model_version = 2
MODEL_PATH = 'encoder/infersent%s.pkl' % model_version
hyperparameters = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(hyperparameters)
model.load_state_dict(torch.load(MODEL_PATH))
use_cuda = False
model = model.cuda() if use_cuda else model
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)
model.build_vocab_k_words(K=10000) #10000




train_df = pandas.read_csv('/home/stefan/projects/disaster_tweets/train.csv')

tweets = train_df.text.to_list()

embeddings = model.encode(tweets)
# train_df['embedding'] = [np.zeros(4096) for i in range(train_df.shape[0])]

embeddings_list = [embeddings[x] for x in range(embeddings.shape[0])]


# train_df['embedding'] = embeddings_list
# print(train_df.head())
# for i in range(train_df.shape[0]):
# # for i in range(10):
# 	tweet_text = train_df['text'][i]
# 	tweet_embedding = model.encode(tweet_text)


# train_df.to_csv('train_w_embeddings.csv')



train_dict = {}
for i in range(train_df.shape[0]):
# for i in range(10):
	# tweet_text = train_df['text'][i]
	# print(train_df['id'][i])
	# tweet_embedding = model.encode(tweet_text)
	train_dict[str(train_df['id'][i])] = (embeddings_list[i], train_df['target'][i])
	# train_dict[embeddings_list[i]] = train_df['target'][i]






with open('train_dict.p', 'wb') as file:
    pickle.dump(train_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
# with open('train_dict.json', 'w') as file:
#     json.dump(train_dict, file)
# TODO save as json dict
# dict = {embeddings:label}



# timeline = pickle.load(open( "timeline.p", "rb" ) )



# update timeline (this should remain hardcoded for every single feed because they may vary in their stucture)

# new york times
# newyorktimes_titles = [get_headlines(newyorktimes_feed)[i] for i in range(len(newyorktimes_feed))]
# newyorktimes_summaries = [clear_summary(get_summaries(newyorktimes_feed)[i]) for i in range(len(newyorktimes_feed))]
# newyorktimes_title_embeddings = model.encode(newyorktimes_titles, bsize=64, tokenize=False, verbose=True)  
# newyorktimes_summary_embeddings = model.encode(newyorktimes_summaries, bsize=64, tokenize=False, verbose=True)  

# newyorktimes_entries = [
# 	FeedEntry(
# 		now.date(),
# 		newyorktimes_titles[i],
# 		newyorktimes_summaries[i],
# 		newyorktimes_title_embeddings[i],
# 		newyorktimes_summary_embeddings[i]) 
# 	for i in range(len(newyorktimes_feed))]

# if (str(now.date()) not in timeline['newyorktimes'].keys()):
# 	timeline['newyorktimes'][str(now.date())] = []
# for entry in newyorktimes_entries:
# 	if (entry.title not in [existing_entry.title for existing_entry in timeline['newyorktimes'][str(now.date())]]):
# 		timeline['newyorktimes'][str(now.date())].append(entry)
# 		print("entry appended")
	
# 	else:
# 		print("skipping...already exists")


# # write updates to timeline file
# pickle.dump(timeline, open( "timeline.p", "wb" ))
























# ##############test embeddings
# import nltk
# # nltk.download('punkt')







# sentences = []
# # example on how to create entry objects for every current feed entry
# guardian_entries = [FeedEntry(now.date(),
# 		get_headlines(guardian_feed)[i],
# 		get_summaries(guardian_feed)[i],
# 		get_tags(guardian_feed)[i]) for i in range(len(guardian_feed))]


# aljazeera_entries = [FeedEntry(now.date(),
# 		get_headlines(aljazeera_feed)[i],
# 		get_summaries(aljazeera_feed)[i],
# 		get_tags(aljazeera_feed)[i]) for i in range(len(aljazeera_feed))]


# nytimes_entries = [FeedEntry(now.date(),
# 		get_headlines(nytimes_feed)[i],
# 		get_summaries(nytimes_feed)[i],
# 		get_tags(nytimes_feed)[i]) for i in range(len(nytimes_feed))]
# # print(nytimes_entries)


# aljazeera_titles = [entry.get_summary for entry in aljazeera_entries]
# nytimes_titles = [entry.get_summary for entry in nytimes_entries]
# # print(aljazeera_titles)
# # print(nytimes_titles)


# # aljazeera_sample_embedding = model.encode(aljazeera_titles[np.random.randint(0,len(aljazeera_titles)-1)])[0]
# # aljazeera_sample_embedding = model.encode(aljazeera_titles[0])

# # print(aljazeera_sample_embedding[0])

# aljazeera_embeddings = model.encode(aljazeera_titles, bsize=64, tokenize=False, verbose=True) 
# nytimes_embeddings = model.encode(nytimes_titles, bsize=64, tokenize=False, verbose=True)


# r = np.random.randint(0, len(aljazeera_embeddings)-1)
# aljazeera_sample_embedding = aljazeera_embeddings[r]

# # print(len(aljazeera_embeddings))
# # print(len(nytimes_embeddings))
# min_dist = -1
# # closest_index = 0
# # aljazeera_sample_embedding_normalized = math_utils.normalize(aljazeera_sample_embedding)
# # for i,embedding in enumerate(nytimes_embeddings):

# # 	current_dist = math_utils.cosine_distance(math_utils.normalize(embedding), aljazeera_sample_embedding_normalized)
# # 	# current_dist = math_utils.euclidean_distance(aljazeera_sample_embedding, embedding)
# # 	print(current_dist)

# # 	print(nytimes_titles[i])
# # 	print(aljazeera_titles[r])
# # 	print("\n")
# # 	if current_dist >= min_dist:
# # 		min_dist = current_dist
# # 		closest_index = i


# # 	# print(current_dist)
# # print("closest:\n")
# # print(min_dist)
# # print(nytimes_titles[closest_index])
# # print("\n")
# # print(aljazeera_titles[r])
# # # print("\n")

# # # _, _ = model.visualize(nytimes_titles[closest_index])
# # # _, _ = model.visualize(aljazeera_titles[r])




# timeline = pickle.load(open( "timeline.p", "rb" ) )

# for entry in nytimes_entries:
# 	if entry.title not in [existing_entry.title for existing_entry in timeline['newyorktimes'][str(now.date())]]:
# 		timeline['newyorktimes'][str(now.date())].append(entry)
# 		# print(entry.title)
# 		print(timeline['newyorktimes'][str(now.date())])
# 		print("entry appended")
# 	else:
# 		print("skipping...already exists")

# for entry in aljazeera_entries:
# 	if entry.title not in [existing_entry.title for existing_entry in timeline['aljazeera'][str(now.date())]]:
# 		timeline['aljazeera'][str(now.date())].append(entry)
# 		print("entry appended")
# 	else:
# 		print("skipping...already exists")

# # for entry in bbc_entries:
# # 	if entry.title not in [existing_entry.title for existing_entry in timeline['bbc'][str(now.date())]]:
# 		# timeline['bbc'][str(now.date())].append(entry)
# 	# 	print("entry appended")
# # else:
# # 	print("skipping...already exists")

# for entry in guardian_entries:
# 	if entry.title not in [existing_entry.title for existing_entry in timeline['theguardian'][str(now.date())]]:
# 		timeline['theguardian'][str(now.date())].append(entry)
# 		print("entry appended")
# 	else:
# 		print("skipping...already exists")

# # # for entry in reuters_entries:
# # # 	if entry.title not in [existing_entry.title for existing_entry in timeline['reuters'][str(now.date())]]:
# # 		# timeline['reuters'][str(now.date())].append(entry)
# # 	# 	print("entry appended")
# # else:
# # 	print("skipping...already exists")


# pickle.dump(timeline, open( "timeline.p", "wb" ))


# print(timeline)


 	

