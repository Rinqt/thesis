import logging
import pickle

from gensim.models import Word2Vec
from datetime import datetime


def train_item2vec(train_data):
    now = datetime.now()
    print('Training is started...', now.strftime("%H:%M:%S"))

    model = Word2Vec(train_data, min_count=0, size=128, workers=16, window=1, sg=0, iter=3072)

    now = datetime.now()
    print('Training is over...', now.strftime("%H:%M:%S"))

    model_name = 'item2vec_{date:%Y-%m-%d_%H:%M:%S}_.model'.format(date=datetime.now())
    file_handler = open("item2vec.model", "wb")
    pickle.dump(model, file_handler)
    file_handler.close()


def item2vec_calculate_similarity(model, item_id, item_list):
    """
    :summary:  Creates a recommendation list that contains ids of the item that is similar to given id.
    :workflow:
                -> Load the item2vec model from database.
                -> Use model to find similar items to given item id.
                -> Create a recommendation list and start appending predicted item ids (Note that some item ids
                   might not be in the database)
                -> Return the list.

    :return: recommended_items_set: List of item ids to use for recommendation
    """
    try:
        similar_items = model.most_similar(str(item_id), topn=20)
    except:
        logging.error('Item2Vec Algorithm could not find the given item in the vocabulary')
        return [-1]

    recommended_items_set = []
    for rec_item in similar_items:
        if int(rec_item[0]) in item_list:
            recommended_items_set.append(rec_item[0])

    return recommended_items_set[:10]
