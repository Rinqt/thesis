import logging
import pickle
import pandas as pd
import numpy as np

from datetime import datetime

from gensim.models import Doc2Vec

from helper_classes import PreProcessData as preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Doc2Vec.py')


def train_doc2vec(train_data):

    travel_package = train_data()
    columns = ['id', 'nazev', 'popisek', 'do']
    travel_package_df = pd.DataFrame(data=travel_package, columns=columns)
    travel_package_df = travel_package_df.sort_values('do').drop_duplicates('id', keep='last').sort_index()
    travel_package_df = travel_package_df.drop(columns=['do'])

    # Combine every column into one, separating by comma
    result_df = travel_package_df.apply(lambda x: ','.join(x.astype(str)), axis=1)

    # Pre-Process the text
    result_df = preprocess.clean_text(result_df)
    train_corpus = list(preprocess.read_corpus(result_df))

    now = datetime.now()
    print('Training is started...', now.strftime("%H:%M:%S"))

    model = Doc2Vec(vector_size=128, min_count=0, epochs=64, workers=16, window=1)
    logging.warning('Doc2Vec model is completed!')

    model.build_vocab(train_corpus)
    logging.warning('Vocabulary build is completed!')

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    logging.warning('Doc2Vec model training is completed!')

    now = datetime.now()
    print('Training is over...', now.strftime("%H:%M:%S"))

    model_name = 'doc2vec_{date:%Y-%m-%d_%H:%M:%S}_.model'.format(date=datetime.now())
    file_handler = open("doc2vec.model", "wb")
    pickle.dump(model, file_handler)
    file_handler.close()

    file = open('doc2vec.model', 'rb')
    model = pickle.load(file)
    pickle_dict = {}
    for index, sentence in enumerate(train_corpus):
        doc_tag = int(sentence.tags[0])
        pickle_dict[doc_tag] = {}

        doc_vectors = model.docvecs[index]
        doc_words = sentence.words[:10]

        pickle_dict[doc_tag][doc_tag] = [doc_vectors, doc_words]

    vocabulary_name = 'items_pickle_{date:%Y-%m-%d_%H:%M:%S}_.items'.format(date=datetime.now())
    file_handler = open("items_pickle.items", "wb")
    pickle.dump(pickle_dict, file_handler)
    file_handler.close()


def doc2vec_calculate_similarity(model, item_id, item_list):
    try:
        selected_item_vector = item_list.get(int(item_id))
        selected_item_vector = selected_item_vector[int(item_id)][0]
        selected_item_vector = np.asarray(selected_item_vector)
        similar_items = model.docvecs.most_similar([selected_item_vector], topn=31)
    except:
        logging.error('Doc2Vec Algorithm could not find the given item in the vocabulary')
        return [-1]


    recommended_items = []
    for sim_item in similar_items[1:]:
        item_key = int(sim_item[0])
        if item_key in item_list.keys():
            recommended_items.append(int(item_key))

    return recommended_items[:10]
