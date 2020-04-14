import logging
import numpy as np


def lstm_calculate_similarity(model, item_id, item_list):
    seq_list = [[i] for i in item_id]
    #masked = np.full((1, 109, 1), fill_value=-9.0)
    #masked = np.full((1, 109, 1522), fill_value=-9.0)
    masked = np.full((1, 109, 256), fill_value=-9.0)

    seq_list = np.array([seq_list])

    for s, x in enumerate(seq_list):
        seq_len = x.shape[0]
        #masked[s, -seq_len:, :] = x
        masked[s, -seq_len:, :] = x[:seq_len, 0, :]

    try:
        similar_items = model.predict(np.array(masked))
    except:
        logging.error('LSTM Algorithm could not find the given item in the vocabulary')
        return [-1]

    top_10 = (-similar_items).argsort()[0][:10]

    recommended_items_set = []
    for rec_item in top_10:
        if rec_item in item_list:
            # encode the prediction
            encoded_item = item_list.get(rec_item)
            recommended_items_set.append(encoded_item)

    return recommended_items_set[:10]