import numpy as np
import logging
from tqdm import tqdm
from math import log
from evaluation_utils.Metrics import recall, precision, mrr
from database_utils.DatabaseHelper import insert_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Evaluation.py')


class Evaluator(object):
    def __init__(self, user_id, model_name, predictor, item_dictionary, session_id, encoded_sequence_to_evaluate, decoded_sequence, evaluate_all_sequence, top_k, increment_by):
        self.user_id = user_id
        self.model_name = model_name
        self.predictor = predictor
        self.item_dictionary = item_dictionary
        self.session_id = session_id
        self.encoded_sequence_to_evaluate = encoded_sequence_to_evaluate
        self.decoded_sequence = decoded_sequence
        self.evaluate_all_sequence = evaluate_all_sequence
        self.top_k = top_k
        self.increment_by = increment_by
        self.ndcg = 0.
        self.real = 0.
        self.real_index = 0
        self.metrics = {'precision': precision, 'recall': recall, 'mrr': mrr}
        self.prm = np.zeros(len(self.metrics.values()))
        self.recommendation = None

    def sequential_evaluation(self):
        if self.evaluate_all_sequence:
            progress_bar_length = len(self.decoded_sequence) - 1
        else:
            progress_bar_length = 1

        with tqdm(total=progress_bar_length) as progress_bar:

            if self.evaluate_all_sequence:
                stack = np.full((1, 20, len(self.item_dictionary)), fill_value=-9.0, dtype='float32')
                catalog_counter = 0
                slicer = 0
                for index in range(len(self.encoded_sequence_to_evaluate) - 1):
                    is_trivial_prediction = False
                    self.prm = np.zeros(len(self.metrics.values()))
                    self.ndcg = 0

                    temp_seq = self.encoded_sequence_to_evaluate[:slicer + 1]
                    if self.encoded_sequence_to_evaluate[slicer].max() == 1.0:

                        if slicer + 1 < len(self.encoded_sequence_to_evaluate):
                            while self.encoded_sequence_to_evaluate[slicer + 1].max() != 1.0:
                                slicer += 1
                                self.increment_by += 1
                                catalog_counter += 1
                                temp_seq = self.encoded_sequence_to_evaluate[:slicer + 1]
                                progress_bar.update(1)

                        if slicer >= 19:
                            temp_seq = temp_seq[-20:]

                        seq_len = temp_seq.shape[0]
                        stack[0, -seq_len:, :] = temp_seq[:]

                        self.evaluate_sequence(stack, self.decoded_sequence)

                        if self.ndcg == -1:
                            is_trivial_prediction = True

                        insert_evaluation(user_id=self.user_id,
                                          session_id=self.session_id,
                                          precision=self.prm[0],
                                          recall=self.prm[1],
                                          mrr=self.prm[2],
                                          ndcg=self.ndcg,
                                          predictor_name=self.model_name,
                                          trivial_prediction=is_trivial_prediction,
                                          catalog_count=catalog_counter,
                                          ground_truth=self.real,
                                          sequence=' '.join(map(str, self.decoded_sequence)),
                                          input_sequence=' '.join(map(str, self.decoded_sequence[:seq_len])),
                                          predictions=' '.join(map(str, self.recommendation)))

                        if is_trivial_prediction:
                            return self.prm, self.ndcg

                        self.increment_by += 1
                        progress_bar.update(1)
                    else:
                        catalog_counter += 1
                        self.increment_by += 1
                        progress_bar.update(1)

                    slicer += 1

                    if self.increment_by == len(self.decoded_sequence):
                        break

                if (len(self.decoded_sequence)) - 1 - catalog_counter == 0:
                    return self.prm, self.ndcg
                # Return the sequence results: Precision, Recall, Mrr, nDCG
                self.prm = self.prm / (len(self.decoded_sequence) - 1 - catalog_counter)
                self.ndcg = self.ndcg / (len(self.decoded_sequence) - 1 - catalog_counter)

            else:
                stack = np.full((1, 20, len(self.item_dictionary)), fill_value=-9.0, dtype='float32')
                seq_len = self.encoded_sequence_to_evaluate[:-1].shape[0]

                if seq_len >= 20:
                    stack[0, -20:, :] = self.encoded_sequence_to_evaluate[-20:]
                else:
                    stack[0, -seq_len:, :] = self.encoded_sequence_to_evaluate[:-1]

                self.evaluate_sequence(stack, self.decoded_sequence)

                # Calculate Precision, Recall and MMR
                for i, metric_function in enumerate(self.metrics.values()):
                    self.prm[i] = metric_function([self.real], self.recommendation)

                # Calculate nDCG
                if 0.0 not in self.prm:
                    # Then calculate nDCG
                    self.calculate_ndcg()

                progress_bar.update(1)

        return self.prm, self.ndcg

    def evaluate_sequence(self, sequence, gt):
        user_profile = sequence
        ground_truth = gt[self.increment_by:]

        self.recommendation = self.lstm_calculate_similarity(user_sequence=user_profile)[:self.top_k]

        if self.evaluate_all_sequence:
            self.real = ground_truth[0]
            self.real_index = gt.index(self.real)

            # Calculate Precision, Recall and MMR
            for i, metric_function in enumerate(self.metrics.values()):
                self.prm[i] += metric_function([self.real], self.recommendation)
            # Calculate nDCG
            if 0.0 not in self.prm:
                # Then calculate nDCG
                self.calculate_ndcg()

        else:
            self.real = ground_truth[-1]

    def lstm_calculate_similarity(self, user_sequence):
        try:
            similar_items = self.predictor.predict(user_sequence)
        except Exception as err:
            logging.error('LSTM Algorithm could not find the given item in the vocabulary', '\n', err)
            return [-1]

        top_k = (-similar_items).argsort()[0][:self.top_k]

        recommended_items_set = []
        for rec_item in top_k:
            if rec_item in self.item_dictionary:
                encoded_item = self.item_dictionary.get(rec_item)
                recommended_items_set.append(encoded_item)

        return recommended_items_set[:self.top_k]

    def calculate_ndcg(self):
        one_zero_list = []
        for pred in self.recommendation:
            if pred == self.real:
                one_zero_list.append(1)
            else:
                one_zero_list.append(0)

        pos = 1
        if 1 in one_zero_list:
            if self.real in self.decoded_sequence[self.real_index + 1:]:
                self.ndcg = -1
                return self.ndcg

            dcg_score = 0
            for item in one_zero_list:
                if item != 0:
                    dcg_score += 1 / log(pos+1, 2)
                pos += 1

            self.ndcg += dcg_score

            # TODO: Mention this!
            if len(set(self.decoded_sequence)) == 1:
                self.ndcg = 10
        else:
            return 0.