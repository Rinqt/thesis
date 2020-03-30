import logging
import numpy as np

from tqdm import tqdm
from helper_classes.Metrics import recall, precision, mrr

from recommenders.Item2Vec import item2vec_calculate_similarity
from recommenders.Doc2Vec import doc2vec_calculate_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Evaluation.py')


def get_recommendation_list(temp):
    return list(map(lambda x: x, temp))


class Evaluator(object):

    def __init__(self, recommender_model, recommender_name, metrics, items, test_sequence, given_k, look_ahead, top_n, step):
        self.recommender_model = recommender_model
        self.recommender_name = recommender_name
        self.metrics = metrics
        self.items = items
        self.test_sequences = test_sequence
        self.given_k = given_k
        self.look_ahead = look_ahead
        self.top_n = top_n
        self.step = step

    def sequential_evaluation(self):
        if self.given_k == 0:
            raise ValueError('given_k must be != 0')

        metrics = np.zeros(len(self.metrics.values()))

        with tqdm(total=len(self.test_sequences)) as pbar:

            for i, test_sequence in enumerate(self.test_sequences):
                metrics += self.sequence_sequential_evaluation(test_sequence)
                pbar.update(1)

        return metrics / len(self.test_sequences)

    def sequence_sequential_evaluation(self, test_sequence):
        if self.given_k < 0:
            self.given_k = len(test_sequence) + self.given_k

        eval_res = 0.0
        eval_cnt = 0

        for gk in range(self.given_k, len(test_sequence), self.step):
            eval_res += self.evaluate_sequence(test_sequence)
            eval_cnt += 1

        return eval_res / eval_cnt

    def evaluate_sequence(self, test_sequence):
        recommendation_list = []

        # safety checks
        if self.given_k < 0:
            self.given_k = len(test_sequence) + self.given_k

        user_profile = test_sequence[:self.given_k]
        ground_truth = test_sequence[self.given_k:]

        # restrict ground truth to look_ahead
        ground_truth = ground_truth[:self.look_ahead] if self.look_ahead != 'all' else ground_truth
        #ground_truth = list(map(lambda x: [x], ground_truth))  # list of list format

        if not ground_truth:
            # if ground_truth is missing, evaluation functions are 0
            return np.zeros(len(self.metrics.values()))

        if self.recommender_name == 'i2v':
            item_id = user_profile[-1]
            recommendations = item2vec_calculate_similarity(model=self.recommender_model,
                                                            item_id=int(item_id),
                                                            item_list=self.items)[:self.top_n]
            if recommendations == [-1]:
                # no recommendation found
                return np.zeros(len(self.metrics.values()))
            else:
                recommendation_list = get_recommendation_list(recommendations)

        elif self.recommender_name == 'd2v':
            item_id = user_profile[-1]
            recommendations = doc2vec_calculate_similarity(model=self.recommender_model,
                                                           item_id=int(item_id),
                                                           item_list=self.items)[:self.top_n]
            recommendation_list = get_recommendation_list(recommendations)

        tmp_results = []

        for f in self.metrics.values():
            tmp_results.append(f(ground_truth, recommendation_list))
        return np.array(tmp_results)


def evaluate_models(model_names):
    from helper_classes.DatabaseHelper import fetch_all_items, insert_evaluation
    from helper_classes import PreProcessData as preprocess
    import pickle

    metrics = {'precision': precision, 'recall': recall, 'mrr': mrr}
    sequence_test = preprocess.concat_user_sequences(sequence_low_limit=2)
    train_data = sequence_test.values.tolist()

    for model_name in model_names:
        if model_name == 'i2v':
            model_path = f'D:/Thesis/recsys_thesis/models/item2vec.model'
            items = fetch_all_items()

        else:
            model_path = f'D:/Thesis/recsys_thesis/models/doc2vec.model'
            items_path = f'D:/Thesis/recsys_thesis/models/items_pickle.items'
            items_file = open(items_path, 'rb')
            items = pickle.load(items_file)

        file = open(model_path, 'rb')
        model = pickle.load(file)
        given_k = 1
        look_ahead = 'all'
        top_n = 10
        step = 1

        for user_id, user_sequence in train_data:
            item2vec_evaluator = Evaluator(recommender_model=model,
                                           recommender_name=model_name,
                                           metrics=metrics,
                                           items=items,
                                           test_sequence=user_sequence,
                                           given_k=given_k,
                                           look_ahead=look_ahead,
                                           top_n=top_n,
                                           step=step)

            evaluation_result = item2vec_evaluator.sequential_evaluation()


            metric_results = list(zip(metrics.keys(), evaluation_result))

            user_average_precision = round(metric_results[0][1], 2)
            user_average_recall = round(metric_results[1][1], 2)
            user_average_mrr = round(metric_results[2][1], 2)

            logger.info(f'User {user_id} has been evaluated.. >> {user_average_precision} >> {user_average_recall} >> {user_average_mrr}')
            insert_evaluation(user_id, user_average_precision, user_average_recall, user_average_mrr, model_name)


#model_names = ['i2v', 'd2v']
#evaluate_models(model_names)
