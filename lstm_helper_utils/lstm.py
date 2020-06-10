import datetime
import logging

import numpy as np
import pandas as pd
import pickle
import joblib
import math
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.utils import shuffle
from kerastuner import RandomSearch
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('lstm.py')
np.random.seed(42)

class BaseLSTM(object):
    def __init__(self, model_description, path_params, training_params, hyper_params):

        # LSTM Mask Layer Value
        self.mask_value: float = -9.0

        # Path Objects
        self.path_item_dictionary: str = path_params['path_item_dictionary']
        self.path_user_sequences_dataframe: str = path_params['path_user_sequences_dataframe']
        self.path_dataframe_to_train: str = path_params['path_dataframe_to_train']
        self.path_evaluation_dataframe: str = path_params['path_dataframe_evaluation']
        self.path_lstm_model_feature_list: str = path_params['path_lstm_model_feature_list']
        self.path_lstm_model_targets_list: str = path_params['path_lstm_model_targets_list']
        self.path_data_information: str = path_params['path_data_information']
        self.path_best_lstm_model: str = path_params['path_best_lstm_model']
        self.lstm_model_save_dir: str = path_params['model_directory']
        self.lstm_tuner_save_dir: str = path_params['lstm_tuner_save_dir']

        # Data Objects
        self.item_dictionary = None
        self.user_sequences_df = None

        # Numpy Arrays to be used for pre-process and/or training the LSTM Model
        self.user_sequence_list = None
        self.all_features = None
        self.all_targets = None
        self.evaluation_data = None
        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None

        # LSTM Model
        self.lstm_model = None
        self.model_history = None
        self.lstm_model_description: str = model_description

        # LSTM Training
        self.objective_function: str = hyper_params['objective_function']
        self.max_trials: int = training_params['max_trials']
        self.execution_per_trial: int = training_params['execution_per_trial']
        self.epochs: int = training_params['epochs']
        self.batch_size: int = training_params['batch_size']
        self.k_split: int = training_params['k_split']
        self.random_state: int = training_params['random_state']
        self.split_names = ['train_features_fold_', 'test_features_fold_', 'train_targets_fold_', 'test_targets_fold_']
        self.best_possible_models = []

        self.hyper_params =hyper_params

        # LSTM Training Size
        self.train_size: int = 0
        self.test_fraction = training_params['test_fraction']
        self.test_size: int = 0
        self.evaluation_fraction = training_params['evaluation_fraction']
        self.evaluation_size: int = 0

        # Data Related
        self.max_sequence_size: int = 20
        self.number_of_distinct_items: int = 0

    def log_event(self, message):
        current_time = datetime.datetime.now()
        logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))

    def shuffle_data(self):
        self.user_sequences_df = shuffle(self.user_sequences_df, random_state=self.random_state)
        self.user_sequences_df.reset_index(inplace=True, drop=True)
        self.log_event('|----> Data is shuffled..')

    def load_data(self):
        self.log_event('|----> Starting to load data..')

        # Load Mapped Item Dictionary
        items_file = open(self.path_item_dictionary, 'rb')
        self.item_dictionary = pickle.load(items_file)
        self.number_of_distinct_items = len(self.item_dictionary)

        # Load User Item Sequence with the catalog items
        self.user_sequences_df = pd.read_pickle(self.path_user_sequences_dataframe)
        self.user_sequences_df = self.user_sequences_df[:1000]

        # Shuffle the data, and split the evaluation data from the training data
        self.shuffle_data()

        self.evaluation_size = int(len(self.user_sequences_df) * self.evaluation_fraction)

        self.evaluation_data = self.user_sequences_df.loc[np.random.choice(self.user_sequences_df.index, size=self.evaluation_size, replace=False)]
        self.user_sequences_df.drop(list(self.evaluation_data.index.values), inplace=True)

        self.evaluation_data.reset_index(inplace=True, drop=True)
        self.user_sequences_df.reset_index(inplace=True, drop=True)

        # Convert necessary columns to list
        self.user_sequence_list = self.user_sequences_df['item_sequence'].values

        self.save_data_to_disk(data_to_save=self.evaluation_data, data_name=self.path_evaluation_dataframe)
        del self.evaluation_data

        self.save_data_to_disk(data_to_save=self.user_sequences_df, data_name=self.path_dataframe_to_train)

        self.log_event('|--------+ Memory Management: user_sequence_df has been removed from memory..')

        self.log_event('|----> Necessary data has been loaded..')

    def save_data_to_disk(self, data_to_save, data_name):
        if not os.path.exists(data_name):
            joblib.dump(value=data_to_save, filename=data_name)
            self.log_event('|--------> Saving Data: Shuffled data has been saved into project folder..')
        else:
            self.log_event('|--------> Saving Data: {} already exist in the project folder..'.format(data_name))

    def retrieve_base_info_dictionary(self):
        info_dict = {
            'number_of_distinct_items': self.number_of_distinct_items,
            'max_sequence_size': self.max_sequence_size,
            'item_dictionary': self.item_dictionary,
        }
        return info_dict

    def load_fold_k_data_and_fit(self, split_number):
        self.train_features = joblib.load(self.lstm_model_save_dir + 'split_' + str(split_number) + '/train_split_' + str(split_number) + '_all_features.pkl')
        self.train_targets = joblib.load(self.lstm_model_save_dir + 'split_' + str(split_number) + '/train_split_' + str(split_number) + '_all_targets.pkl')
        self.test_features = joblib.load(self.lstm_model_save_dir + 'split_' + str(split_number) + '/test_split_' + str(split_number) + '_all_features.pkl')
        self.test_targets = joblib.load(self.lstm_model_save_dir + 'split_' + str(split_number) + '/test_split_' + str(split_number) + '_all_targets.pkl')
        self.fit_model(split_number=split_number)

    def load_train_data(self):
        if not os.path.exists(self.path_data_information):
            self.user_sequences_df = joblib.load(self.path_dataframe_to_train)

            items_file = open(self.path_item_dictionary, 'rb')
            self.item_dictionary = pickle.load(items_file)
            self.number_of_distinct_items = len(self.item_dictionary)

            joblib.dump(value=self.retrieve_base_info_dictionary(), filename=self.path_data_information)
        else:
            self.user_sequences_df = joblib.load(self.path_dataframe_to_train)
            self.load_object_parameters()

    def check_file_exists(self, path):
        if os.path.exists(path):
            return True
        else:
            return False

    def load_object_parameters(self):
        info_dict = joblib.load(self.path_data_information)

        self.number_of_distinct_items = info_dict['number_of_distinct_items']
        self.item_dictionary = info_dict['item_dictionary']
        self.max_sequence_size = info_dict['max_sequence_size']

        return info_dict

    def apply_mask(self, all_features, number_of_features):
        self.log_event('|----> Masking the Feature List..')

        masked_features = np.full((len(all_features), self.max_sequence_size, number_of_features), fill_value=self.mask_value, dtype='float32')

        for s, x in enumerate(all_features):
            seq_len = x.shape[0]
            difference = seq_len - self.max_sequence_size

            if difference > 0:
                x = x[difference:]
                seq_len = seq_len - difference

            if masked_features.shape[2] == 1:
                masked_features[s, -seq_len:, :] = x[:seq_len].reshape(seq_len, 1)
            else:
                masked_features[s, -seq_len:, :] = x[:seq_len]

        self.all_features = masked_features
        del masked_features
        self.log_event('|----> Masking is completed..')

    def encode_targets(self):
        self.log_event('|----> One Hot Encoding the Targets..')

        count = 0
        targets = np.zeros(shape=(len(self.all_targets), self.number_of_distinct_items), dtype='int8')

        for target in self.all_targets:
            encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(target)]
            vector = np.zeros(shape=self.number_of_distinct_items, dtype='int8')
            vector[int(encoding)] = 1
            targets[count] = vector
            count += 1

        self.all_targets = targets
        del targets
        self.log_event('|----> One Hot Encoding has finished..')

    def create_features_and_targets(self):
        self.log_event('|----> Creating feature and label list for user sequences..')

        self.all_features = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype=object)
        self.all_targets = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype='int32')

        for index, sequence in enumerate(self.user_sequence_list):
            new_sequence = np.full(len(sequence) - 1, fill_value=self.mask_value, dtype='int32')
            for idx, item in enumerate(sequence[:-1]):
                new_sequence[idx] = item

            self.all_features[index] = new_sequence
            self.all_targets[index] = sequence[-1]

        del self.user_sequence_list
        self.log_event('|--------+ Memory Management: user_sequence_list has been removed from memory..')

        self.log_event('|----> Features and targets are ready..')

    def create_common_folds_to_use(self):
        folds = KFold(n_splits=self.k_split, random_state=self.random_state, shuffle=True)
        for split_count, (train_index, test_index) in enumerate(folds.split(self.user_sequences_df)):
            self.save_data_to_disk(data_to_save=self.user_sequences_df[self.user_sequences_df.index.isin(train_index)], data_name='D:/model_data/common_data/folds/train_split_' + str(split_count) + '.fold')
            self.save_data_to_disk(data_to_save=self.user_sequences_df[self.user_sequences_df.index.isin(test_index)], data_name='D:/model_data/common_data/folds/test_split_' + str(split_count) + '.fold')

    def create_folds(self):
        folds = KFold(n_splits=self.k_split, random_state=self.random_state, shuffle=True)
        for i, (train_index, val_index) in enumerate(folds.split(self.all_features, self.all_targets.argmax(1))):
            split_folder_path = f'E:/folds/' + self.lstm_model_description + '/split_' + str(i)
            if not os.path.exists(split_folder_path):
                os.makedirs(split_folder_path)

                self.train_features = self.all_features[train_index]
                self.save_data_to_disk(data_to_save=self.train_features, save_dir=f'E:/folds/' + self.lstm_model_description + '/split_' + str(i),
                                       model_name=self.lstm_model_description, data_name='train_features_fold_' + str(i) + '.pkl')
                self.train_features = None

                self.test_features = self.all_features[val_index]
                self.save_data_to_disk(data_to_save=self.test_features, save_dir=f'E:/folds/' + self.lstm_model_description + '/split_' + str(i),
                                       model_name=self.lstm_model_description, data_name='test_features_fold_' + str(i) + '.pkl')
                self.test_features = None

                self.train_targets = self.all_targets[train_index]
                self.save_data_to_disk(data_to_save=self.train_targets, save_dir=f'E:/folds/' + self.lstm_model_description + '/split_' + str(i),
                                       model_name=self.lstm_model_description, data_name='train_targets_fold_' + str(i) + '.pkl')
                self.train_targets = None

                self.test_targets = self.all_targets[val_index]
                self.save_data_to_disk(data_to_save=self.test_targets, save_dir=f'E:/folds/' + self.lstm_model_description + '/split_' + str(i),
                                       model_name=self.lstm_model_description, data_name='test_targets_fold_' + str(i) + '.pkl')
                self.train_targets = None

        del self.all_features, self.all_targets

    def fit_model(self, split_number):
        self.log_event('Training with HyperParameter Tuning is started..')

        tuner = RandomSearch(self.build_model,
                             objective=self.objective_function,
                             max_trials=self.max_trials,
                             executions_per_trial=self.execution_per_trial,
                             seed=self.random_state,
                             project_name='split_' + str(split_number),
                             directory=os.path.normpath(self.lstm_tuner_save_dir))

        tuner.search(self.train_features, self.train_targets,
                     epochs=self.epochs,
                     batch_size=self.batch_size,
                     verbose=2,
                     validation_data=(self.test_features, self.test_targets))

        model = tuner.get_best_models(num_models=1)[0]

        print(model.summary())
        keras.utils.plot_model(model,
                               to_file=self.lstm_model_save_dir + self.lstm_model_description + '.png',
                               show_shapes=True,
                               show_layer_names=True)

        # Fit the best model of the split with the data
        history = model.fit(x=self.train_features,
                            y=self.train_targets,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=2,
                            validation_data=(self.test_features, self.test_targets))

        # Save the model
        current_time = datetime.datetime.now()
        model.save(self.lstm_tuner_save_dir + '/split_' + str(split_number) + '/' + self.lstm_model_description + '_' + str(current_time.strftime('%Y-%m-%d_%H-%M-%S')) + '_' + '.h5')

        self.lstm_model = model
        self.model_history = history



        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(self.lstm_tuner_save_dir + '/split_' + str(split_number) + '/best_model_history.csv', index=False, header=True)
        self.best_possible_models.append(hist_df)
        self.print_summary(split_number)
        self.log_event('Training with HyperParameter Tuning is finished..')

    def fit(self, train_data, split_count):
        self.train_features = train_data[0]
        self.train_targets = train_data[1]
        self.test_features = train_data[2]
        self.test_targets = train_data[3]

        del train_data

        self.fit_model(split_count)

    def retrieve_best_model(self, best_possible_models):
        avg_val_accuracy_list = []
        for model in range(len(self.best_possible_models)):
            val_accuracy_list = self.best_possible_models[model]['val_accuracy'].values
            avg_accuracy = sum(val_accuracy_list) / len(val_accuracy_list)
            avg_val_accuracy_list.append(avg_accuracy)

            #avg_val_accuracy = sum(model.history.history['val_accuracy'])/len(model.history.history['val_accuracy'])
            #avg_val_accuracy_list.append(avg_val_accuracy)

        index = avg_val_accuracy_list.index(max(avg_val_accuracy_list))

        model_path = self.lstm_tuner_save_dir + '/split_' + str(index)
        best_model_path = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".h5")][0]

        best_model = load_model(best_model_path)

        # Save the best model to main folder
        best_model.save(self.lstm_model_save_dir + self.lstm_model_description +'_best_model.h5')

    def build_model(self, hp):
        self.log_event('    -> Creating a model.')
        model = Sequential()

        model.add(self.add_mask_layer())

        # Add One LSTM Layer with Batch Normalization
        model.add(LSTM(units=hp.Int('first_layer',
                                    min_value=self.hyper_params['lstm_units']['min'],
                                    max_value=self.hyper_params['lstm_units']['max'],
                                    step=self.hyper_params['lstm_units']['step']),
                       return_sequences=True,
                       dropout=self.hyper_params['lstm_layer_dropout'],
                       recurrent_dropout=0.1,
                       activation=self.hyper_params['lstm_layer_activation']))

        model.add(BatchNormalization())

        # Add Dropout
        model.add(Dropout(hp.Choice('dropout_one', values=self.hyper_params['dropout'])))

        # Add the second LSTM Layer with Batch Normalization
        model.add(LSTM(units=hp.Int('second_layer',
                                    min_value=self.hyper_params['lstm_units']['min'],
                                    max_value=self.hyper_params['lstm_units']['max'],
                                    step=self.hyper_params['lstm_units']['step']),
                       return_sequences=False,
                       dropout=self.hyper_params['lstm_layer_dropout'],
                       recurrent_dropout=0.1,
                       activation=self.hyper_params['lstm_layer_activation']))

        model.add(BatchNormalization())

        # Add Dropout
        model.add(Dropout(hp.Choice('dropout_one', values=self.hyper_params['dropout'])))

        # Add Output Layer
        model.add(Dense(self.number_of_distinct_items, activation=self.hyper_params['dense_activation']))

        # Compile the model
        opt = Adam(hp.Choice('learning_rate', values=self.hyper_params['learning_rate']))

        model.compile(loss=self.hyper_params['loss'], optimizer=opt, metrics=self.hyper_params['metric'])

        self.log_event('    -> Returning the model.')
        return model

    def add_mask_layer(self):
        return  Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, self.number_of_distinct_items))

    def encode_item(self, item_id, encoding_dict):
        vector = np.zeros(len(encoding_dict))
        encoding = list(encoding_dict.keys())[list(encoding_dict.values()).index(item_id)]
        vector[encoding] = 1
        return vector

    def scale_down_vector(self, vector_to_calc_l2):
        from numpy.linalg import norm

        l2_norm = norm(vector_to_calc_l2)
        return vector_to_calc_l2 / l2_norm

    def print_summary(self, i):
        current_time = datetime.datetime.now()
        print('Model is successfully trained.. ' + str(current_time))
        print('Train on {} samples and Test on {} samples'.format(len(self.train_features), len(self.test_features)))

        plt.plot(self.model_history.history['accuracy'])
        plt.plot(self.model_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.lstm_tuner_save_dir + '/split_' + str(i) + '/' + self.lstm_model_description + '_accuracy_' + str(current_time.strftime('%Y-%m-%d_%H-%M-%S')) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()

        # Plot training & validation loss values
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(self.lstm_tuner_save_dir + '/split_' + str(i) + '/' + self.lstm_model_description + '_loss_' + str(current_time.strftime('%Y-%m-%d_%H-%M-%S')) + '.png', bbox_inches='tight')
        plt.plot()
        plt.close()

    def load_best_tune_model(self, path_split):
        model_path = self.lstm_tuner_save_dir + '/' + path_split + '/'
        best_model_path = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".csv")][0]
        best_model = pd.read_csv(best_model_path)
        self.best_possible_models.append(best_model)
        self.log_event('Current folder has trained model in it. No need for traning.')

    def load_evaluation_data(self):
        items_file = open(self.path_item_dictionary, 'rb')
        self.item_dictionary = pickle.load(items_file)
        self.number_of_distinct_items = len(self.item_dictionary)

        evaluation_dataframe = joblib.load(self.path_evaluation_dataframe)

        evaluation_dataframe.drop(['user_log_list'], axis=1, inplace=True)
        return evaluation_dataframe

    def evaluation_creator(self, model, model_name, sequence_to_evaluate, evaluate_all_items_in_the_sequence):
        from evaluation.SequenceEvaluation import Evaluator

        sequence_count = 0
        ndcg_count = 0
        for user_id, session_id, user_sequence, decoded_individual_sequence in sequence_to_evaluate:
            evaluator = Evaluator(user_id=user_id,
                                  model_name=model_name,
                                  predictor=model,
                                  item_dictionary=self.item_dictionary,
                                  session_id=session_id,
                                  encoded_sequence_to_evaluate=user_sequence,
                                  decoded_sequence=decoded_individual_sequence,
                                  evaluate_all_sequence=evaluate_all_items_in_the_sequence,
                                  top_k=10,
                                  increment_by=1)

            prm, ndcg = evaluator.sequential_evaluation()

            if ndcg == -1:
                print('trivial prediction..')
            if ndcg != 0 and ndcg != 10:
                ndcg_count += 1

            prm = np.round(prm, decimals=3)
            ndcg = round(ndcg, 3)

            individual_session_average_precision = prm[0]
            individual_session_average_recall = prm[1]
            individual_session_average_mrr = prm[2]
            individual_session_average_ndcg = ndcg

            logger.info(f'User {user_id} has been evaluated.. >> '
                        f'{individual_session_average_precision} >> '
                        f'{individual_session_average_recall} >> '
                        f'{individual_session_average_mrr} >> '
                        f'>> {individual_session_average_ndcg}')

            # Insert to DB
            #insert_evaluation(user_id, individual_session_average_precision, individual_session_average_recall, individual_session_average_mrr, individual_session_average_ndcg,
            #                  'version_five_16_epoch_predict_all')
            sequence_count += 1

        print("Total evaluated user sequences: {} and  total nDCG count above zero: {}".format(sequence_count, ndcg_count))


