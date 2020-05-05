"""
    LSTM Version One:
        Features: User items kept as integer. Features only consist of sequence of item that user visitid per session.
        Labels: One Hot Encoding
"""

import datetime
import logging
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from kerastuner import RandomSearch

from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('version_one.py')

def log_event(message):
    current_time = datetime.datetime.now()
    logger.info(message + ' >> ' + str(current_time.strftime('%d-%m-%Y %H-%M-%S')))


class LstmV1(object):
    def __init__(self, sequence_limit, test_size):
        self.sequence_limit = sequence_limit
        self.test_size = test_size

        self.mask_value = -9.0
        self.mapped_items_dict = None
        self.distinct_item_count = None
        self.max_item_in_sequence = None
        self.sequences_df = None
        self.user_sequence_list = None
        self.feature_list = None
        self.target_list = None
        self.train_features = None
        self.train_targets = None
        self.test_features = None
        self.test_targets = None
        self.loaded_model = None
        self.model_history = None
        self.lstm_model = None

    def create_sequences(self):
        log_event('Starting to load data..')

        # Load Mapped Item Dictionary
        items_path = f'D:/Thesis/recsys_thesis/item_dictionary.pickle'
        items_file = open(items_path, 'rb')
        self.mapped_items_dict = pickle.load(items_file)
        self.distinct_item_count = len(self.mapped_items_dict)

        # Load User Item Sequence with the catalog items
        self.sequences_df = pd.read_pickle("D:/Thesis/recsys_thesis/user_item_sequence_with_catalog_items.pickle")
        self.user_sequence_list = self.sequences_df['itemSequence'].values

        self.max_item_in_sequence = 20

        log_event('Necessary data has been loaded..')

    def create_train_data(self):
        """
        Summary:
            Method divides the data set into to pieces, namely Features and Targets.
                Features = Every single item inside the user session except the last item.
                Target   = Last item of the user session

                Suppose user has following session = [1, 2, 3, 4, 5, 6, 7, 8]
                After create_train_data method finishes, above session will be divided as:
                    Features = [1, 2, 3, 4, 5, 6, 7]
                    Target   = [8]
        
        Workflow:
            1. Iterate through the entire dataset.
            2. For each user sequence, add all items except the last one into a feature list
            3. Add the last item in the sequence into a target list

        Returns:
            self.feature_list = All user sequences exlucding the last item 
            self.target_list  = Last item of all user sequences

        """

        log_event('   -> Creating feature and label list for user sequences..')

        feature_list = []
        target_list = []

        for sequence in self.user_sequence_list:
            t = []
            for item in sequence[:-1]:
                t.append(item)

            feature_list.append(np.array(t))
            target_list.append(sequence[-1])


        self.feature_list = np.array(feature_list)
        self.target_list = target_list

        log_event('   -> Features and labels are ready..')

    def encode_targets(self):
        """
        Summary:
            Method is responsible of encoding the target list.

            Example:
                Target item = 3
                One-Hot-Encoded representation of the item 3:                
                    item_1, item_2, item_3, item_4, item_5
                    [  0,      0,      1,      0,      0 ]

        Workflow:
            1. Iterate through target list.
            2. Find the mapped number of the item by using 'mapped_items_dict' dictionary.
            3. Create a vector which has a length of distinct item count.
            4. Change the item location of the vector as 1.

        Returns:
            One-Hot-Encoded target_list
        """
        log_event('   -> One Hot Encoding the targets..')

        count = 0
        for target in self.target_list:
            encoding = list(self.mapped_items_dict.keys())[list(self.mapped_items_dict.values()).index(target)]
            vector = [0 for _ in range(self.distinct_item_count)]
            vector[int(encoding)] = 1
            self.target_list[count] = vector
            count += 1

        log_event('   -> One Hot Encoding has finished..')

    def prepare_data(self):
        log_event('Start to prepare data for training..')

        # Step 1: Create Feature and Label list
        self.create_train_data()

        # Step 2: Shuffle the entire dataset
        self.feature_list, self.target_list = shuffle(self.feature_list, self.target_list, random_state=42)

        # Step 3: Apply Mask
        self.feature_list = self.apply_mask()

        # Step 4: One-Hot-Encode the targets
        self.encode_targets()

        # Step 5: Create Train, Test Split
        test_size = int(len(self.feature_list) * self.test_size)
        train_size = len(self.feature_list) - test_size

        self.train_features = self.feature_list[:train_size]
        self.test_features = self.feature_list[train_size:]

        self.train_targets = self.target_list[:train_size]
        self.test_targets = self.target_list[train_size:]

        log_event('Data is ready to be trained..')

    def fit_model(self):
        log_event('Training with HyperParameter Tuning is started..')
        current_time = datetime.datetime.now()
        start = str(current_time.strftime('%d-%m-%Y_%H-%M-%S'))

        DIRECTORY = 'D:\\Thesis\\recsys_thesis\\lstm_versions\\hyper_parameter_tuning\\'

        tuner = RandomSearch(self.build_model,
                             objective='val_accuracy',
                             max_trials=3,
                             executions_per_trial=5,
                             seed=42,
                             project_name='model_1',
                             directory=DIRECTORY)

        tuner.search(x=self.train_features,
                     y=np.array(self.train_targets),
                     epochs=64,
                     batch_size=128,
                     verbose=2,
                     validation_data=(self.test_features, np.array(self.test_targets)))

        with open(f'tuner_model_one_' + str(current_time.strftime('%d-%m-%Y_%H-%M-%S')) + '.pickle', 'wb') as file:
            pickle.dump(tuner, file)

        current_time = datetime.datetime.now()
        print('Tuner Best Hyper Parameters: \n', tuner.get_best_hyperparameters()[0].values)
        print('Start: ', start, ' end: ' + str(current_time.strftime('%d-%m-%Y_%H-%M-%S')))

        log_event('Training with HyperParameter Tuning is finished..')

    def build_model(self, hp):
        log_event('    -> Creating a model.')

        model = Sequential()

        model.add(Masking(mask_value=self.mask_value, input_shape=(self.max_item_in_sequence, 1)))

        # Add One LSTM Layer with Batch Normalization
        model.add(LSTM(units=hp.Int('first_layer', min_value=8, max_value=64, step=8), return_sequences=True, dropout=0.1, recurrent_dropout=0.1, activation='tanh'))
        model.add(BatchNormalization())

        # Add the second LSTM Layer with Batch Normalization
        model.add(LSTM(units=hp.Int('second_layer', min_value=8, max_value=64, step=8), return_sequences=False, dropout=0.1, recurrent_dropout=0.1, activation='tanh'))
        model.add(BatchNormalization())

        # Add Dropout
        model.add(Dropout(hp.Choice('dropout', values=[0.3, 0.4, 0.5])))

        # Add Output Layer
        model.add(Dense(self.distinct_item_count, activation='softmax'))

        # Compile the model
        opt = Adam(hp.Choice('learning_rate', values=[0.0002, 0.0003, 0.0004]))

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

        log_event('    -> Returning the model.')

        return model

    def apply_mask(self):
        """
        Summary: 
            Method is responsible of creating a 3D array:
                1st dimension represents the distinct item count.
                2nd dimension represents the maximum item size in the sequence,
                3rd dimension represents the number of features.

                Suppose user Kaan has 3 different sessions and max item size in the sequence is 5. 
                    Kaan has visited 2 items in session 1, 
                                     4 items in session 2,
                                     1 items in session 3.

                In the 3D array, Kaan's sessions will be represented as follows:
                    1st session = [-9, -9, -9, item1, item2]
                    2nd session = [-9, item1, item2, item3, item4]
                    3rd session = [-9, -9, -9, -9, item1]
                
                As seen above, items are added in time stamp order. The last item that is visited in the session, will
                be the last item of the array.


            -9s represents the masking value which will have no impact to the RNN and allow us to use old sequence data.

        Workflow:
            1. Create a 3D array of size distinct item count and fill it with mask value.
            2. Iterate through the feature list. 
            3. Check if the user has longer item count in her/his session.
            4. If yes, divide the session in a way that keep the last visited items.
            5. Copy the user items into 3D array

        Returns:
            masked_features
        """
        log_event('   -> Creating a 3D array and masking the features..')
        masked_features = np.full((len(self.feature_list), self.max_item_in_sequence, 1), fill_value=self.mask_value)

        for s, x in enumerate(self.feature_list):
            seq_len = x.shape[0]
            if seq_len > self.max_item_in_sequence:
                difference = seq_len - self.max_item_in_sequence
                x = x[difference:]
                masked_features[s, -seq_len:, :] = x[:seq_len].reshape(seq_len - difference, 1)
            else:
                masked_features[s, -seq_len:, :] = x[:seq_len].reshape(seq_len, 1)

        log_event('   -> Masked Features are created..')
        return masked_features

    def load_mapped_items(self):
        """
        Summary:
            Method is responsible from loading a pickle file which represents the items.
            """
        items_path = f'D:/Thesis/recsys_thesis/item_dictionary.pickle'
        items_file = open(items_path, 'rb')

        self.mapped_items_dict = pickle.load(items_file)
        self.distinct_item_count = len(self.mapped_items_dict)

    def print_summary(self):
        current_time = datetime.datetime.now()
        print('Model is successfully trained.. ' + str(current_time))
        print('Train on {} samples and Test on {} samples'.format(len(self.train_features), len(self.test_features)))
        print('''Parameters:
            sequence_limit: {}
            test_size: {}
            max_item_in_sequence: {}
            distinct_item_count: {}
            mask_value: {}
            train_shape: {}
            test_shape: {}
            train_target_length: {}
            test_target_length {}'''.format(self.sequence_limit, self.test_size,
                                            self.max_item_in_sequence, self.distinct_item_count, self.mask_value,
                                            self.train_features.shape, self.test_features.shape,
                                            len(self.train_targets), len(self.test_targets)))

        plt.plot(self.model_history.history['accuracy'])
        plt.plot(self.model_history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'hyper_parameter_tuning\model_1\plots\lstm_model_accuracy_' + str(current_time.strftime('%Y-%m-%d_%H-%M-%S')) + '.png', bbox_inches='tight')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f'hyper_parameter_tuning/model_1/plots/lstm_model_loss_' + str(current_time.strftime('%Y-%m-%d_%H-%M-%S')) + '.png', bbox_inches='tight')
        plt.show()

    def save_model(self):
        current_time = datetime.datetime.now()

        model_json = self.lstm_model.to_json()
        model_name = f'hyper_parameter_tuning/model_1/best_model/lstm_version_one_best_model_' + str(current_time.strftime('%Y-%m-%d_%H-%M-%S')) + '.json'
        weight_name = f'hyper_parameter_tuning/model_1/best_model/lstm_version_one_best_model_' + str(current_time.strftime('%Y-%m-%d_%H-%M-%S')) + '.h5'
        with open(model_name, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.lstm_model.save_weights(weight_name)
        print("Saved lstm_version_one_best_model to disk")

    def load_model(self):
        """
            Load the best model that Keras Tuner found and train the model on the training data.
        """
        loaded_tuner = pickle.load(open("D:/Thesis/recsys_thesis/lstm_versions/hyper_parameter_tuning/model_1/tuner_model_one_02-05-2020_16-33-54.pickle", "rb"))
        model = loaded_tuner.get_best_models(num_models=1)[0]
        history = model.fit(self.train_features, np.array(self.train_targets),
                            batch_size=128, epochs=64,
                            verbose=True,
                            validation_data=(self.test_features, np.array(self.test_targets)),
                            shuffle=True)

        self.lstm_model = model
        self.model_history = history
        self.print_summary()
        self.save_model()


# Switch train_model to True in order to create a new Rnn.
# If false, code will load the best model that Keras Tuner found, crate an Rnn with the best model settings and save the model.
train_model = False

lstm_version_1 = LstmV1(sequence_limit=2, test_size=0.20)
lstm_version_1.create_sequences()
lstm_version_1.prepare_data()

if not train_model:
    lstm_version_1.load_model()
else:
    lstm_version_1.fit_model()
