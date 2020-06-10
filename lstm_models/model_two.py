from lstm_helper_utils.lstm import *


class LstmV2(BaseLSTM):
    def __init__(self, model_description, path_params, training_params, hyper_params):

        super().__init__(model_description=model_description,
                         path_params=path_params,
                         training_params=training_params,
                         hyper_params=hyper_params)

    def train_model(self):
        self.log_event('|-- Start to prepare data for training..')

        if not self.check_file_exists(self.lstm_model_save_dir):
            # Create the model folder
            os.makedirs(self.lstm_model_save_dir)

            # Load the Train data and create the data_information.pkl
            self.load_train_data()

        # Check if the necessary pickle file exist to start training
        if self.check_file_exists(self.path_data_information):

            # Step 1: Load Necessary Data into Memory
            self.load_train_data()

            for split_count in range(self.k_split):
                index = 0
                path_split = 'split_' + str(split_count)

                if self.check_file_exists(self.lstm_model_save_dir + path_split):
                    if self.check_file_exists(self.lstm_tuner_save_dir + '/' + path_split):
                        try:
                            self.load_best_tune_model(path_split)
                        except IndexError:
                            self.load_fold_k_data_and_fit(split_number=split_count)
                    else:
                        self.load_fold_k_data_and_fit(split_number=split_count)
                        continue
                else:
                    # Create a folder for the split data
                    os.mkdir(self.lstm_model_save_dir + path_split + '/')

                    train_data = np.full(4, fill_value=self.mask_value, dtype=object)
                    for idx, split_name in enumerate(['train_split_', 'test_split_']):
                        path_training_data = self.lstm_model_save_dir + 'split_' + str(split_count) + '/' + split_name + str(split_count) + '_all_features.pkl'
                        path_testing_data = self.lstm_model_save_dir + 'split_' + str(split_count) + '/' + split_name + str(split_count) + '_all_targets.pkl'

                        # Step 1: Load the Split Data
                        self.user_sequences_df = joblib.load('D:/model_data/common_data/folds/' + split_name + str(split_count) + '.fold')
                        self.user_sequence_list = self.user_sequences_df['item_sequence'].values

                        # Step 2: Create Features and Targets
                        self.create_features_and_targets()

                        # Step 3: Encode Features
                        self.encode_features()

                        # Step 4: Apply Mask to All Features
                        self.apply_mask(all_features=self.all_features, number_of_features=self.number_of_distinct_items)

                        # Step 5: Encode Targets
                        self.encode_targets()

                        # Step 6: Save Prepared Split Data to Disk for future trainings
                        train_data[index] = self.all_features
                        self.save_data_to_disk(data_to_save=self.all_features, data_name=path_training_data)
                        index += 1

                        train_data[index] = self.all_targets
                        self.save_data_to_disk(data_to_save=self.all_targets, data_name=path_testing_data)
                        index += 1

                    # Step 7: Fit the Model with the Split Training Data
                    self.fit(train_data, split_count)

            self.retrieve_best_model(self.best_possible_models)

    def encode_features(self):
        """
        Summary:
            Method is responsible of encoding the feature list.

            Example:
                Feature item = 3
                One-Hot-Encoded representation of the item 3:
                    item_1, item_2, item_3, item_4, item_5
                    [  0,      0,      1,      0,      0 ]

                Suppose user has item1, item2 and item5 in her/his sequence:
                    user_sequence = [item1, item2, item5]
                    one_hot_encoded_user_sequence = [ [1, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0],
                                                      [0, 0, 0, 0, 1] ]


        Workflow:
            1. Iterate through feature list.
            2. Create a temp array to keep items of one sequence
            3. Iterate through the items and find the mapped number of the item by using 'mapped_items_dict' dictionary.
            4. Create a vector which has a length of distinct item count.
            5. Change the item location of the vector as 1.
            6. Add the item into temp array.
            7. Change the corresponding location of the feature list with the OHE user sequence

        Returns:
            One-Hot-Encoded target_list
        """
        self.log_event('   -> One Hot Encoding the features..')

        count = 0
        for user_sequence in self.all_features:
            encoded_sequence = np.full(shape=(len(user_sequence), self.number_of_distinct_items), fill_value=self.mask_value, dtype='float32')
            for idx, item in enumerate(user_sequence):
                if item != 0:
                    encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(item)]
                    vector = np.zeros(shape=self.number_of_distinct_items, dtype='float32')
                    vector[int(encoding)] = 1
                    encoded_sequence[idx] = vector

            self.all_features[count] = encoded_sequence
            count += 1

        self.log_event('|----> One Hot Encoding has finished..')

    def add_mask_layer(self):
        return Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, self.number_of_distinct_items))

    def evaluate_model(self):
        model = load_model(self.path_best_lstm_model)
        
        if not os.path.exists(self.lstm_model_save_dir + 'evaluation_ready_sequences.pkl'):
            evaluation_dataframe = self.load_evaluation_data()

            self.all_features = evaluation_dataframe['item_sequence'].values
            evaluation_dataframe['decoded_item_sequence'] = evaluation_dataframe['item_sequence'].copy(deep=True)

            evaluation_dataframe.drop(['session_start_time'], axis=1, inplace=True)
            evaluation_dataframe.drop(['catalog_item_list'], axis=1, inplace=True)
            evaluation_dataframe.drop(['good_catalog_items'], axis=1, inplace=True)

            self.encode_features()
            evaluation_dataframe['item_sequence'] = self.all_features

            sequences = evaluation_dataframe.values.tolist()
            joblib.dump(sequences, self.lstm_model_save_dir + 'evaluation_ready_sequences.pkl')
        else:
            sequences = joblib.load(self.lstm_model_save_dir + 'evaluation_ready_sequences.pkl')
            self.load_object_parameters()

        self.evaluation_creator(model=model,
                                model_name=self.lstm_model_description,
                                sequence_to_evaluate=sequences,
                                evaluate_all_items_in_the_sequence=True)
