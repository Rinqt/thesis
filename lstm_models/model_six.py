from lstm_helper_utils.lstm import *


class LstmV6(BaseLSTM):
    def __init__(self, model_description, path_params, training_params, hyper_params):

        super().__init__(model_description=model_description,
                         path_params=path_params,
                         training_params=training_params,
                         hyper_params=hyper_params)

        self.auxiliary_dataframe = None
        self.auxiliary_all_features = None
        self.auxiliary_all_targets = None
        self.auxiliary_vector_size = 0
        self.auxiliary_data_feature_list = None

        self.user_sequence_start = None

        self.catalog_items = None
        self.good_catalog_items = None

        self.constant_for_catalog_item = 2.0
        self.constant_for_visible_item = 5.0
        self.constant_for_click_event = 15.0
        self.l2_norm_constant = 0.


    def train_model(self):
        self.log_event('|-- Start to prepare data for training..')

        # Load Auxiliary Data
        self.auxiliary_dataframe = pd.read_pickle(f'D:/Thesis/recsys_thesis/data/auxiliary_df.pickle')

        # Set the auxiliary vector size
        self.auxiliary_vector_size = len(self.auxiliary_dataframe.iloc[0].all_vectors)

        # Check if model folder exist
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
                    try:
                        self.load_best_tune_model(path_split)
                    except IndexError:
                        self.load_fold_k_data_and_fit(split_number=split_count)

                else:
                    # Create a folder for the split data
                    os.mkdir(self.lstm_model_save_dir + path_split + '/')

                    train_data = np.full(4, fill_value=self.mask_value, dtype=object)
                    for idx, split_name in enumerate(['train_split_', 'test_split_']):
                        path_training_data = self.lstm_model_save_dir + 'split_' + str(split_count) + '/' + split_name + str(split_count) + '_all_features.pkl'
                        path_testing_data = self.lstm_model_save_dir + 'split_' + str(split_count) + '/' + split_name + str(split_count) + '_all_targets.pkl'

                        # Step 1: Load the Split Data
                        #self.user_sequences_df = joblib.load('D:/model_data/common_data/folds/' + split_name + str(split_count) + '.fold')

                        self.user_sequence_list = self.user_sequences_df['item_sequence'].values
                        self.catalog_items = self.user_sequences_df['catalog_item_list'].values
                        self.good_catalog_items = self.user_sequences_df['good_catalog_items'].values
                        self.user_sequence_start = self.user_sequences_df['session_start_time'].values

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

    def create_features_and_targets(self):
        self.log_event('|----> Creating feature and label list for user sequences..')

        self.auxiliary_all_targets = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype=object)

        self.all_features = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype=object)
        self.all_targets = np.full(len(self.user_sequence_list), fill_value=self.mask_value, dtype='int32')

        for index, sequence in enumerate(self.user_sequence_list):
            new_sequence = np.full(len(sequence) - 1, fill_value=self.mask_value, dtype='int32')
            for idx, item in enumerate(sequence[:-1]):
                new_sequence[idx] = item

            self.all_features[index] = new_sequence
            self.all_targets[index] = sequence[-1]
            self.auxiliary_all_targets[index] = self.find_auxiliary_data(item=sequence[-1], seq_start=self.user_sequence_start[index][-1])

        del self.user_sequence_list
        self.log_event('|--------+ Memory Management: user_sequence_list has been removed from memory..')
        self.log_event('|----> Features and targets are ready..')

    def find_auxiliary_data(self, item, seq_start):
        # Find the auxiliary input for the respective input
        respective_df_rows = self.auxiliary_dataframe[self.auxiliary_dataframe['item_id'] == item]

        if len(respective_df_rows) == 0:
            return np.zeros(self.auxiliary_vector_size, dtype='float32')

        correct_time_stamp_item = respective_df_rows[(respective_df_rows['valid_from'] <= seq_start.date())]

        if len(correct_time_stamp_item) > 0:
            if len(correct_time_stamp_item[correct_time_stamp_item['valid_to'].isnull()]) > 0:
                return correct_time_stamp_item[correct_time_stamp_item['valid_to'].isnull()].iloc[0].all_vectors
            else:
                if len(respective_df_rows[(respective_df_rows['valid_from'] <= seq_start.date()) & (seq_start.date() <= respective_df_rows['valid_to'])]) > 1:
                    # Take the mean of the all item vectors
                    all_possible_items = respective_df_rows[(respective_df_rows['valid_from'] <= seq_start.date()) & (seq_start.date() <= respective_df_rows['valid_to'])]
                    self.get_mean_of_vectors(all_possible_items)

        return np.zeros(self.auxiliary_vector_size, dtype='float32')

    def get_mean_of_vectors(self, items):
        # Create item vectors
        item_vectors = items['all_vectors'].to_numpy()

        # Calculate the mean of the vectors and return it
        avg_vector = item_vectors / len(item_vectors)
        return avg_vector

    def encode_features(self):
        self.log_event('   -> K Hot Encoding the features..')

        self.auxiliary_all_features = np.zeros(len(self.all_features), dtype=object)

        user_count = 0
        for individual_sequence in self.all_features:
            encoded_sequence = np.full(shape=(len(individual_sequence), self.number_of_distinct_items), fill_value=-9.0, dtype='float32')
            encoded_auxiliary_info = np.full(shape=(len(individual_sequence), self.auxiliary_vector_size), fill_value=-9.0, dtype='float32')

            catalog_counter = 0
            for idx, item in enumerate(individual_sequence):
                if item != 0:
                    encoded_sequence[idx] = self.encode_item(item_id=item, encoding_dict=self.item_dictionary)

                    # Find corresponding item information
                    encoded_auxiliary_info[idx] = self.find_auxiliary_data(item=item, seq_start=self.user_sequence_start[user_count][idx])

                else:
                    encoded_sequence[idx], encoded_auxiliary_info[idx], catalog_counter = self.encode_catalog(catalog_items_for_the_session=self.catalog_items[user_count],
                                                                                                              good_catalog_item_for_the_session=self.good_catalog_items[user_count],
                                                                                                              catalog_counter=catalog_counter,
                                                                                                              user_count=user_count,
                                                                                                              idx=idx)

            self.all_features[user_count] = encoded_sequence
            self.auxiliary_all_features[user_count] = encoded_auxiliary_info

            user_count += 1

            if user_count % 5000 == 0:
                self.log_event('/\/\/\/\/\/\/\/\/^--->' + str(user_count))

    def encode_catalog(self, catalog_items_for_the_session, good_catalog_item_for_the_session, catalog_counter, user_count, idx):
        catalog_vector = np.zeros(self.number_of_distinct_items, dtype='float32')
        catalog_auxiliary_info = np.zeros(self.auxiliary_vector_size, dtype='float32')

        if isinstance(catalog_items_for_the_session, list):
            corresponding_catalog_page = catalog_items_for_the_session[catalog_counter]
            if corresponding_catalog_page != 'No Catalog Item' and corresponding_catalog_page != 'No Item Found':
                for cat_item in corresponding_catalog_page:
                    cat_item_id = cat_item[0]
                    encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(cat_item_id)]
                    catalog_vector[int(encoding)] = encoding

                    # Find corresponding item information
                    catalog_auxiliary_info += self.find_auxiliary_data(item=cat_item_id, seq_start=self.user_sequence_start[user_count][idx])

                if isinstance(good_catalog_item_for_the_session, list):
                    corresponding_good_catalog_items = good_catalog_item_for_the_session[catalog_counter]
                    if corresponding_good_catalog_items != 'No Item Found':
                        for good_item in corresponding_good_catalog_items:
                            good_item_id = good_item[0]
                            status = good_item[1]
                            encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(good_item_id)]
                            if status == 1:
                                catalog_vector[encoding] = self.constant_for_click_event
                            if status == 0 and catalog_vector[encoding] != self.constant_for_click_event:
                                catalog_vector[encoding] = self.constant_for_visible_item

                                # Find corresponding item information
                                catalog_auxiliary_info += self.find_auxiliary_data(item=good_item_id, seq_start=self.user_sequence_start[user_count][idx])

                temp = [0.0, 1.0, self.constant_for_catalog_item, self.constant_for_visible_item, self.constant_for_click_event]
                catalog_vector = np.array([self.constant_for_catalog_item if item not in temp else item for item in catalog_vector])
                catalog_auxiliary_info = np.array([self.constant_for_catalog_item if item not in temp else item for item in catalog_auxiliary_info])

                # Scale down the values to have sum of vector == 1
                catalog_vector = self.scale_down_vector(vector_to_calc_l2=catalog_vector)
                catalog_auxiliary_info = self.scale_down_vector(vector_to_calc_l2=catalog_auxiliary_info)

                #temp_au = [0, 1]
                #catalog_auxiliary_info = np.array([1 if item not in temp_au else item for item in catalog_auxiliary_info])

                catalog_counter += 1
                return catalog_vector, catalog_auxiliary_info, catalog_counter

        catalog_counter += 1
        return np.full(shape=self.number_of_distinct_items, fill_value=-9.0, dtype='float32'), np.full(shape=self.auxiliary_vector_size, fill_value=-9.0, dtype='float32'), catalog_counter
