from lstm_helper_utils.lstm import *


class LstmV5(BaseLSTM):
    def __init__(self, model_description, path_params, training_params, hyper_params):

        super().__init__(model_description=model_description,
                         path_params=path_params,
                         training_params=training_params,
                         hyper_params=hyper_params)

        self.catalog_items = None
        self.good_catalog_items = None

        self.constant_for_catalog_item = 2.0
        self.constant_for_visible_item = 5.0
        self.constant_for_click_event = 15.0
        self.l2_norm_constant = 0.

    def train_model(self):
        self.log_event('|-- Start to prepare data for training..')

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
                        self.user_sequences_df = joblib.load('D:/model_data/common_data/folds/' + split_name + str(split_count) + '.fold')

                        self.user_sequence_list = self.user_sequences_df['item_sequence'].values
                        self.catalog_items = self.user_sequences_df['catalog_item_list'].values
                        self.good_catalog_items = self.user_sequences_df['good_catalog_items'].values

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
        user_count = 0
        for individual_sequence in self.all_features:
            encoded_sequence = np.full(shape=(len(individual_sequence), self.number_of_distinct_items), fill_value=-9.0, dtype='float32')
            catalog_counter = 0

            for idx, item in enumerate(individual_sequence):
                if item != 0:
                    encoded_sequence[idx] = self.encode_item(item_id=item, encoding_dict=self.item_dictionary)
                else:
                    encoded_sequence[idx], catalog_counter = self.encode_catalog(catalog_items_for_the_session=self.catalog_items[user_count],
                                                                                 good_catalog_item_for_the_session=self.good_catalog_items[user_count],
                                                                                 catalog_counter=catalog_counter)

            self.all_features[user_count] = encoded_sequence

            user_count += 1

            if user_count % 5000 == 0:
                self.log_event('       ' + str(user_count))

        del self.good_catalog_items, self.catalog_items
        self.log_event('   ----> self.good_catalog_items, self.catalog_items objects are deleted for memory management..!')
        self.log_event('   -> K Hot Encoding is over..')

    def encode_catalog(self, catalog_items_for_the_session, good_catalog_item_for_the_session, catalog_counter):
        catalog_vector = np.zeros(self.number_of_distinct_items, dtype='float32')

        if isinstance(catalog_items_for_the_session, list):
            corresponding_catalog_page = catalog_items_for_the_session[catalog_counter]
            if corresponding_catalog_page != 'No Catalog Item' and corresponding_catalog_page != 'No Item Found':
                for cat_item in corresponding_catalog_page:
                    cat_item_id = cat_item[0]
                    encoding = list(self.item_dictionary.keys())[list(self.item_dictionary.values()).index(cat_item_id)]
                    catalog_vector[int(encoding)] = encoding

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

                temp = [0.0, 1.0, self.constant_for_catalog_item, self.constant_for_visible_item, self.constant_for_click_event]
                catalog_vector = np.array([self.constant_for_catalog_item if item not in temp else item for item in catalog_vector])

                # Scale down the values to have sum of vector == 1
                catalog_vector = self.scale_down_vector(vector_to_calc_l2=catalog_vector)

                catalog_counter += 1
                return catalog_vector, catalog_counter

        catalog_counter += 1
        return np.full(shape=self.number_of_distinct_items, fill_value=-9.0, dtype='float32'), catalog_counter

    def add_mask_layer(self):
        return Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, self.number_of_distinct_items))

    def evaluate_model(self):
        model = load_model(self.path_best_lstm_model)

        if not os.path.exists(self.lstm_model_save_dir + 'evaluation_ready_sequences.pkl'):
            evaluation_dataframe = self.load_evaluation_data()

            self.all_features = evaluation_dataframe['item_sequence'].values
            self.catalog_items = evaluation_dataframe['catalog_item_list'].values
            self.good_catalog_items = evaluation_dataframe['good_catalog_items'].values
            evaluation_dataframe['decoded_item_sequence'] = evaluation_dataframe['item_sequence'].copy(deep=True)

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
