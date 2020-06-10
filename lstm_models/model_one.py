from lstm_helper_utils.lstm import *

class LstmV1(BaseLSTM):
    def __init__(self, model_description, path_params, training_params, hyper_params):

        super().__init__(model_description=model_description,
                         path_params=path_params,
                         training_params=training_params,
                         hyper_params=hyper_params)

    def train_model(self):
        self.log_event('|-- Start to prepare data for training..')

        if not os.path.isfile(self.path_lstm_model_feature_list):

            # Step 1: Load Necessary Data into Memory
            self.load_data()

            # Step 2: Save the raw sequence list into project folder
            self.save_data_to_disk(data_to_save=self.user_sequence_list, save_dir=self.lstm_model_save_dir,
                              model_name=self.lstm_model_description, data_name='user_sequence_list.pkl')

            # Step 2.1: Save the evaluation dataframe into project folder
            self.save_data_to_disk(data_to_save=self.evaluation_data, save_dir=self.lstm_model_save_dir,
                              model_name=self.lstm_model_description, data_name='user_sequence_list_for_evaluation.pkl')
            del self.evaluation_data

            # Step 3: Create Feature and Target list
            self.create_features_and_targets()

            # Step 4: Apply Mask to Features
            self.apply_mask(all_features=self.all_features, number_of_features=1)

            # Step 5: One-Hot-Encode the Targets
            self.encode_targets()

            # Step 6: Save the Prepared Data into project folder
            self.save_data_to_disk(data_to_save=self.all_features, save_dir=self.lstm_model_save_dir,
                              model_name=self.lstm_model_description, data_name='masked_feature_list.pkl')

            self.save_data_to_disk(data_to_save=self.all_targets, save_dir=self.lstm_model_save_dir,
                              model_name=self.lstm_model_description,  data_name='encoded_target_list.pkl')

            # Step 7: Divide data into folds and save in the disk
            self.create_folds()
            self.log_event('|-- Data has been divided into folds and save to the disk..')

            # Step 8: Save necessary information to use later when we load the data
            joblib.dump(value=self.retrieve_base_info_dictionary(), filename=self.lstm_model_save_dir + 'data_information.pkl')

        else:
            info_dict = joblib.load(self.lstm_model_save_dir + 'data_information.pkl')
            info_dict = self.load_object_parameters(info_dict=info_dict)

        self.log_event('|-- Data is ready to be trained..')

        self.fit_model()

    def add_mask_layer(self):
        return Masking(mask_value=self.mask_value, input_shape=(self.max_sequence_size, 1))

    def create_features_and_targets(self):
        self.log_event('|----> Creating feature and label list for user sequences..')

        self.all_features = np.empty(len(self.user_sequence_list), dtype=object)
        self.all_targets = np.empty(len(self.user_sequence_list), dtype='int32')
        counter = 0

        for index, sequence in enumerate(self.user_sequence_list):
            is_only_item_sequence = False

            new_sequence = np.full(len(sequence) - 1, fill_value=self.mask_value, dtype='float32')
            for idx, item in enumerate(sequence[:-1]):
                if item != 0:
                    new_sequence[idx] = item
                    is_only_item_sequence = True

            if is_only_item_sequence:
                self.all_features[index] = new_sequence
                self.all_targets[index] = sequence[-1]
                counter += 1


        temp_features = np.empty(counter, dtype=object)
        temp_targets = np.empty(counter, dtype='int32')
        c = 0
        for feature, target in zip(self.all_features, self.all_targets):
            if feature is not None:
                temp_features[c] = feature
                temp_targets[c] = target
                c += 1

        del self.user_sequence_list

        self.all_features =temp_features
        self.all_targets = temp_targets

        self.log_event('|--------+ Memory Management: user_sequence_list has been removed from memory..')

        self.log_event('|----> Features and targets are ready..')



