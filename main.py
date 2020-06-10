from lstm_models.model_one import LstmV1
from lstm_models.model_two import LstmV2
from lstm_models.model_five import LstmV5
from lstm_models.model_six import LstmV6


hyper_params = {
    'objective_function': 'val_accuracy',
    'lstm_units': {'min':32, 'max':128, 'step':8},
    'dropout': [0.1, 0.3, 0.5],
    'lstm_layer_activation': 'relu',
    'lstm_layer_dropout': 0.01,
    'dense_activation': 'softmax',
    'learning_rate': [0.0001, 0.001, 0.01],
    'metric': ['accuracy'],
    'loss': 'categorical_crossentropy',
}

def create_model_data(model_parent_folder_name, file_name, model_name, tuner_directory):
    model_parameters = {
        'model_directory': f'D:/model_data/' + model_parent_folder_name + '/',
        'lstm_tuner_save_dir': 'D:\\model_data\\' + model_parent_folder_name + '\\hp_tuning\\' + tuner_directory,
        'path_lstm_model_feature_list': f'D:/model_data/common_data/user_sequence_list.pkl',
        'path_lstm_model_targets_list': f'D:/model_data/' + model_parent_folder_name + '/' + file_name + '_encoded_target_list.pkl',
        'path_data_information': f'D:/model_data/' + model_parent_folder_name + '/' + file_name + '_data_information.pkl',
        'path_best_lstm_model': f'D:/model_data/' + model_parent_folder_name + '/' + model_name + '_best_model.h5',
        'path_dataframe_to_train': f'D:/model_data/common_data/user_sequence_dataframe.pkl',
        'path_dataframe_evaluation': f'D:/model_data/common_data/evaluation_dataframe.pkl',
        'path_item_dictionary': f'D:/model_data/items.dictionary',
        'path_user_sequences_dataframe': f'D:/model_data/preprocessed_dataframe.pickle',
    }

    return model_parameters

models_dict = {'LstmV2': LstmV2}

models_info = {
    'model_five_512_batch_size_8_epoch' : {'test_fraction' : 0.25, 'evaluation_fraction' : 0.25, 'max_trials' : 3, 'execution_per_trial' : 3, 'epochs' : 8, 'batch_size' : 512, 'k_split' : 4, 'random_state' : 42},
}

for i, model_class in enumerate(models_dict.values()):

    for model_type in models_info:
        p_params = create_model_data('model_two', file_name='model_two', model_name=model_type, tuner_directory=model_type)
        model_object = model_class(model_description=model_type,
                                   path_params=p_params,
                                   training_params=models_info[model_type],
                                   hyper_params=hyper_params)

        model_object.train_model()
        model_object.evaluate_model()
