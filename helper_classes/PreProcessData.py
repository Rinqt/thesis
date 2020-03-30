import gensim
import pandas as pd
from gensim.models.doc2vec import TaggedDocument


def concat_user_sequences(sequence_low_limit=3):
    """
        sequence_low_limit = Parameter to control sequence length.
        Method is used to create following data:
            userID | all sequences belongs to a user

        1. List of items which user visited is fetched from the database.
        2. Create a DataFrame with the data.
        3. Aggregate item numbers into a list (sequences list) which belongs to same session for each user.
        4. Drop the sequences which has less than 'sequence_low_limit'
        5. Aggregate all sequences which belongs to same user together.

        At the end, method will return a DataFrame similar to following:
                    <userID>     <itemSequence>
                     12345   [ [seq_1], [seq_2], [seq_3] ]

        --------
        Returns: DataFrame
    """
    from helper_classes.DatabaseHelper import fetch_visited_items_by_user
    import pandas as pd

    columns = ['userID', 'objectID', 'sessionID']

    items_visited_list = fetch_visited_items_by_user()

    # Below line is for fast testing:
    #item_visit_df = pd.DataFrame(data=items_visited_list[:100], columns=columns)
    item_visit_df = pd.DataFrame(data=items_visited_list, columns=columns)

    item_sequence_df = item_visit_df.groupby(['userID', 'sessionID'], as_index=False).agg(lambda x: list(map(str, x)))
    item_sequence_df.rename(columns={'objectID': 'itemSequence'}, inplace=True)
    item_sequence_df = item_sequence_df.drop(['sessionID'], axis=1)
    item_sequence_df['length'] = item_sequence_df['itemSequence'].str.len()
    item_sequence_df = item_sequence_df.drop(item_sequence_df[item_sequence_df.length < sequence_low_limit].index)

    final_df = item_sequence_df.groupby(['userID'], as_index=False).agg(lambda x: list(map(list, x)))
    return final_df


def prepare_word2vec_train_data(sequence_data):
    """
        Method is responsible of creating a list which contains item ids that user have visited.
        Every item id is represented as string.
        Word2Vec expects an input as following
            list => ['itemid1', 'itemid2', 'itemid3']
            Basically, an iterable with strings inside.

        1. Iterate on 'sequence_data'.
        2. Get the second index, sequence_data[2] since it is basically a tuple which contains item_id and [sequences]
        3. Iterate on sequence list and append each sequence to a new list which will be returned at the end.

        --------
        Returns: list

    """
    item_word_list = []
    for data in sequence_data:
        sequence_list = data[1]
        for sequence in sequence_list:
            str_concat = ','.join(map(str, sequence))
            qwe = str_concat.split(',')

            item_word_list.append(qwe)

    return item_word_list


def clean_text(df):
    df_clean = pd.DataFrame({'clean': df})
    df_clean.loc[:, "clean"] = df_clean.clean.apply(lambda x: str.lower(x))
    df_clean.loc[:, "clean"] = df_clean.clean.apply(lambda x: remove_stop_words(x))

    df = df_clean["clean"]
    return df

def read_corpus(df):
    """
    :summary: Tag the documents to be used in Doc2Vec Model Training.
    :param df: dataframe
    :return:
    """
    for i, line in enumerate(df):
        tokens = gensim.utils.simple_preprocess(line)
        item_id = line.split(',')[0]
        yield TaggedDocument(tokens, [item_id])


def remove_stop_words(_s):
    from stop_words import get_stop_words
    from string import punctuation
    import re

    stop_words = get_stop_words('czech')

    _s = ' '.join(word for word in _s.split() if word not in (stop_words and punctuation))

    punctuation = r"""!"#$%&'()*+-./:;<=>?@[\]^_`{|}~"""  # Remove comma from the default set
    table = str.maketrans(dict.fromkeys(punctuation))
    new_s = _s.translate(table)

    # Remove the html tags
    cleaned_s = re.sub('<[^<]+?>', '', new_s)
    return cleaned_s
