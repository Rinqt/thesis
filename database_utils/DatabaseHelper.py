import mysql.connector
from mysql.connector import errorcode

config = {
  'user': 'root',
  'password': 'password',
  'host': '127.0.0.1',
  'database': 'travel_agency',
  'raise_on_warnings': True,
  'auth_plugin': 'mysql_native_password'
}


def create_connection():
    """
        Method is used to establish connection with a database.

        -------
        Returns: Connection Object
    """
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)



def fetch_user_item_seq_info_with_date():
    query_user_ids = """ Select userID, objectID, sessionID, startDatetime from travel_agency.implicit_events order by userID, startDatetime; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [(user_id, item_id, session_id, startDatetime) for user_id, item_id, session_id, startDatetime in data_tuple]
    return data

def fetch_visited_items_by_user():
    """
        Method fetch the following data:
            userID | objectID | sessionID

        After data is fetched, it is transformed into a list which will be returned at the end.

        -----
        Returns: list
    """
    query_user_ids = """ SELECT userID, objectID, sessionID 
                         FROM travel_agency.implicit_events 
                         ORDER BY userID, startDatetime """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [(user_id, item_id, session_id) for user_id, item_id, session_id in data_tuple]
    return data


def fetch_user_sessions():
    """
        Method fetch the following data:
            userID | sessionID | objectID | pageType

        After data is fetched, it is transformed into a list which will be returned at the end.

        -----
        Returns: list
    """
    query_user_ids = """ SELECT userID, sessionID, objectID, pageType, objectsListed, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, logFile
                         FROM travel_agency.implicit_events 
                         ORDER BY userID, startDatetime """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    returned_data = cursor.fetchall()
    data = [(user_id, item_id, session_id, page_type, objectsListed, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, logFile) for user_id, item_id, session_id, page_type, objectsListed, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, logFile in returned_data]
    return data



def find_object_id(item_id):
    query_user_ids = """ SELECT objectID 
                         FROM travel_agency.implicit_events 
                         WHERE objectID = %s;"""

    conn = create_connection()
    cursor = conn.cursor()

    data_set = (item_id,)
    cursor.execute(query_user_ids, data_set)

    result = cursor.fetchone()

    # Return 0 if item does not exist, otherwise return 1
    if result is None:
        return 0
    else:
        return 1


def fetch_all_items():
    query_user_ids = """ SELECT DISTINCT (objectID) 
                         FROM travel_agency.implicit_events 
                         WHERE objectID <> 0 ORDER BY objectID;"""

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [item for t in data_tuple for item in t]
    return data




def insert_evaluation(user_id, session_id, precision, recall, mrr, ndcg, predictor_name, trivial_prediction, catalog_count, ground_truth, sequence, input_sequence, predictions):
    query = """ INSERT INTO travel_agency.model_evaluation 
                       (user_id, session_id, travel_agency.model_evaluation.precision, recall, mrr, ndcg, predictor_name, 
                        trivial_prediction, catalog_count, ground_truth, sequence, input_sequence, predictions) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s); """

    conn = create_connection()
    cursor = conn.cursor()

    data = (user_id, session_id, precision, recall, mrr, ndcg, predictor_name, trivial_prediction, catalog_count, ground_truth, sequence, input_sequence, predictions)
    cursor.execute(query, data)
    conn.commit()

    cursor.close()


def get_evaluation_score(model_name):
    query = """ SELECT user_id, avg_precision, avg_recall, avg_mrr 
                FROM travel_agency.performance_metrics 
                WHERE model_name = %s  ORDER BY avg_precision DESC"""

    conn = create_connection()
    cursor = conn.cursor()

    data_set = (model_name,)
    cursor.execute(query, data_set)

    result = cursor.fetchall()
    return result


def fetch_travel_packages_data():
    query = """ SELECT DISTINCT(id_serial), nazev, popisek, do
                FROM travel_agency.new_serial_table ORDER BY id_serial; """

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute(query)

    result = cursor.fetchall()
    return result


def fetch_user_item_seq_info():
    query_user_ids = """ Select userID, sessionID, objectID from travel_agency.implicit_events where objectID <> 0 order by userID; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [(user_id, session_id, item_id) for user_id, session_id, item_id, in data_tuple]
    return data


def get_all_users_with_logs():
    query_user_info = """ SELECT userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, objectsListed,  logFile
                         FROM travel_agency.implicit_events 
                         WHERE objectID = 0 AND objectsListed <> ''
                         ORDER BY userID; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_info)

    data_tuple = cursor.fetchall()
    data = [(userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, objectsListed, logFile) for userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, objectsListed, logFile in data_tuple]
    return data

def get_all_catalog_items_less_columns():
    query_user_info = """ SELECT userID, objectID, sessionID, objectsListed
                         FROM travel_agency.implicit_events 
                         WHERE pageType = 'katalog' OR pageType = 'index' AND LENGTH(objectsListed) - LENGTH(REPLACE(objectsListed, ';', '')) > 1
                         ORDER BY userID; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_info)

    data_tuple = cursor.fetchall()
    data = [(userID, objectID, sessionID, objectsListed) for userID, objectID, sessionID, objectsListed in data_tuple]
    return data


def get_all_catalog_items():
    query_user_info = """ SELECT userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, objectsListed, travel_agency.implicit_events.logFile
                         FROM travel_agency.implicit_events
                         WHERE pageType = 'katalog' OR pageType = 'index' OR pageType = 'informace'
                         ORDER BY userID; """
    #WHERE pageType = 'katalog' OR pageType = 'index' OR pageType = 'informace' AND LENGTH(objectsListed) - LENGTH(REPLACE(objectsListed, ';', '')) > 1
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_info)

    data_tuple = cursor.fetchall()
    data = [(userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, objectsListed, logFile) for userID, objectID, sessionID, windowSizeX, windowSizeY, pageSizeX, pageSizeY, timeOnPage, objectsListed, logFile in data_tuple]
    return data


def get_auxiliary_data():
    query_data = """ SELECT id_serial, id_zajezd, nazev, od, do, ubytovani_kategorie, id_typ, zeme, destinace, prumerna_cena, prumerna_cena_noc, min_cena, sleva, delka, informace_list, valid_from, valid_to
                     FROM travel_agency.new_serial_table
                     ORDER BY id_serial ASC, id_record DESC"""

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_data)

    data_tuple = cursor.fetchall()
    data = [(id_serial, id_zajezd, nazev, od, do, ubytovani_kategorie, id_typ, zeme, destinace, prumerna_cena, prumerna_cena_noc, min_cena, sleva, delka, informace_list, valid_from, valid_to) for id_serial, id_zajezd, nazev, od, do, ubytovani_kategorie, id_typ, zeme, destinace, prumerna_cena, prumerna_cena_noc, min_cena, sleva, delka, informace_list, valid_from, valid_to in data_tuple]
    return data

def split_country_name(c_name, c_list):
    if ':' in c_name:
        countries = c_name.split(':')
        for c in countries:
            if c not in c_list:
                c_list.append(c.lower())
    else:
        if len(c_name) > 0:
            if c_name not in c_list:
                c_list.append(c_name.lower())

def get_distinct_info():
    query_data = """ SELECT DISTINCT zeme, sleva, ubytovani_kategorie, id_typ, delka, informace_list, destinace
                     FROM travel_agency.new_serial_table """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_data)

    data_tuple = cursor.fetchall()

    dest_country, discount_types, accommodation_type, id_type, duration = [], [], [], [], []

    for i in data_tuple:

        country = i[0].lower()
        discount = int(i[1])
        accommodation = int(i[2])
        id_typ = i[3]
        length = i[4]
        follow_up_destination = i[5].lower()
        dest = i[6].lower()

        split_country_name(country, dest_country)
        split_country_name(follow_up_destination, dest_country)
        split_country_name(dest, dest_country)

        if discount not in discount_types:
            discount_types.append(discount)

        if accommodation not in accommodation_type:
            accommodation_type.append(accommodation)

        if id_typ not in id_type:
            id_type.append(id_typ)

        if length not in duration:
            duration.append(length)

    destinations = sorted(dest_country)
    discount_types = sorted(discount_types)
    accommodation_type = sorted(accommodation_type)
    id_type = sorted(id_type)
    duration = sorted(duration)

    return destinations, discount_types, accommodation_type, id_type, duration

