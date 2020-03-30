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


def fetch_visited_items_by_user():
    """
        Method fetch the following data:
            userID | objectID | sessionID

        After data is fetched, it is transformed into a list which will be returned at the end.

        -----
        Returns: list
    """
    query_user_ids = """ SELECT userID, objectID, sessionID 
                         FROM travel_agency.new_implicit_events 
                         WHERE objectID <> 0
                         ORDER BY userID; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [(user_id, item_id, session_id) for user_id, item_id, session_id in data_tuple]
    return data


def find_object_id(item_id):
    query_user_ids = """ SELECT objectID 
                         FROM travel_agency.new_implicit_events 
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
    query_user_ids = """ SELECT objectID 
                         FROM travel_agency.new_implicit_events 
                         WHERE objectID <> 0;"""

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    #data = [list(item_id) for item_id in data_tuple]
    data = [item for t in data_tuple for item in t]
    return data


def insert_evaluation(user_id, precision, recall, mrr, model_name):
    query = """ INSERT INTO travel_agency.performance_metrics 
                       (user_id, avg_precision, avg_recall, avg_mrr, model_name) 
                VALUES (%s, %s, %s, %s, %s); """

    conn = create_connection()
    cursor = conn.cursor()

    data = (user_id, precision, recall, mrr, model_name)
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
    query_user_ids = """ Select userID, sessionID, objectID from travel_agency.new_implicit_events order by userID; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [(user_id, session_id, item_id) for user_id, session_id, item_id, in data_tuple]
    return data


def fetch_user_item_seq_info_with_date():
    query_user_ids = """ Select userID, sessionID, objectID, startDatetime, endDatetime from travel_agency.new_implicit_events order by userID; """

    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(query_user_ids)

    data_tuple = cursor.fetchall()
    data = [(user_id, session_id, item_id, startDatetime, endDatetime) for user_id, session_id, item_id, startDatetime, endDatetime in data_tuple]
    return data