## Process MySQL data

import mysql.connector
import json

db_names = ['google_play_all', 'mac_all_data']    # , 'pc_windows_data'
ad_terms = ['ad', 'ads', 'advert']



def check_ad(text):
    terms = text.strip().split(' ')
    for term in terms:
        if term in ['ad','ads','commercial','commercials'] or term.startswith('advert') :
            return True
    return False

def connect_db(db_name, ad_dict, ad_count):
    cursor = cnx.cursor()
    cursor.execute("USE %s" % db_name)
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()  ## get table names

    ad_dict[db_name] = {}
    for tb_name in tables:
        query = ("SELECT * FROM %s")
        try:
            cursor.execute(query % tb_name)
        except mysql.connector.errors.ProgrammingError:
            print(tb_name)
            continue
        lines = cursor.fetchall()
        for line in lines:
            try:
                text = line[2] + ' ' +line[3]
            except TypeError as e:   # The error usually occurs to Chinese reviews
                continue
            ad_flag = check_ad(text)
            if ad_flag:
                if tb_name[0] not in ad_dict[db_name]:
                    ad_dict[db_name][tb_name[0]] = []
                d = {}
                if db_name == 'mac_all_data':
                    d["author"] = line[1]
                    d["review"] = line[2]
                    d["title"] = line[3]
                    d["date"] = line[4]
                    d["stars"] = line[5]
                elif db_name == 'google_play_all':
                    d["author"] = line[3]
                    d["review"] = line[5]
                    d["title"] = line[4]
                    d["date"] = line[6]
                    d["stars"] = line[7]
                else:
                    d["author"] = line[1]
                    d["review"] = line[2]
                    d["title"] = line[3]
                    d["date"] = line[4]
                    d["stars"] = line[5]
                ad_count += 1
                ad_dict[db_name][tb_name[0]].append(d)
            else:
                continue
    cursor.close()
    return ad_dict, ad_count


if __name__ == "__main__":
    cnx = mysql.connector.connect(user='root', password='100861992', host='127.0.0.1')
    fw = open('../sql_data/ad_review_google_mac.json', 'w')
    ad_count = 0
    ad_dict = {}
    ad_dict, ad_count = connect_db(db_names[0], ad_dict, ad_count)
    ad_dict, ad_count = connect_db(db_names[1], ad_dict, ad_count)
    print("No. of ad reviews is ", ad_count)
    cnx.close()
    json.dump(ad_dict, fw, indent=4)
    fw.close()