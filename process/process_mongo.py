## Process the data in local MongoDB

import json
from pymongo import MongoClient
from process_sql import check_ad

def get_review(collects, gp_ad_reviews, PLATFORM="gp"):
    ad_count = 0
    all_count = 0
    gp_ad_reviews[PLATFORM] = {}
    for collect in collects:
        if collect == "clean_master":
            continue
        if PLATFORM == "gp":
            db = google_db
        else:
            db = ios_db
        for article in db[collect].find():
            all_count += 1
            text = article["title_review"]
            ad_flag = check_ad(text)
            if ad_flag:
                d = {}
                d["title"] = article["title_review"].split("\n")[0]
                d["review"] = article["title_review"].split("\n")[-1]
                d["date"] = article["date"]
                d["rate"] = article["stars"]
                d["version"] = article["version"]
                if collect not in gp_ad_reviews[PLATFORM]:
                    gp_ad_reviews[PLATFORM][collect] = []
                gp_ad_reviews[PLATFORM][collect].append(d)
                ad_count += 1
    print("Number of all reviews on ", PLATFORM, all_count)
    return gp_ad_reviews, ad_count

if __name__ == "__main__":
    client = MongoClient('localhost', 27017)
    google_db = client.appannie_google
    ios_db = client.appannie_ios

    gp_collects = google_db.list_collection_names()
    ios_collects = ios_db.list_collection_names()

    gp_ad_reviews = {}
    ios_ad_reviews = {}

    gp_ad_reviews, gp_ad_count = get_review(gp_collects, gp_ad_reviews, PLATFORM="gp")
    print("No. of Google ad reviews is ", gp_ad_count)
    fw = open('../platform_data/gp_mongo_ad_reviews.json', 'w')
    json.dump(gp_ad_reviews, fw, indent=4)
    fw.close()

    ios_ad_reviews, ios_ad_count = get_review(ios_collects, ios_ad_reviews, PLATFORM="ios")
    print("No. of Ios ad reviews is ", ios_ad_count)
    fw = open('../platform_data/ios_mongo_ad_reviews.json', 'w')
    json.dump(ios_ad_reviews, fw, indent=4)
    fw.close()
