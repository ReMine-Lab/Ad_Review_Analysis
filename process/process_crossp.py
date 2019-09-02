## Process cross platform datasets in "./platform_data/"

from argparse import Namespace
from process_sql import check_ad
import os
import json
import csv, random

args = Namespace(
    google_dir='../platform_data/Specific_apps/',
    ios_dir='../platform_data/Specific_apps_iTunes/',
    win_dir='../platform_data/Specific_apps_microsoft/',
)

def get_review(dir, ad_reviews, ad_count):
    all_count = 0
    files = os.listdir(dir)
    platform = dir.split("/")[2]
    ad_reviews[platform] = {}
    for idx, f in enumerate(files):
        fr = open(os.path.join(dir, f),errors='ignore')
        lines = fr.readlines()
        fr.close()
        all_count += len(lines)
        for line in lines:
            terms = line.split('******')
            text = terms[2]+terms[3]
            ad_flag = check_ad(text)
            if ad_flag:
                d = {}
                d["author"] = terms[1]
                d["review"] = terms[2]
                d["title"] = terms[3]
                d["date"] = terms[4]
                d["rate"] = terms[5]
                d["version"] = terms[6].strip('\n')
                if f not in ad_reviews[platform]:
                    ad_reviews[platform][f] = []
                ad_reviews[platform][f].append(d)
                ad_count += 1
                ad_review_list.append([f, terms[2], terms[3], terms[5], terms[4], terms[6].strip('\n'), terms[1]])
    print("Number of all reviews on ", platform, all_count)
    return ad_reviews, ad_count

if __name__ == "__main__":
    ad_reviews = {}
    ad_review_list = []
    ad_count = 0
    ad_reviews, google_ad_count = get_review(args.google_dir, ad_reviews, ad_count)
    print("No. of ad reviews in Google Play is ", google_ad_count)
    ad_reviews, new_ad_count = get_review(args.ios_dir, ad_reviews, google_ad_count)
    ios_ad_count = new_ad_count-google_ad_count
    print("No. of ad reviews in iTunes is ", ios_ad_count)
    fw = open('../platform_data/ad_reviews.json', 'w')
    json.dump(ad_reviews, fw, indent=4)
    fw.close()


    # random_google_list = random.sample(ad_review_list[:google_ad_count], int(google_ad_count*1000/new_ad_count))
    # with open('../platform_data/google_ads.csv', mode='w', newline='') as fw:
    #     writer_ = csv.writer(fw, delimiter=',')
    #     for ad_review in random_google_list:
    #         writer_.writerow(ad_review)
    #
    # random_google_list = random.sample(ad_review_list[google_ad_count:], int(ios_ad_count * 1000/new_ad_count))
    # with open('../platform_data/ios_ads.csv', mode='w', newline='') as fw:
    #     writer_ = csv.writer(fw, delimiter=',')
    #     for ad_review in random_google_list:
    #         writer_.writerow(ad_review)
