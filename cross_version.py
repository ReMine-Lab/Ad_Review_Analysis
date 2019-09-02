# Analyze the cross version performance, specifically whether one issue is fixed by developers

import xlrd
import csv, os
from argparse import Namespace
from pymongo import MongoClient
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


args = Namespace(
    google_dir='./platform_data/Specific_apps/',
    ios_dir='./platform_data/Specific_apps_iTunes/',
    win_dir='./platform_data/Specific_apps_microsoft/',
)
all_app_names = ['Camera','Duolingo','eBay','Evernote','Facebook','HERE','Instagram','LINE','Messenger','Netflix','Skype','Spotify','Tango','TED','Twitter','Viber','VLC','WeChat','WhatsApp','YouTube','Alipay','Minion','Snapchat','Subway','pinterest','Tripadvisor','Trivia','Tom','Amazon','Shareit','Candy','Sound','Waze']
version_apps = ['Evernote', 'Spotify', 'VLC', 'Facebook', 'Tango', 'Camera', 'TED', 'LINE', 'Messenger', 'WeChat', 'HERE']
label_dict = {'content': 0, 'security': 1, 'other': 2, 'crash': 3, 'frequency': 4, 'popup': 5, 'too many': 6, 'non-skippable': 7, 'timing': 8, 'slow': 9, 'size': 10, 'position': 11, 'auto-play': 12, 'landscape': 13, 'notification': 14, 'volume': 15}

review_fr_path = './ad_reviews/labeled_ad_reviews_2.xlsx'
workbook = xlrd.open_workbook(review_fr_path)


def get_app_topic_version():
    ## Get the app-version-topic dictionary
    app_ver_topic_dict = {}
    fr = open('./platform_data/pred_result.csv', newline='')
    reader = csv.reader(fr, delimiter=',')
    for idx, row in enumerate(reader):
        if idx == 0:
            continue
        labels = row[-1].split(';')
        app_name = row[1]
        app_flag = check_app_name(app_name, all_app_names)
        if app_flag:
            platform = row[0]
            version = row[5]
            if version in ['None', 'Unknown']:
                continue
            if platform in ['gp', 'Specific_apps']:
                platform = 'gp'
            else:
                platform = 'ios'
            if platform not in app_ver_topic_dict:
                app_ver_topic_dict[platform] = {}
            if app_flag not in app_ver_topic_dict[platform]:
                app_ver_topic_dict[platform][app_flag] = {}
            if version not in app_ver_topic_dict[platform][app_flag]:
                app_ver_topic_dict[platform][app_flag][version] = [0 for i in range(len(label_dict))]
            for l in labels:
                if l == 'other':
                    continue
                app_ver_topic_dict[platform][app_flag][version][label_dict[l]] += 1
    return app_ver_topic_dict

def compute_issue_duration(topic_ver_values, threshold=5):
    reduce_vers = []  # record whether the percentage equals zero and shows significant decrease
    topic_ver_values = np.array(topic_ver_values)
    mu = np.mean(topic_ver_values)
    sigma = np.var(topic_ver_values)
    for i, value in enumerate(topic_ver_values):
        if abs((value-mu)/sigma/100) > threshold and value < mu:
            reduce_vers.append(2)
        elif value == float(0.0):
            reduce_vers.append(-1)
        else:
            reduce_vers.append(1)

    durations = []
    if set(reduce_vers).__len__() == 1 and reduce_vers[0] == 2:
        return [len(reduce_vers)]
    if 2 in reduce_vers:
        for i, value in enumerate(reduce_vers):
            if value == 2:
                if -1 not in reduce_vers[:i] and 2 not in reduce_vers[:i]:
                    if i == 0:
                        continue
                    durations.append(i)
                else:
                    for j in range(i-1,-1,-1):
                        if reduce_vers[j] == -1 or reduce_vers[j] == 2:
                            durations.append(i-j)
                            break

        if durations == float(0.0):
            return 0.0
        else:
            return np.mean(durations)
    else:
        return 0.0

def get_fixed_versions(ver_topic_dict, ver_count_dict):
    ## Check whether the ad issue in one version is fixed
    gp_ver_count = 0
    ios_ver_count = 0
    gp_periods = [[] for _ in range(len(label_dict))]
    ios_periods = [[] for _ in range(len(label_dict))]
    for pt, app in ver_topic_dict.items():
        for app, versions in app.items():
            if check_app_name(app, version_apps):
                continue
            sorted_vers = sorted(versions.keys())
            if len(sorted_vers) < 3:  # Remove version num fewer than 3
                print('Platform', pt, ' App ', app, ' with fewer than 3 versions and deleted')
                continue
            if pt == 'gp':
                gp_ver_count += len(sorted_vers)
            else:
                ios_ver_count += len(sorted_vers)
            topic_ver_probs = [[0 for _ in range(len(sorted_vers))] for _ in range(len(label_dict))]
            # iter(versions.keys()), key=lambda s: map(int, s.split('.'))
            for idv, version in enumerate(sorted_vers):
                ver_num = ver_count_dict[pt][app][version]
                ver_prob = [t/float(ver_num) for t in versions[version]]
                for idx, prob in enumerate(ver_prob):
                    topic_ver_probs[idx][idv] = prob

            for idj, topic in enumerate(topic_ver_probs):
                duration = compute_issue_duration(topic)
                if duration != float(0.0):
                    if pt == 'gp':
                        gp_periods[idj].append(duration)
                    else:
                        ios_periods[idj].append(duration)

                    # topic_periods[idj].append(duration)
    print('Total version num on google ', gp_ver_count, ' ios is ', ios_ver_count)
    gp_avg_durations = [0.0 for i in range(len(label_dict))]
    ios_avg_durations = [0.0 for i in range(len(label_dict))]
    for idx, prob in enumerate(gp_periods):
        gp_avg_durations[idx] = np.mean(prob)
        ios_avg_durations[idx] = np.mean(ios_periods[idx])
    print('Difference between gp and ios durations ', scipy.stats.mannwhitneyu(gp_avg_durations, ios_avg_durations)[1])
    print('Average durations for gp and ios ', np.nanmean(gp_avg_durations), np.nanmean(ios_avg_durations))
    print('Median durations for gp and ios ', np.nanmedian(gp_avg_durations), np.nanmedian(ios_avg_durations))
    return gp_avg_durations, gp_periods

def check_app_name(name, app_names):
    for app_name in app_names:
        if app_name.lower() in name.lower():
            return app_name
    return False

def get_dir_review(dir, app_version_count_dict):
    all_count = 0
    files = os.listdir(dir)
    platform = dir.split("/")[2]
    if platform in ['gp', 'Specific_apps']:
        platform = 'gp'
    else:
        platform = 'ios'
    if platform not in app_version_count_dict:
        app_version_count_dict[platform] = {}
    for idx, f in enumerate(files):
        app_flag = check_app_name(f, all_app_names)
        if app_flag:
            fr = open(os.path.join(dir, f), errors='ignore')
            lines = fr.readlines()
            fr.close()
            all_count += len(lines)
            for line in lines:
                terms = line.split('******')
                d = {}
                d["author"] = terms[1]
                d["review"] = terms[2]
                d["title"] = terms[3]
                d["date"] = terms[4]
                d["rate"] = terms[5]
                d["version"] = terms[6].strip('\n')
                if d["version"] in ['None', 'Unknown']:
                    continue
                if app_flag not in app_version_count_dict[platform]:
                    app_version_count_dict[platform][app_flag] = {}
                if d["version"] not in app_version_count_dict[platform][app_flag]:
                    app_version_count_dict[platform][app_flag][d["version"]] = 0
                app_version_count_dict[platform][app_flag][d["version"]] += 1
    return app_version_count_dict

def get_mongo_reviews(collects, db, app_version_count_dict, PLATFORM="gp"):
    if PLATFORM not in app_version_count_dict:
        app_version_count_dict[PLATFORM] = {}
    for collect in collects:
        if collect == "clean_master":
            continue
        app_flag = check_app_name(collect, all_app_names)
        if app_flag:
            tb_name = app_flag
        else:
            tb_name = collect
        for article in db[collect].find():
            d = {}
            d["title"] = article["title_review"].split("\n")[0]
            d["review"] = article["title_review"].split("\n")[-1]
            d["date"] = article["date"]
            d["rate"] = article["stars"]
            d["version"] = article["version"]
            if d["version"] == "Unknown" or d["version"] == "None":
                continue
            if tb_name not in app_version_count_dict[PLATFORM]:
                app_version_count_dict[PLATFORM][tb_name] = {}
            if d["version"] not in app_version_count_dict[PLATFORM][tb_name]:
                app_version_count_dict[PLATFORM][tb_name][d["version"]] = 0
            app_version_count_dict[PLATFORM][tb_name][d["version"]] += 1
    return app_version_count_dict

if __name__ == "__main__":
    app_version_count_dict = {}
    app_version_count_dict = get_dir_review(args.google_dir, app_version_count_dict)
    app_version_count_dict = get_dir_review(args.ios_dir, app_version_count_dict)

    client = MongoClient('localhost', 27017)
    google_db = client.appannie_google
    ios_db = client.appannie_ios
    gp_collects = google_db.list_collection_names()
    ios_collects = ios_db.list_collection_names()
    app_version_count_dict = get_mongo_reviews(gp_collects, google_db, app_version_count_dict)
    app_version_count_dict = get_mongo_reviews(ios_collects, ios_db, app_version_count_dict, PLATFORM='ios')

    print(len(app_version_count_dict['gp']), app_version_count_dict['gp'].keys())
    print(len(app_version_count_dict['ios']), app_version_count_dict['ios'].keys())
    cross_apps = list(set(list(app_version_count_dict['gp'].keys())) & set(list(app_version_count_dict['ios'].keys())))

    app_ver_topic_dict = get_app_topic_version()
    print(app_ver_topic_dict)
    topic_avg_periods, topic_periods = get_fixed_versions(app_ver_topic_dict, app_version_count_dict)
    print(topic_avg_periods)

    ## Plot the fixing duration
    input_dict = {'label': [], 'duration': []}
    labels = list(label_dict.keys())
    labels[7] = 'non-skip'
    for i, period in enumerate(topic_periods):
        print("Median values are ")
        print(i, np.median(period))
        if i == 2:
            continue
        if period != period: # Remove nan value
            continue
        l_tick = labels[i]
        if i == 13:
            l_tick = 'orientation'
        for p in period:
            input_dict['label'].append(l_tick)
            input_dict['duration'].append(p)
    sns.set()
    ax = sns.boxplot(x='label', y='duration', data=input_dict)
    ax.set(xlabel='Ad Issue Type', ylabel='Duration')
    # ax.set(ylim=(0.8, 3.2))
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    plt.xticks(rotation=35,fontsize=12)
    plt.yticks(fontsize=14)
    plt.show()
