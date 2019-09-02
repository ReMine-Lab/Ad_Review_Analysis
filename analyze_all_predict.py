# Analyze the prediction results on all app reviews
import csv, json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from skbio.stats.composition import clr, clr_inv

label_dict = {'content': 0, 'security': 1, 'other': 2, 'crash': 3, 'frequency': 4, 'popup': 5, 'too many': 6, 'non-skippable': 7, 'timing': 8, 'slow': 9, 'size': 10, 'position': 11, 'auto-play': 12, 'landscape': 13, 'notification': 14, 'volume': 15}

label_symbol = {'content': 'CO', 'security': 'SE', 'other': 'O', 'crash': 'CR', 'frequency': 'F', 'popup': 'PP', 'too many': 'M', 'non-skippable': 'NS', 'timing': 'T', 'slow': 'SL', 'size': 'SI', 'position': 'PT', 'auto-play': 'AP', 'landscape': 'L', 'notification': 'NF', 'volume': 'V'}

app_names = ['Camera', 'Duolingo', 'eBay', 'Facebook', 'Instagram', 'Spotify', 'YouTube', 'Trivia', 'Sound']
all_app_names = ['Camera','Duolingo','eBay','Evernote','Facebook','HERE','Instagram','LINE','Messenger','Netflix','Skype','Spotify','Tango','TED','Twitter','Viber','VLC','WeChat','WhatsApp','YouTube','Alipay','Minion','Snapchat','Subway','pinterest','Tripadvisor','Trivia','Tom','Amazon','Shareit','Candy','Sound','Waze']


def check_app(app_name, app_names):
    for app in app_names:
        if app.lower() in app_name:
            return app
    return False

## Define variables
instances = []
label_rating_dict = {'label':[],'rating':[]}
label_rating_list = [[] for _ in range(len(label_dict))]
rating_dict = {}  ## {label: [rating]}
rating_count_dict = {'High': [0 for i in range(len(label_dict))], 'Neutral': [0 for _ in range(len(label_dict))], 'Low': [0 for j in range(len(label_dict))]}
co_occur_label_arr = np.zeros((len(label_dict), len(label_dict)), dtype=int)
platform_count_dict = {} ## {app_name: [gp_count, ios_count]}
platform_topic_dict = {} ## {app_name: [[gp_topic_count],[ios_topic_count]]}
platform_count_list = [[0 for _ in range(len(label_dict))], [0 for _ in range(len(label_dict))]]  ## [[gp_topic_count], [ios_topic_count]]
spotify_ver_gp_topic_dict = {} ## {version: [topic_count]}
spotify_ver_ios_topic_dict = {} ## {version: [topic_count]}
youtube_ver_gp_topic_dict = {} ## {version: [topic_count]}
youtube_ver_ios_topic_dict = {} ## {version: [topic_count]}

fr = open('./platform_data/pred_result.csv', newline='')
reader = csv.reader(fr, delimiter=',')
for idx, row in enumerate(reader):
    if idx == 0:
        continue
    labels = row[-1].split(';')
    app_name = row[1]
    platform = row[0]
    for l in labels:
        # if l not in label_rating_dict:
        #     label_rating_dict[l] = []
        # label_rating_dict[l].append(float(row[2]))
        # if l == 'other':
        #     continue
        if l not in rating_dict:
            rating_dict[l] = []
        rating_dict[l].append(float(row[2]))


        if l == 'non-skippable':
            label_rating_dict['label'].append('non-skip')
        elif l == 'landscape':
            label_rating_dict['label'].append('orientation')
        else:
            label_rating_dict['label'].append(l) #label_symbol[l]
        label_rating_dict['rating'].append(float(row[2]))
        label_rating_list[label_dict[l]].append(float(row[2]))

        # Count reviews for apps of different platforms
        target_flag = check_app(app_name.lower(), all_app_names)
        if target_flag:
            if target_flag not in platform_topic_dict:
                platform_topic_dict[target_flag] = [[0 for i in range(len(label_dict))], [0 for i in range(len(label_dict))]]
            if platform in ['gp', 'Specific_apps']:
                platform_topic_dict[target_flag][0][label_dict[l]] += 1
            else:
                platform_topic_dict[target_flag][1][label_dict[l]] += 1

        # Count reviews for different platforms
        if platform in ['gp', 'Specific_apps']:
            platform_count_list[0][label_dict[l]] += 1
        else:
            platform_count_list[1][label_dict[l]] += 1

        # count reviews for different rating groups
        rate = float(row[2])
        if rate <2.5:
            rating_count_dict['Low'][label_dict[l]] += 1
        elif rate >3.5:
            rating_count_dict['High'][label_dict[l]] += 1
        else:
            rating_count_dict['Neutral'][label_dict[l]] += 1

        # Count topics of Spotify Music from different versions
        if 'spotify' in app_name.lower():
            version = row[5]
            if platform in ['gp', 'Specific_apps']:
                if version not in spotify_ver_gp_topic_dict:
                    spotify_ver_gp_topic_dict[version] = [0 for i in range(len(label_dict))]
                spotify_ver_gp_topic_dict[version][label_dict[l]] += 1
            else:
                if version not in spotify_ver_ios_topic_dict:
                    spotify_ver_ios_topic_dict[version] = [0 for i in range(len(label_dict))]
                spotify_ver_ios_topic_dict[version][label_dict[l]] += 1

        # Count topics of YouTube from different versions
        if 'tube' in app_name.lower():
            version = row[5]
            if platform in ['gp', 'Specific_apps']:
                if version not in youtube_ver_gp_topic_dict:
                    youtube_ver_gp_topic_dict[version] = [0 for i in range(len(label_dict))]
                youtube_ver_gp_topic_dict[version][label_dict[l]] += 1
            else:
                if version not in youtube_ver_ios_topic_dict:
                    youtube_ver_ios_topic_dict[version] = [0 for i in range(len(label_dict))]
                youtube_ver_ios_topic_dict[version][label_dict[l]] += 1

    ## Count co-occurrence of different topics
    if len(labels) > 1:
        for i in range(len(labels)):
            for j in range(i, len(labels)):
                co_occur_label_arr[label_dict[labels[i]]][label_dict[labels[j]]] += 1
                co_occur_label_arr[label_dict[labels[j]]][label_dict[labels[i]]] += 1

    ## Count reviews of different platforms
    if app_name not in platform_count_dict:
        platform_count_dict[app_name] = [0,0]
    if platform in ['gp', 'Specific_apps']:
        platform_count_dict[app_name][0] += 1
    else:
        platform_count_dict[app_name][1] += 1


## Pring rating count dictionary
for k, v in rating_count_dict.items():
    print(k)
    print(v)

print(label_dict.keys())
## Print the cooccurence topic dict
# for i in range(len(label_dict)):
#     print(co_occur_label_arr[i].tolist())

## Print the review counts of apps in different platforms
# for k,v in platform_count_dict.items():
#     print(k)
#     print(v)

## Print the review counts of topics for specific apps in different platforms
for k,v in platform_topic_dict.items():
    print(k)
    print(v)


## Print the review counts of topics for all apps in different platform
topic_gp_list = [[] for i in range(len(label_dict))] ## [[topic_count] of app number length]
topic_ios_list = [[] for i in range(len(label_dict))]
app_count = 0
for k, v in platform_topic_dict.items():
    if sum(v[0]) == float(0.0) or sum(v[1]) == float(0.0):
        continue
    arr1 = np.array(v[0][:2]+v[0][3:])/float(sum(v[0])) # take percentage
    arr2 = np.array(v[1][:2]+v[0][3:])/float(sum(v[1]))

    # arr1 = np.array(v[0][:2]+v[0][3:])  # take number for analysis
    # arr2 = np.array(v[1][:2]+v[1][3:])
    # print(k, scipy.stats.mannwhitneyu(arr1,arr2)[1])
    try:
        # print(k, scipy.stats.chi2_contingency([arr1, arr2])[1]) #np.array([arr2, arr1]).T
        print(k, scipy.stats.f_oneway(arr1, arr2)[1])
    except ValueError:
        continue
    for j in range(len(label_dict)-1):
        topic_gp_list[j].append(arr1[j])
        topic_ios_list[j].append(arr2[j])
# Look at the number distributions for the ad issues across platforms
for i in range(len(label_dict)-1):
    # if i == 2:  ## Remove the 'Other' type
    #     continue
    # print(scipy.stats.f_oneway(topic_ios_list[i], topic_gp_list[i])[1])

    # stat, p, dof, expected = scipy.stats.chi2_contingency([topic_ios_list[i], topic_gp_list[i]]) ##np.array(topic_ios_list[i], topic_gp_list[i]).T
    # prob = 0.95
    # critical = scipy.stats.chi2.ppf(prob, dof)
    # print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    # if abs(stat) >= critical:
    #     print('Dependent (reject H0)')
    # else:
    #     print('Independent (fail to reject H0)')
    # # interpret p-value
    # alpha = 1.0 - prob
    # print('significance=%.3f, p=%.3f' % (alpha, p))
    # if p <= alpha:
    #     print('Dependent (reject H0)')
    # else:
    #     print('Independent (fail to reject H0)')
    print(scipy.stats.mannwhitneyu(topic_gp_list[i], topic_ios_list[i])[1])



## Print the t-test result of topic counts from different platforms
# print(scipy.stats.mannwhitneyu(platform_count_list[0],platform_count_list[1]))

## Print the topic counts of different versions in Spotify
# for k, v in spotify_ver_gp_topic_dict.items():
#     print(k)
#     print(v)

# Print the topic counts of different versions in YouTube
# for k, v in youtube_ver_ios_topic_dict.items():
#     print(k)
#     print(v)



# print(np.median(np.array(rating_dict['security'])))
# print(np.median(np.array(rating_dict['notification'])))
# print(np.mean(np.array(rating_dict['security'])))
# print(np.mean(np.array(rating_dict['notification'])))
## Mann-Whitney test for between-groups comparisons with Bonferroni correction for multpiple comparisons

current = "security"
for j in list(label_dict.keys()):
    if current != j:
        print(current, j, scipy.stats.mannwhitneyu(rating_dict[current], rating_dict[j])[1])
current = "notification"
for j in list(label_dict.keys()):
    if current != j:
        print(current, j, scipy.stats.mannwhitneyu(rating_dict[current], rating_dict[j])[1])

fw = open('./platform_data/rating_dict.csv', 'w')
json.dump(rating_dict, fw)
fw.close()





## Plot rating distribution of different ad issue types
# input_dict = {'label': [], 'rating': []}
# labels = list(label_dict.keys())
# labels[7] = 'non-skip'
# for i, ratings in enumerate(label_rating_list):
#     if i == 2:
#         continue
#     l_tick = labels[i]
#     if i == 13:
#         l_tick = 'orientation'
#     for r in ratings:
#         input_dict['label'].append(l_tick)
#         input_dict['rating'].append(r)
# sns.set()
# ax = sns.boxplot(x='label', y='rating', data=input_dict)
# ax.set(xlabel='Ad Issue Type', ylabel='Rating')
# ax.tick_params(axis='x', colors='black')
# ax.tick_params(axis='y', colors='black')
# plt.xticks(rotation=35,fontsize=12)
# plt.yticks(fontsize=14)
# plt.show()



## Observe the correlation between issues and user ratings
## Use Chi-square test for trend
# all_rating_groups = [rating_count_dict['Low'][2], rating_count_dict['Neutral'][2], rating_count_dict['High'][2]]
# rating_groups = [[0,0,0] for _ in range(len(label_rating_list))] # split reviews into three groups: low, neutral, and high
# o_rating_groups = [0,0,0]
# r = np.array([1,2,3])
# R = np.array([1,4,9])
# for idx, ratings in enumerate(label_rating_list):
#     if idx == 2:
#         continue
#     for r in ratings:
#         if r < 2.9:
#             rating_groups[idx][0] += 1
#         elif r > 3.1:
#             rating_groups[idx][2] += 1
#         else:
#             rating_groups[idx][1] += 1
#
# del rating_groups[2]  ## Remove the 'other' group
# print(scipy.stats.chi2_contingency(np.array(rating_groups)))

## Chi square test for trend
# if idx == len(rating_groups)-1:
#     continue
# l_rating_groups = rating_groups[idx]
# o_rating_groups = rating_groups[idx+1]
#
# l_rating_groups = np.array(l_rating_groups)
# o_rating_groups = np.array(o_rating_groups)
# n = l_rating_groups + o_rating_groups
# T1 = sum(np.multiply(l_rating_groups, r))
# T2 = sum(np.multiply(n, r))
# T3 = sum(np.multiply(n, R))
# C = sum(l_rating_groups)
# D = sum(o_rating_groups)
# N = sum(n)
# V = C * D * (N * T3 - T2*T2) / float((N*N* (N - 1)))
# X2=(T1-(C * T2 / float(N)))*(T1-(C * T2 / float(N)))/ V
# p_value = 1 - scipy.stats.chi2.ppf(X2, 1)
# print("Chi square test results for trend", X2)
# print(p_value)

