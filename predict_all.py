## Predict all the ad reviews based on keyword matching

from classifier import Keyword_Match, Multi_labeling
import json, re
from argparse import Namespace
from process_sql import check_ad
from nltk.stem.wordnet import WordNetLemmatizer
import scipy

args = Namespace(
    mysql_fp='./platform_data/ad_reviews.json',
    gp_mongo_fp='./platform_data/gp_mongo_ad_reviews.json',
    ios_mongo_fp='./platform_data/ios_mongo_ad_reviews.json',
)
special_words = ['process', 'access', 'less', 'as']

class PredictAll():
    def __init__(self, label_dict, keyword_dict):
        self.label_dict = label_dict
        self.keyword_dict = keyword_dict
        self.ad_reviews = []
        self.true_labels = []
        self.review_data = []
        self.versions = []
        self.dates = []
        self.ratings = []
        self.platforms = []
        self.app_names = []

    def lemma_word(self, words, digit=False):
        words = words.replace("'", "")
        word_list = words.split(' ')
        new_word_list = []
        for word in word_list:
            word = WordNetLemmatizer().lemmatize(word, 'v')
            if word not in special_words:
                word = WordNetLemmatizer().lemmatize(word, 'n')
            if digit:
                if word.isdigit():
                    word = '<digit>'
            new_word_list.append(word)
        new_word = ' '.join(new_word_list)
        return new_word

    def read_json(self, fpath):
        fr = open(fpath)
        lines = json.load(fr)
        fr.close()

        for platfor, apps in lines.items():
            for app, reviews in apps.items():
                for review in reviews:
                    text = review['review']+review['title']
                    text = text.lower()
                    text = text.replace('_', '. ')
                    text = re.sub(' +', ' ', text)
                    text = text.replace('&#39;', '')
                    sentences = [i.strip() for i in text.split('.') if i.strip()]
                    ## take the ad-related sentences if sentence length larger than 2
                    if len(sentences) > 3:
                        ad_sents = [i for i in sentences if check_ad(i)]
                    else:
                        ad_sents = sentences

                    for i, sent in enumerate(sentences):
                        sentences[i] = self.lemma_word(sent, digit=True)
                    for i, sent in enumerate(ad_sents):
                        ad_sents[i] = self.lemma_word(sent, digit=True)
                    self.ad_reviews.append(' . '.join(ad_sents))
                    self.review_data.append(' . '.join(sentences))
                    self.versions.append(review['version'])
                    self.ratings.append(review['rate'])
                    self.dates.append(review['date'])
                    self.platforms.append(platfor)
                    self.app_names.append(app)

    def classify(self):
        self.read_json(args.mysql_fp)
        self.read_json(args.gp_mongo_fp)
        self.read_json(args.ios_mongo_fp)

        return self.ad_reviews, self.review_data, self.versions, self.ratings, self.dates, self.platforms, self.app_names

    def km_classification(self):
        km = Keyword_Match(self.label_dict, [], self.ad_reviews, self.review_data, self.keyword_dict)
        pred_labels = km.classify()
        print("No. of ad reviews is ", len(pred_labels))
        return pred_labels

    def output_prediction(self, pred_labels):
        label_arr = [0 for i in range(len(self.label_dict))]
        idx_label_dict = {y:x for x,y in self.label_dict.items()}

        if scipy.sparse.issparse(pred_labels):
            pred_labels = pred_labels.toarray().astype(int)
            print(pred_labels.shape)
            new_pre_labels = [[] for i in range(pred_labels.shape[0])]
            for i in range(len(pred_labels)):
                for j in range(len(pred_labels[i])):
                    if pred_labels[i][j] == 1:
                        new_pre_labels[i].append(j)
        else:
            new_pre_labels = pred_labels

        pred_label_detail = [[] for i in range(len(new_pre_labels))]
        for idx, pred_label in enumerate(new_pre_labels):
            for i in pred_label:
                pred_label_detail[idx].append(idx_label_dict[i])
                label_arr[i] += 1
        print(label_arr)
        print(self.label_dict)
        for idx, i in enumerate(label_arr):
            print(i)

        return pred_label_detail


