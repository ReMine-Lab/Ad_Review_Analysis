### load dataset for classification
import xlrd, csv
import re
from nltk.stem.wordnet import WordNetLemmatizer
from process.process_sql import check_ad
from sklearn import cross_validation
from classifier import Keyword_Match, Multi_labeling
import random
import numpy as np
from metric import accuracy,f_score_by_label,f_score_micro,exact_match
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import scipy
from predict_all import PredictAll

# fr_path = './ad_reviews/labeled_ad_reviews.xlsx'
fr_path = './ad_reviews/labeled_ad_reviews_2.xlsx'
workbook = xlrd.open_workbook(fr_path)
special_words = ['process', 'access', 'less', 'as']
combine_classes = {'interrupt': 'popup', 'ad block': 'too many', 'video ad': 'auto-play', 'paid': 'other', 'intrusive': 'annoying', 'page location':'position','video ads': 'auto-play'}
count_other = 0
sample_num = 280
run_predicted = False
multi_labeling = False
predict_other_review = True

def lemma_word(words, digit=False):
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

def get_keywords(remove_annoying=False):
    keyword_sheet = workbook.sheet_by_index(1)
    class_dict = {}
    keywords = []
    for i, row in enumerate(keyword_sheet.get_rows()):
        if i == 0: # (Optionally) skip headers
            continue
        keyword = keyword_sheet.cell_value(i,0)
        if remove_annoying and keyword in ["annoying", "intrusive"]:
            continue
        class_dict[keyword] = list(set([lemma_word(w) for w in keyword_sheet.cell_value(i,1).split(',')]))
        keywords.append(keyword)

    for k in keywords:
        if k in combine_classes and combine_classes[k] in class_dict:
            class_dict[combine_classes[k]] += class_dict.pop(k)
    return class_dict

def get_review_from_sheet(sheet, label_dict, true_labels, ad_review, review_data, single_label=False, remove_annoying=False):
    global count_other
    for i, row in enumerate(sheet.get_rows()):
        if i == 0:
            continue
        label = sheet.cell_value(i,1).split('\n')
        ## Keep reviews with single label
        if single_label and len(label) > 1:
            continue
        ## Count the number of reviews with "other" label
        if 'other' in label:
            count_other += 1
        ## Remove "annoying" label
        if remove_annoying:
            if 'annoying' in label:
                label = list(filter(lambda a: a != 'annoying', label))
            if 'intrusive' in label:
                label.remove('intrusive')
            if len(label) < 1:
                continue
        for idx, l in enumerate(label):
            if l in combine_classes.keys():
                label[idx] = combine_classes[l]
                if combine_classes[l] not in label_dict:
                    label_dict[combine_classes[l]] = len(label_dict)
            else:
                if l not in label_dict:
                    label_dict[l] = len(label_dict)
        true_labels.append([label_dict[l] for l in label])
        text = sheet.cell_value(i,3)+sheet.cell_value(i,4)
        if text[-1] not in ['_', '.']:
            text +='.'
        ad_sent, sentence = process_raw_text(text)
        ad_review.append(ad_sent)
        review_data.append(sentence)
    return label_dict, true_labels, ad_review, review_data

def process_raw_text(text):
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
        sentences[i] = lemma_word(sent, digit=True)
    for i, sent in enumerate(ad_sents):
        ad_sents[i] = lemma_word(sent, digit=True)
    return ' . '.join(ad_sents), ' . '.join(sentences)


def get_training_test():
    ios_sheet = workbook.sheet_by_index(2)
    google_sheet = workbook.sheet_by_index(3)
    label_dict = {}
    true_labels = []
    ad_review = []
    review_data = []

    label_dict, true_labels, ad_review, review_data = get_review_from_sheet(ios_sheet, label_dict, true_labels, ad_review, review_data, single_label=False, remove_annoying=True)
    label_dict, true_labels, ad_review, review_data = get_review_from_sheet(google_sheet, label_dict, true_labels, ad_review, review_data, single_label=False, remove_annoying=True)

    # data_train, data_test, train_labels, test_labels = cross_validation.train_test_split(review_data, true_labels, test_size=0.2, random_state=0) #, stratify=Target
    # return data_train, data_test, train_labels, test_labels
    return label_dict, true_labels, ad_review, review_data

def get_samples(true_labels, ad_review, review_data, sample_num):
    label_samples = random.sample(list(enumerate(true_labels)), sample_num)
    ad_samples = []
    true_label_samples = []
    review_samples = []
    for idx, label in label_samples:
        true_label_samples.append(label)
        ad_samples.append(ad_review[idx])
        review_samples.append(review_data[idx])
    return true_label_samples, ad_samples, review_samples

def split_training_test(true_labels, ad_review, label_dict):
    vectorizer = TfidfVectorizer(use_idf=False, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(ad_review)
    Y = np.zeros((len(ad_review), len(label_dict)))
    for i, label in enumerate(true_labels):
        for j in label:
            Y[i][j] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    label_dict, true_labels, ad_review, review_data = get_training_test()
    keyword_dict = get_keywords(remove_annoying=True)
    # print(keyword_dict)
    if not predict_other_review:
        if multi_labeling:
            train_data, test_data, train_labels, test_labels = split_training_test(true_labels, ad_review, label_dict)
            ml = Multi_labeling(label_dict, train_labels, train_data, test_labels, test_data)
            cc_art_f1, knc_art_f1, sgd_art_f1 = ml.classify()
            print("F1_scores are ", cc_art_f1, knc_art_f1, sgd_art_f1)


        else:
            if run_predicted:
                print(label_dict)
                print("No. of test reviews is ", len(ad_review))
                pred_sheet = workbook.sheet_by_index(4)
                sample_true_labels = []
                sample_pred_labels = []
                for i, row in enumerate(pred_sheet.get_rows()):
                    if i == 0:  # (Optionally) skip headers
                        print(row)
                        continue
                    sample_true_labels.append([label_dict[i.strip()] for i in pred_sheet.cell_value(i, 0).split('\n')])
                    sample_pred_labels.append([label_dict[i.strip()] for i in pred_sheet.cell_value(i, 1).split('\n')])
                print("Accuracy is ", accuracy(sample_true_labels, sample_pred_labels))
                print("Ematch is ", exact_match(sample_true_labels, sample_pred_labels))
                print("F_micro is ", f_score_micro(sample_true_labels, sample_pred_labels))
                pre_label, rec_label, f_label, prec_result, recall_result = f_score_by_label(sample_true_labels, sample_pred_labels, len(label_dict))
                print("F_label is ", pre_label, rec_label, f_label)
                print("Precision by label result:")
                for i in prec_result:
                    print(i)
                print("Recall by label result:")
                for i in recall_result:
                    print(i)
            else:
                print(label_dict)
                print("No. of test reviews is ", len(ad_review))
                print("No. of reviews labeled with other is ", count_other)

                Flag = True
                Result_count = 0
                while Flag:
                    true_label_samples, ad_samples, review_samples = get_samples(true_labels, ad_review, review_data, sample_num)
                    km = Keyword_Match(label_dict, true_label_samples, ad_samples, review_samples, keyword_dict)
                    acc, ematch, pre_micro, rec_micro, f_micro, pre_label, rec_label, f_label, prec_result, recall_result = km.validate()
                    if len([i for i in prec_result if i<=0.55 or np.isnan(i)]) < 2 and len([j for j in recall_result if j<=0.55 or np.isnan(j)]) < 2:
                        Result_count += 1
                        if Result_count > 0:
                            km.save_result()
                            Flag = False
                        print(acc, ematch, pre_micro, rec_micro, f_micro, pre_label, rec_label, f_label)
                        print("Precision by label result:")
                        for i in prec_result:
                            print(i)
                        print("Recall by label result:")
                        for i in recall_result:
                            print(i)

    else:
        # Predict all the other reviews
        pa = PredictAll(label_dict, keyword_dict)
        all_ad_review, all_review_data, versions, ratings, dates, platforms, app_names = pa.classify()


        # Use KM_classification to classify 10% ad reviews
        km_pred_all_labels = pa.km_classification()
        print("KM prediction result is: ")
        pred_label_details = pa.output_prediction(km_pred_all_labels)
        pred_labels = [';'.join(l) for l in pred_label_details]
        combine_list = list(map(list, zip(platforms,app_names,ratings,all_ad_review,all_review_data,versions,dates,pred_labels)))


        fw = open('./platform_data/pred_result.csv', 'w', newline='')
        writer = csv.writer(fw, delimiter=',')
        writer.writerow(['Platform', 'App', 'Rating', 'Ad Review', 'Review', 'Version', 'Date', 'Label'])
        for i in combine_list:
            writer.writerow(i)
        fw.close()


        # Use ML_classification to classify all the reviews
        # part_train_adreview = all_ad_review[:int(0.1*len(all_ad_review))]
        # part_train_label = km_pred_all_labels[:int(0.1*len(km_pred_all_labels))]
        # print("No. of km training data is ", len(part_train_adreview), len(part_train_label))
        #
        # ## Use multi labeling to predict all the other reviews
        # vectorizer = TfidfVectorizer(use_idf=False, stop_words=stopwords.words('english'))
        # all_X_vec = vectorizer.fit(all_ad_review)
        # X = all_X_vec.transform(part_train_adreview+ad_review)  # ad_review
        # Y = np.zeros((len(part_train_adreview+ad_review), len(label_dict))).astype(int)       # ad_review
        # for i, label in enumerate(part_train_label+true_labels):   # true_labels
        #     for j in label:
        #         Y[i][j] = 1
        #
        # # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=0)
        #
        # ml = Multi_labeling(label_dict, Y, X, [], [])  # label_dict, y_train, X_train, y_test, X_test
        # ml.classify()
        # # cc_art_f1, knc_art_f1, sgd_art_f1 = ml.classify()
        # # print("F1_scores are ", cc_art_f1, knc_art_f1, sgd_art_f1)
        # all_X = all_X_vec.transform(all_ad_review)
        # y_pred = ml.pred_all_other(all_X)
        # pa.output_prediction(y_pred)
