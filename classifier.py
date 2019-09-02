### classify the labeled ad reviews
from metric import accuracy,exact_match,f_score_micro,f_score_by_label
import csv
import numpy as np


class Keyword_Match:
    def __init__(self,label_dict, true_labels, ad_review, review_data, keyword_dict):
        self.label_dict = label_dict
        self.true_labels = true_labels
        self.ad_review = ad_review
        self.review_data = review_data
        self.keyword_dict = keyword_dict

    def classify(self):
        ## predict classes of input reviews
        pred_labels = [[] for i in range(len(self.ad_review))]
        for idx, review in enumerate(self.ad_review):
            for k, v in self.keyword_dict.items():
                if k == 'security':
                    context = self.review_data[idx]
                else:
                    context = review
                for j in v:
                    if j in context:
                        if k == 'popup' and j == "the middle of" and "the middle of the screen" in context:
                            continue
                        if k == 'crash' and j == "pause" and "pauses by itself" in context:
                            continue
                        if len(j.split()) == 1 and j in context.split():
                            pred_labels[idx].append(self.label_dict[k])
                            break
                        elif len(j.split()) > 1:
                            pred_labels[idx].append(self.label_dict[k])
                            break
            if pred_labels[idx] and self.label_dict['other'] in pred_labels[idx]:
                pred_labels[idx].remove(self.label_dict['other'])
            if not pred_labels[idx]:
                pred_labels[idx].append(self.label_dict['other'])
        return pred_labels

    def validate(self):
        self.pred_labels = self.classify()
        acc = accuracy(self.true_labels, self.pred_labels)
        ematch = exact_match(self.true_labels, self.pred_labels)
        pre_micro, rec_micro, f_micro = f_score_micro(self.true_labels, self.pred_labels)
        pre_label, rec_label, f_label, prec_result, recall_result = f_score_by_label(self.true_labels, self.pred_labels, len(self.label_dict))
        return acc, ematch, pre_micro, rec_micro, f_micro, pre_label, rec_label, f_label, prec_result, recall_result

    def save_result(self):
        ## Save predicted labels to file
        idx_to_label = {v: k for k, v in self.label_dict.items()}
        true_categories = ['\n'.join([idx_to_label[j] for j in i]) for i in self.true_labels]
        pred_categories = ['\n'.join([idx_to_label[j] for j in i]) for i in self.pred_labels]
        rows = zip(true_categories, pred_categories, self.ad_review, self.review_data)
        fw = open('./ad_reviews/pred_result.csv', 'w', newline='')
        writer = csv.writer(fw, delimiter=',')
        for row in rows:
            writer.writerow(row)
        fw.close()


class Multi_labeling:
    def __init__(self, label_dict, train_labels, train_data, test_labels, test_data):
        self.label_dict = label_dict
        self.train_labels = train_labels
        self.train_data = train_data
        self.test_labels = test_labels
        self.test_data = test_data

    def classify(self):
        from skmultilearn.problem_transform import ClassifierChain
        from sklearn.svm import SVC,LinearSVC
        import sklearn.metrics as metrics

        # =============================
        #      ClassifierChain        #
        # =============================
        from sklearn.multiclass import OneVsRestClassifier
        # from sklearn.multioutput import ClassifierChain
        from sklearn.linear_model import LogisticRegression
        # cc = ClassifierChain(LogisticRegression())
        self.cc = ClassifierChain(LinearSVC())
        self.cc.fit(self.train_data, self.train_labels)
        # y_pred = self.cc.predict(self.test_data)
        # cc_art_f1 = metrics.f1_score(self.test_labels, y_pred, average='micro')



        # # initialize Classifier Chain multi-label classifier
        # # with an SVM classifier
        # # SVM in scikit only supports the X matrix in sparse representation
        # classifier = ClassifierChain(
        #     classifier=SVC(),
        #     require_dense=[False, True]
        # )
        # # train
        # classifier.fit(self.train_data, self.train_labels)
        # # predict
        # predictions = classifier.predict(self.test_data)
        # print(predictions)
        # art_f1 = metrics.f1_score(self.test_labels, predictions, average='macro')
        # return art_f1




        # =============================
        #    KNeighborsClassifier     #
        # =============================
        from sklearn.neighbors import KNeighborsClassifier
        knc = KNeighborsClassifier()

        knc.fit(self.train_data, self.train_labels)
        # Y_pred = knc.predict(self.test_data)
        # knc_art_f1 = metrics.f1_score(self.test_labels, Y_pred, average='micro')




        # =============================
        #           SGDClassifier     #
        # =============================
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import SGDClassifier
        sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=0, max_iter=6, tol=None)
        clf = OneVsRestClassifier(sgd)
        clf.fit(self.train_data, self.train_labels)
        # y_pred = clf.predict(self.test_data)
        # sgd_art_f1 = metrics.f1_score(self.test_labels, y_pred, average='micro')
        # return cc_art_f1, knc_art_f1, sgd_art_f1

    def pred_all_other(self, input_data):
        y_pred = self.cc.predict(input_data)
        return y_pred