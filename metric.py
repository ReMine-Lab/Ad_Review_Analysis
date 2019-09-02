## Metrics that validate the multi-label results
import numpy as np

def accuracy(true_labels, pred_labels):
    review_num = len(true_labels)
    count = 0.0
    for i, true_label in enumerate(true_labels):
        count += (set(true_label)&set(pred_labels[i])).__len__()/float(set(true_label+pred_labels[i]).__len__())
    return count/review_num

def exact_match(true_labels, pred_labels):
    review_num = len(true_labels)
    count = 0
    for i, true_label in enumerate(true_labels):
        if true_label == pred_labels[i]:
            count += 1
    return count/float(review_num)

def f_score_micro(true_labels, pred_labels):
    pred_num = sum([len(i) for i in pred_labels])
    true_num = sum([len(i) for i in true_labels])
    pred_count  = 0
    for i, pred_label in enumerate(pred_labels):
        pred_count += (set(pred_label) & set(true_labels[i])).__len__()
    precision_micro = pred_count / float(pred_num)
    recall_micro = pred_count / float(true_num)
    f_micro = 2*precision_micro*recall_micro / (precision_micro+recall_micro)
    return precision_micro, recall_micro, f_micro

def f_score_by_label(true_labels, pred_labels, label_num):
    assert len(true_labels) == len(pred_labels)
    inter_matrix = np.zeros(label_num, dtype=float)
    prec_matrix = np.zeros(label_num, dtype=float)
    rec_matrix = np.zeros(label_num, dtype=float)
    for i in range(len(true_labels)):
        inter_list = list(set(true_labels[i]) & set(pred_labels[i]))
        for j in inter_list:
            inter_matrix[j] += 1
        for k in true_labels[i]:
            rec_matrix[k] += 1

    for i in range(len(pred_labels)):
        for k in pred_labels[i]:
            prec_matrix[k] += 1


    # for i in range(len(true_labels)):
    #     for j in range(len(pred_labels[i])):
    #         if pred_labels[i][j] in true_labels[i]:
    #             conf_matrix[pred_labels[i][j], pred_labels[i][j]] += 1
    #
    # precisions = []
    # recalls = []
    # for j in range(label_num):
    #     if np.sum(conf_matrix[:,j]) != 0:
    #         precisions.append(conf_matrix[j,j]/np.sum(conf_matrix[:,j]))
    #     if np.sum(conf_matrix[j]) != 0:
    #         recalls.append(conf_matrix[j,j]/np.sum(conf_matrix[j]))
    # precision_by_label = float(np.mean(precisions))
    # recall_by_label = float(np.mean(recalls))

    precision_by_label = np.nanmean(np.divide(inter_matrix, prec_matrix))
    recall_by_label = np.nanmean(np.divide(inter_matrix, rec_matrix))
    f_by_label = 2*precision_by_label*recall_by_label / float(precision_by_label+recall_by_label)
    prec_result = np.divide(inter_matrix, prec_matrix)
    recall_result = np.divide(inter_matrix, rec_matrix)
    return precision_by_label, recall_by_label, f_by_label, prec_result, recall_result




