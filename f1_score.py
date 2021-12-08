import numpy as np
def calculate_f1_scores(results_lists):
    conf_matrix = create_conf_matrix(results_lists)

    precision = calculate_precision(conf_matrix)
    recall = calculate_recall(conf_matrix)
    f1_scores = calculate_f1(precision, recall)
            
    return f1_scores

def create_conf_matrix(list_of_result_arrays):
    conf_matrix = np.zeros((27, 27))
    for l in list_of_result_arrays:
        for i in range(27):
            actual = i
            predicted = l[i]
            if actual == predicted:
                conf_matrix[i,i] += 1
            else:
                conf_matrix[predicted, actual] += 1
    return conf_matrix

def calculate_precision(matrix):
    precision = np.zeros(27, dtype = float)
    i = 0

    for row in matrix:
        true_pos = float(row[i])
        total = float(np.sum(row))
        if total == 0:
            precision[i] = 0 
        else:
            precision[i] = true_pos/total
        i += 1
    #print(precision)
    return precision

def calculate_recall(matrix):
    recall = np.zeros(27, dtype = float)
    i = 0

    column_sums = matrix.sum(axis=0)
    for row in matrix:
        true_pos = float(row[i])
        total = column_sums[i]
        if total == 0:
            recall[i] = 0 
        else:
            recall[i] = true_pos/total
        i += 1
    #print(recall)
    return recall

def calculate_f1(precision, recall):
    f1_scores = np.zeros(27, dtype = float)
    for i in range(27):
        denom = precision[i] + recall[i]
        if denom != 0:
            f1_scores[i] = (2 * float(precision[i]) * float(recall[i])) / (float(precision[i]) + float(recall[i]))
    return f1_scores

