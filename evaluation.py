def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    # INITIALIZING TRUE POSITIVES AND TRUE NEGATIVES TP, TN
    tp = 0
    tn = 0
    for t in range(0, len(data)):
        truth_class = data[t][1]
        classifier_class = classifier.predict(data[t][0])
        if truth_class == 'offensive' and classifier_class == 'offensive':
            tp += 1
        if truth_class == 'nonoffensive' and classifier_class == 'nonoffensive':
            tn += 1

    # CALCULATING ACCURACY
    acc = (tp + tn)/len(data)
    return acc
    ################################################################



def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    # INITIALIZING TRUE POSITIVES, TRUE NEGATIVES, FALSE NEGATIVES AND FALSE POSITIVES TP, TN, FN, FP
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for t in range(0, len(data)):
        truth_class = data[t][1]
        classifier_class = classifier.predict(data[t][0])
        if truth_class == 'offensive' and classifier_class == 'offensive':
            tp += 1
        elif truth_class == 'nonoffensive' and classifier_class == 'nonoffensive':
            tn += 1
        elif truth_class == 'nonoffensive' and classifier_class == 'offensive':
            fp += 1
        else:
            fn += 1

    # CALCULATING PRECISION AND RECALL
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    beta = 1

    # CALCULATING F1 SCORE
    f1 = (beta*beta + 1)*(precision*recall)/(beta*beta*precision + recall)
    return f1
    ################################################################
