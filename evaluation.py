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
    tp = 0
    for t in len(data):
        truth_class = data[t][1]
        classifier_class = classifier.predict(data[t])
        if truth_class == 'offensive' and classifier_class == 'offensive':
            tp += 1

    acc = float(tp)/float(len(data))
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
    tp = 0
    fn = 0
    fp = 0
    for t in len(data):
        truth_class = data[t][1]
        classifier_class = classifier.predict(data[t])
        if truth_class == 'offensive' and classifier_class == 'offensive':
            tp += 1
        if truth_class == 'offensive' and classifier_class == 'nonoffensive':
            fn += 1
        if truth_class == 'nonoffensive' and classifier_class == 'offensive':
            fp += 1

    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    beta = 1

    f1 = (beta*beta + 1)*(prec*rec)/(beta*beta*prec + rec)
    return f1
    ################################################################