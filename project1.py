from string import punctuation, digits
import numpy as np
import random
import csv

# Part I


#pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
#pragma: coderesponse end


#pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):

    '''feature_vector = np.array(feature_vector)
    theta = np.array(theta)'''
    Loss = max(0, 1-label*(np.inner(theta,feature_vector)+theta_0))
    return Loss


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    total_loss = []
    for feature_vector, label in zip(feature_matrix, labels):
        for i in range(len(labels)):
            Loss = hinge_loss_single(feature_vector, label, theta, theta_0)
        total_loss.append(float(Loss))
        sumloss = 0
        for loss in total_loss:
            sumloss = sumloss + loss

    return sumloss / len(total_loss)



#pragma: coderesponse end


#pragma: coderesponse template
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    feature_vector = np.array(feature_vector)
    current_theta = np.array(current_theta)

    x = 0
    float(abs(x)) < 10 ** (-6)
    if label * (np.inner(current_theta, feature_vector) + current_theta_0) <= x:
        current_theta = current_theta + label * feature_vector
        current_theta_0 = current_theta_0 + label
        # print(new_theta,np.inner(current_theta,feature_vector),label*feature_vector)
    else:
        current_theta = current_theta
        current_theta_0 = current_theta_0

    return (current_theta, current_theta_0)


#pragma: coderesponse end


#pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    feature_matrix = np.array(feature_matrix)
    current_theta = np.zeros(shape=(len(feature_matrix[0])), dtype=object)
    # current_theta = np.zeros(shape=(1, feature_matrix.shape[1]+1))
    current_theta_0 = 0

    # for feature_vector, label in zip(feature_matrix, labels):

    for t in range(T):
        # print(t)
        for i in get_order(feature_matrix.shape[0]):
            # print(i)
            # for feature_vector, label in zip(feature_matrix, labels):

            # feature_vector = feature_matrix[i]
            # print(feature_vector)

            (current_theta, current_theta_0) = perceptron_single_step_update(feature_matrix[i], labels[i],
                                                                             current_theta, current_theta_0)

            # current_theta = update[0]
            # current_theta_0 = update[1]

            # print(current_theta)

    return (current_theta, current_theta_0)


#pragma: coderesponse end


#pragma: coderesponse template
def average_perceptron(feature_matrix, labels, T):
    feature_matrix = np.array(feature_matrix)
    current_theta = np.zeros(shape=(len(feature_matrix[0])), dtype=object)
    # current_theta = np.zeros(shape=(1, feature_matrix.shape[1]+1))
    current_theta_0 = 0

    # for feature_vector, label in zip(feature_matrix, labels):
    # total_theta = np.empty(shape=len(feature_matrix[0]))
    total_theta = np.zeros(shape=(len(feature_matrix[0])), dtype=object)
    total_theta_0 = 0

    for t in range(T):
        # print(t)
        for i in get_order(feature_matrix.shape[0]):
            # print(i)
            # for feature_vector, label in zip(feature_matrix, labels):

            # feature_vector = feature_matrix[i]
            # print(feature_vector)

            (current_theta, current_theta_0) = perceptron_single_step_update(feature_matrix[i], labels[i],
                                                                             current_theta, current_theta_0)

            # current_theta = update[0]
            # current_theta_0 = update[1]

            # print(current_theta, current_theta_0)
            # for k in range(T*feature_matrix.shape[0]):
            total_theta = total_theta + current_theta
            total_theta_0 += current_theta_0
        # print(total_theta, total_theta_0)
        '''sumtheta = 0
        sumtheta_0=0
    for theta,theta_0 in zip(total_theta, total_theta_0):
        sumtheta = np.add(total_theta,theta)
        sumtheta_0 = total_theta_0 + theta_0
    print(sumtheta)'''
        # sumtheta = np.sum(total_theta,axis = 0)
        # print(sumtheta)
        # sumtheta_0 = sum(total_theta_0)
        # print(sumtheta_0)
    # return (np.divide(total_theta,T*feature_matrix.shape[0]), total_theta_0/(T*feature_matrix.shape[0]))
    return (total_theta / (T * feature_matrix.shape[0]), total_theta_0 / (T * feature_matrix.shape[0]))


#pragma: coderesponse end


#pragma: coderesponse template
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    feature_vector = np.array(feature_vector)
    current_theta = np.array(current_theta)

    x = 0
    float(abs(x)) < 10 ** (-6)
    # print(label*(np.inner(current_theta,feature_vector)+current_theta_0))
    if label * (np.inner(current_theta, feature_vector) + current_theta_0) <= (1 + x):
        current_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        current_theta_0 += eta * label
        # print(new_theta,np.inner(current_theta,feature_vector),label*feature_vector)
    else:
        current_theta = (1 - eta * L) * current_theta
        current_theta_0 = current_theta_0
    return (current_theta, current_theta_0)
#pragma: coderesponse end


#pragma: coderesponse template
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    feature_matrix = np.array(feature_matrix)
    current_theta = np.zeros(shape=(len(feature_matrix[0])), dtype=object)
    # current_theta = np.zeros(shape=(1, feature_matrix.shape[1]+1))
    current_theta_0 = 0

    # for feature_vector, label in zip(feature_matrix, labels):
    '''etas = np.array([])
    for k in [1,T*feature_matrix.shape[0]]:
        eta = k**(-0.5)
        print(eta)
    etas = '''
    k = 0
    for t in range(T):
        # print(t)
        for i in get_order(feature_matrix.shape[0]):
            # print(i)
            # for feature_vector, label in zip(feature_matrix, labels):

            # feature_vector = feature_matrix[i]
            # print(feature_vector)
            k += 1
            '''for k in [1,T*feature_matrix.shape[0]]:
                eta = k**(-0.5)
                print(eta)'''
            eta = k ** (-0.5)
            # print(k,eta)
            (current_theta, current_theta_0) = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta,
                                                                          current_theta, current_theta_0)
            # current_theta = update[0]
            # current_theta_0 = update[1]

            # print(current_theta)

    return (current_theta, current_theta_0)

#pragma: coderesponse end

# Part II


#pragma: coderesponse template
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    feature_matrix = np.array(feature_matrix)
    x = 0
    labels = []
    float(abs(x)) < 10 ** (-6)
    theta = np.array(theta)

    for feature_vector in feature_matrix:

        linear = np.inner(feature_vector, theta) + theta_0
        # print(linear)
        if linear > x:
            labels.append(1)
        else:
            labels.append(-1)
    return np.array(labels)

#pragma: coderesponse end


#pragma: coderesponse template
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    (train_theta, train_theta_0) = classifier(train_feature_matrix, train_labels, **kwargs)
    # (val_theta, val_theta_0) = classifier(val_feature_matrix, val_labels, **kwargs)
    pred_train_labels = classify(train_feature_matrix, train_theta, train_theta_0)
    pred_val_labels = classify(val_feature_matrix, train_theta, train_theta_0)
    return (accuracy(pred_train_labels, train_labels), accuracy(pred_val_labels, val_labels))

#pragma: coderesponse end


#pragma: coderesponse template
def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
#pragma: coderesponse end


#pragma: coderesponse template
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    stopwords = np.genfromtxt('stopwords.txt', dtype = 'str' )
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    #print(len(dictionary))
    return dictionary
#pragma: coderesponse end


#pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
                #using count
                #feature_matrix[i, dictionary[word]] += 1
    return feature_matrix
#pragma: coderesponse end


#pragma: coderesponse template
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
#pragma: coderesponse end
