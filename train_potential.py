import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.linear_model
from scipy import io
import os
from sklearn.metrics import accuracy_score

np.random.seed(2020)

STRUCTURE_DIR = "/Users/th/Downloads/datafiles/re.mat"
ATOM_DIR = "/Users/th/Downloads/datafiles/re_atom.mat"
OUTPUT_DIR = "/Users/th/Downloads/datafiles"


def load_data(data_dir):
    data = io.loadmat(data_dir)
    for data_name in ["X", "y"]:
        print("\nloaded %s data!" % data_name)
        print(data[data_name].shape)
    training_x = data.get('X')
    training_y = data.get('y')
    training_data = np.concatenate((training_y, training_x), axis=1)
    return training_x, training_y, training_data


def plot_y_yhat(y, yhat):
    plt.scatter(y, yhat)
    plt.ylabel('y_hat')
    plt.xlabel('y')
    plt.show()


def cross_validation(data, alpha_range, max_iter, f):
    np.random.shuffle(data)
    training_x = data[:, 1:]
    training_y = data[:, 0]
    num_array = data[:, 1]
    alpha_errors = []
    for alpha in alpha_range:
        print(r"5-fold error with alpha = {}".format(alpha))
        errors_validation = []
        errors_train = []
        for i in range(5):
            all_id = list(range(len(data)))
            validation_id = all_id[i::5]
            train_id=[i for i in all_id if i not in validation_id]
            train_x, train_y = training_x[train_id], training_y[train_id]
            validation_x,  validation_y = training_x[validation_id],  training_y[validation_id]
            num_array_train, num_array_validation = num_array[train_id], num_array[validation_id]
            model = f(alpha=alpha, max_iter=max_iter, fit_intercept=False)
            model.fit(train_x, train_y)
            potential = model.coef_
            predicted_validation = model.predict(validation_x)
            predicted_train = model.predict(train_x)
            error_validation = np.average(np.absolute(validation_y - predicted_validation) / num_array_validation)
            error_train = np.average(np.absolute(train_y - predicted_train) / num_array_train)
            #if i == 0:
                #plot_y_yhat(train_y, predicted_train)
                #plot_y_yhat(validation_y, predicted_validation)
            errors_validation.append(error_validation)
            errors_train.append(error_train)
        print("Mean error train: {} eV/atom\n".format(np.mean(errors_train)))
        print("Mean error validaiton: {} eV/atom\n".format(np.mean(errors_validation)))
        alpha_errors.append(np.mean(errors_validation))
    return alpha_errors


def make_potential(output_dir):
    model = skl.linear_model.Ridge(alpha=0.01, max_iter=1E6, fit_intercept=False)
    model.fit(training_x, training_y)
    potential = model.coef_[0]
    with open(os.path.join(output_dir, "re_potential"), 'w') as f:
        for i in range(len(potential)):
            f.write(str(potential[i]) + "\n")
        f.close()


training_x, training_y, training_data = load_data(STRUCTURE_DIR)
cross_validation(training_data, [0, 0.01, 0.001, 0.0001, 0.00001], 1E5,
                 sklearn.linear_model.Ridge)
