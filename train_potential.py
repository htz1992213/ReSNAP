import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.linear_model
from scipy import io
import os
import scipy as sp
from sklearn.metrics import accuracy_score

np.random.seed(2020)

OUTPUT_DIR = "/Users/th/Downloads/datafiles"
MODELS = {
    "SVD": sp.linalg.lstsq,
    "LASSO":
        skl.linear_model.Lasso,
    "RIDGE":
        skl.linear_model.Ridge,
    "ELASTIC":
        skl.linear_model.ElasticNet
}


class PotentialTrainer:

    def __init__(self, data_dir, f):
        self.data_dir = data_dir
        self.f = MODELS.get(f)
        data = io.loadmat(self.data_dir)
        for data_name in ["X", "y"]:
            print("\nloaded %s data!" % data_name)
            print(data[data_name].shape)
        self.training_x = data.get('X')
        self.training_y = data.get('y')
        self.training_data = np.concatenate((self.training_y, self.training_x), axis=1)

    @staticmethod
    def plot_y_yhat(y, yhat):
        plt.scatter(y, yhat)
        plt.ylabel('y_hat')
        plt.xlabel('y')
        plt.show()

    def cross_validation(self, alpha_range, max_iter=1E6, plot_image=False):
        data = self.training_data.copy()
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
                train_id = [i for i in all_id if i not in validation_id]
                train_x, train_y = training_x[train_id], training_y[train_id]
                validation_x,  validation_y = training_x[validation_id],  training_y[validation_id]
                num_array_train, num_array_validation = num_array[train_id], num_array[validation_id]
                model = self.f(alpha=alpha, max_iter=max_iter, tol=1e-4,
                               solver='auto', fit_intercept=False)
                model.fit(train_x, train_y)
                predicted_validation = model.predict(validation_x)
                predicted_train = model.predict(train_x)
                error_validation = np.average(np.absolute(validation_y - predicted_validation) / num_array_validation)
                error_train = np.average(np.absolute(train_y - predicted_train) / num_array_train)
                if i == 0 and plot_image:
                    self.plot_y_yhat(train_y, predicted_train)
                    self.plot_y_yhat(validation_y, predicted_validation)
                errors_validation.append(error_validation)
                errors_train.append(error_train)
            print("Mean error train: {} eV/atom".format(np.mean(errors_train)))
            print("Mean error validaiton: {} eV/atom".format(np.mean(errors_validation)))
            alpha_errors.append(np.mean(errors_validation))
        alpha_errors = np.array(alpha_errors)
        max_e = alpha_errors.max()
        min_e = alpha_errors.min()
        diff = max_e - min_e
        print(alpha_errors)
        self.plot_cross(alpha_range, alpha_errors, min_e - diff, max_e + diff)
        return alpha_errors

    @staticmethod
    def plot_cross(param, error, y_low, y_up):
        x = np.arange(len(error))
        plt.plot(x, error, linewidth=2, color='red', marker='.', markersize=12)
        plt.xticks(x, param)
        plt.ylim(y_low, y_up)
        plt.ylabel('Mean validation error (eV/atom)')
        plt.xlabel('Hyperparameter')
        plt.show()

    def make_potential(self, output_dir, alpha=1.0):
        model = skl.linear_model.Ridge(alpha=alpha, max_iter=1E6, tol=1e-4, fit_intercept=False)
        model.fit(self.training_x, self.training_y)
        potential = model.coef_[0]
        with open(os.path.join(output_dir, "re_potential"), 'w') as f:
            for i in range(len(potential)):
                f.write(str(potential[i]) + "\n")
            f.close()


#pt = PotentialTrainer("/Users/th/Downloads/datafiles/re.mat", "RIDGE")
#pt.cross_validation([0, 0.01, 0.001, 0.0001, 0.00001])
