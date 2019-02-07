import data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import evaluation
import copy
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


train, check_agreement, check_correlation, test = None, None, None, None


class Classifier:

    def __init__(self, model, variables, cv, model_name, var_name, fig_name, train_params={}, nn=False):
        self.model = model
        self.cv = cv  # cross validation object
        self.variables = variables  # variables list
        self.scores_val = []
        self.scores_train = []
        self.trained = None  # results for trained model
        self.ks = None
        self.cvm = None
        self.model_name = model_name  # for plot title
        self.var_name = var_name  # for plot title
        self.trained_model = None  # trained model
        self.fig_var = var_name[:3]  # for plot filename
        self.fig_name = fig_name  # for plot filename
        self.nn = nn  # has to be set to 1 for Neural Network
        self.train_params = train_params  # training parameters
        self.val_history_nn = []
        self.train_history_nn = []
        self.scaler = None
        self.predicted = None  # predicted output
        self.predicted_cv = []  # if one wants to have predictions during cv
        self.agreement_probs = None  # probablities for agreement test
        self.correlation_probs = None  # probablities for correlation test

    def create_model(self):
        return copy.deepcopy(self.model)

    def fit(self, data, data_to_predict=test, pred_cv=False):
        # pred_cv is used for predicing on data_to_predict using cross validation, it's easier to include it in the fit function

        print('Fitting ' + self.model_name + ' model with ' + self.var_name
              + ' variables using ' + str(self.cv.n_splits) + '-fold Cross Validation\n')
        X = data[self.variables].values
        y = data['signal'].values

        trained = np.zeros(len(y))
        for i, (train_ind, test_ind) in enumerate(self.cv.split(X, y)):

            mod = self.create_model()
            scaler = StandardScaler()
            X[train_ind] = scaler.fit_transform(X[train_ind])
            X[test_ind] = scaler.transform(X[test_ind])

            if (self.train_params is not {}) and self.fig_name == 'xgb':
                self.train_params['eval_set'] = [
                    (X[test_ind], y[test_ind])]  # for xgb models
            if (self.train_params is not {}) and self.fig_name == 'nn':
                self.train_params['validation_data'] = (
                    X[test_ind], y[test_ind])  # for nn models

            hist = mod.fit(X[train_ind], y[train_ind], **self.train_params)

            if pred_cv == True:
                X_pred = scaler.transform(data_to_predict[self.variables])

            if self.nn == True:
                y_pred_val = mod.predict(X[test_ind])
                y_pred_train = mod.predict(X[train_ind])
                trained[test_ind] = y_pred_val.reshape((test_ind.shape[0],))
                self.val_history_nn.append(hist.history['val_loss'])
                self.train_history_nn.append(hist.history['loss'])
                if pred_cv == True:
                    self.predicted_cv.append(mod.predict(X_pred))

            else:
                y_pred_val = mod.predict_proba(X[test_ind])[:, 1]
                y_pred_train = mod.predict_proba(X[train_ind])[:, 1]
                trained[test_ind] = y_pred_val
                if pred_cv == True:
                    self.predicted_cv.append(mod.predict_proba(X_pred)[:, 1])

            result_val = evaluation.roc_auc_truncated(y[test_ind], y_pred_val)
            result_train = evaluation.roc_auc_truncated(
                y[train_ind], y_pred_train)

            self.scores_val.append(result_val)
            self.scores_train.append(result_train)

            print('Iteration {} gave ROC AUC score of {} for validation set and {} for training set \n'.format(
                i + 1, np.round(result_val, 4), np.round(result_train, 4)))
        print('Mean ROC AUC score for {}-fold CV is:\n{} for validation set \n{} for training set\n'.format(self.cv.n_splits,
                                                                                                            np.round(np.mean(self.scores_val), 4), np.round(np.mean(self.scores_train), 4)))

        self.trained = trained

    def predict(self, data):
        print('Predicting for ' + self.model_name
              + ' model with ' + self.var_name + ' variables\n')

        X = data[self.variables].values

        if self.nn == True:
            X = self.scaler.transform(X)
            self.predicted = self.trained_model.predict(X)
        else:
            self.predicted = self.trained_model.predict_proba(X)[:, 1]
        return self.predicted

    def check_ks_and_cvm(self, data, check_agreement, check_correlation):
        print('Checking KS and CVM for ' + self.model_name +
              ' model with ' + self.var_name + ' variables\n')

        mod = self.create_model()

        X = data[self.variables].values
        y = data['signal'].values

        if self.nn == True:
            train_X, val_X, train_y, val_y = train_test_split(
                X, y, test_size=0.2)
            self.train_params['validation_data'] = (val_X, val_y)
            scaler = StandardScaler()
            train_X = scaler.fit_transform(train_X)
            val_X = scaler.transform(val_X)
            ch_agr = scaler.transform(check_agreement[self.variables])
            ch_cor = scaler.transform(check_correlation[self.variables])
            self.scaler = scaler

        mod.fit(X, y, **self.train_params)

        if self.nn == True:
            agreement_probs = mod.predict(ch_agr)
            correlation_probs = mod.predict(ch_cor).reshape((ch_cor.shape[0],))
        else:
            agreement_probs = mod.predict_proba(
                check_agreement[self.variables].values)[:, 1]
            correlation_probs = mod.predict_proba(
                check_correlation[self.variables].values)[:, 1]

        ks = evaluation.compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
        cvm = evaluation.compute_cvm(
            correlation_probs, check_correlation['mass'])
        print('KS metric = {}. Is it smaller than 0.09? {}'.format(ks, ks < 0.09))
        print('CVM metric = {}. Is it smaller than 0.002? {}\n'.format(
            cvm, cvm < 0.002))
        self.ks = ks
        self.cvm = cvm
        self.trained_model = mod
        self.agreement_probs = agreement_probs
        self.correlation_probs = correlation_probs

    def make_boxplot(self, flag=False):
        fig = plt.figure(1, figsize=(6, 6))
        df = pd.DataFrame()
        df['Validation'] = self.scores_val
        df['Training'] = self.scores_train
        bp = sns.boxplot(data=df, width=0.5, palette='colorblind')
        plt.title('Box plot for ' + self.model_name
                  + ' with ' + self.var_name + ' variables')
        plt.ylabel('Weighted area under ROC curve')
        if flag == True:
            plt.savefig('boxplot_' + self.fig_name
                        + '_' + self.fig_var + '.pdf')
        plt.show()

    def make_pareto_diagram(self, data, flag=False):
        assert self.nn == 0, "Neural Network does not support feature importances"
        df = pd.DataFrame(self.trained_model.feature_importances_ * 100,
                          index=data[self.variables].columns,
                          columns=['importance']).sort_values('importance', ascending=False)
        df['cum_percentage'] = df['importance'].cumsum() / \
            df['importance'].sum() * 100

        df = df[df['importance'] > 1]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(df.index, df["importance"], color="C0")
        ax2 = ax1.twinx()
        ax2.plot(df.index, df["cum_percentage"], color="C1", marker="D", ms=3)
        ax1.yaxis.set_major_formatter(PercentFormatter())
        ax2.yaxis.set_major_formatter(PercentFormatter())
        ax1.set_ylabel('Percentage of feature importances')
        ax2.set_ylabel('Cumulative percentage of feature importances')

        ax1.tick_params(axis="y", colors="C0")
        ax2.tick_params(axis="y", colors="C1")
        plt.title('Feature importances for ' + self.model_name
                  + ' model with ' + self.var_name + ' variables')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.tight_layout()
        if flag == True:
            plt.savefig('feature_' + self.fig_name
                        + '_' + self.fig_var + '.pdf')
        plt.show()

    def plot_learning_curve(self, data, flag=False):
        assert self.nn == 0, "Neural Network does not support feature importances"
        X = data[self.variables].values
        y = data['signal'].values
        train_sizes, train_scores, test_scores = learning_curve(self.trained_model, X, y, n_jobs=1, cv=self.cv,
                                                                train_sizes=np.linspace(.1, 1.0, 5), verbose=1)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title('Learning curve for ' + self.model_name
                  + ' with ' + self.var_name + ' variables')

        plt.xlabel("Training examples")
        plt.ylabel("Accuracy")
        # plt.gca().invert_yaxis()

        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-',
                 color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-',
                 color="b", label="Cross-validation score")
        plt.legend(loc="best")
        plt.ylim(-.1, 1.1)
        if flag == True:
            plt.savefig('learn_' + self.fig_name + '_' + self.fig_var + '.pdf')
        plt.show()

    def plot_loss(self, flag=False):
        assert self.nn == True, "Only Neural Network support plotting loss function"
        assert len(
            self.val_history_nn) != 0, "Fristly fit model to data to plot loss function"
        s1 = np.std(np.array(self.val_history_nn), axis=0)
        m1 = np.mean(np.array(self.val_history_nn), axis=0)
        s2 = np.std(np.array(self.train_history_nn), axis=0)
        m2 = np.mean(np.array(self.train_history_nn), axis=0)
        ep = np.arange(1, s1.shape[0] + 1)
        plt.plot(ep, m1, 'r', label='validation')
        plt.fill_between(ep, m1 - s1, m1 + s1, alpha=0.1, color="r")
        plt.plot(ep, m2, 'b', label='training')
        plt.fill_between(ep, m2 - s2, m2 + s2, alpha=0.1, color="b")
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss for ' + self.model_name
                  + ' with ' + self.var_name + ' variables')
        if flag == True:
            plt.savefig('loss_' + self.fig_name + '_' + self.fig_var + '.pdf')
        plt.show()
