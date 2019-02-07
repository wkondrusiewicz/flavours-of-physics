import classifier as clf
import data
import parameters
import functions as func

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

n = parameters.n
train_params = parameters.train_params_xgb


train, check_agreement, check_correlation, test = data.load_new_features(
    no_test=False)
var_kin, var_geo = data.variables_list()
skf = StratifiedKFold(n_splits=n, shuffle=True)


params = {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 4,
          'subsample': 0.5, 'n_jobs': 4, 'min_child_weight': 15}

train_params = {'early_stopping_rounds': 10, 'verbose': 0}
xgb = XGBClassifier(**params)
xgb.set_params(**train_params)


xgb_kin = clf.Classifier(model=xgb, cv=skf, variables=var_kin, model_name='XGBoost',
                         var_name='kinetic', fig_name='xgb', train_params=train_params)
xgb_kin.fit(train)
xgb_kin.check_ks_and_cvm(
    train, check_agreement=check_agreement, check_correlation=check_correlation)
xgb_kin.predict(data=test)
params = {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 4,
          'subsample': 0.5, 'n_jobs': 4, 'min_child_weight': 15}

train_params = {'early_stopping_rounds': 10, 'verbose': 0}
xgb = XGBClassifier(**params)
xgb.set_params(**train_params)

xgb_geo = clf.Classifier(model=xgb, cv=skf, variables=var_geo, model_name='XGBoost',
                         var_name='geometric', fig_name='xgb', train_params=train_params)
xgb_geo.fit(train)
xgb_geo.check_ks_and_cvm(
    train, check_agreement=check_agreement, check_correlation=check_correlation)
xgb_geo.predict(data=test)


func.save_combined_output('submit.csv', test, 0.4, 0.6, xgb_kin, xgb_geo)
func.check_combined_ks_and_cvm(
    0.4, 0.6, xgb_kin, xgb_geo, check_agreement=check_agreement, check_correlation=check_correlation)

"""
#example of neural network:

import keras
from keras.layers import Input, Dense
from keras.models import Model


inputs = Input(shape=(len(var_kin),))
x = Dense(8, activation='relu')(inputs)
x = Dense(4, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
mod = Model(inputs=inputs, outputs=predictions)
mod.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


nn_kin = clf.Classifier(model=mod, cv=skf, variables=var_kin, model_name='Dense Neural Network',
                        var_name='kinetic', fig_name='nn', train_params=train_params, nn=True)
nn_kin.fit(data=train)
nn_kin.check_ks_and_cvm(
    data=train, check_agreement=check_agreement, check_correlation=check_correlation)
nn_kin.plot_loss()
"""
