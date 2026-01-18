import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# parameters
random_seed = 1
test_size = 0.2
n_estimators = 260
max_depth = 10
output_file = f'model_max_depth={max_depth}_n_estimators={n_estimators}.bin'
path_save = '/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-2/'

# load data and preparation
df = pd.read_csv(r'/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-2/01 Data/Exam_Score_Prediction.csv')
# rename target
df = df.rename(columns={'exam_score': 'y'})

feature_selection = ['gender', 'study_hours', 'class_attendance',
       'sleep_hours', 'sleep_quality', 'study_method',
       'facility_rating']

df = df[feature_selection + ['y']]

# train final model
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.y.values
df_full_train.drop(['y'], axis=1, inplace=True)

y_test = df_test.y.values
df_test.drop(['y'], axis=1, inplace=True)

categorical_columns = df_full_train[feature_selection].select_dtypes(include=['object']).columns.tolist()
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough')

X_train = ct.fit_transform(df_full_train[feature_selection])
X_test = ct.transform(df_test[feature_selection])
    
max_depth = 10
n_estimators = 260
rf = RandomForestRegressor(n_estimators=260, max_depth=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_full_train)
y_pred = rf.predict(X_test)
print("Test RMSE:", round((mean_squared_error(y_test, y_pred))**0.5, 3))

print(f'Saving the model to {output_file}')
with open(path_save + output_file, 'wb') as f_out:
    pickle.dump((rf, ct), f_out)
print(f'Model saved')