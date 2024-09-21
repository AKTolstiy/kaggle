import datetime

import dill

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Добавление федерального субъекта по городу (Для РФ)
def regions(data):
    import pandas as pd

    df = data.copy()
    ru_cities = pd.read_pickle('E:/Code/Skillbox/ru_cities.pkl')

    df['region'] = df.geo_city.map(ru_cities.set_index(keys='City')['Federal subject'])
    df.loc[df.geo_country != 'Russia', 'region'] = df[df.geo_country != 'Russia']['geo_country']
    return df


#функция формирует "дополнительные" фичи
def new_features(data):
    import pandas as pd

    df = data.copy()

    # производные от времени визита:
    # - день месяца, день недели, номер месяца от запуска сервиса
    df['v_date'] = pd.to_datetime(df.visit_date)
    df['day'] = df.v_date.dt.day.astype('int8')
    df['week_day'] = (df.v_date.dt.day_of_week + 1).astype('int8')
    df['n_month'] = ((df.v_date.dt.year - 2021) * 12
                           + df.v_date.dt.month - 5).astype('int8')
    df['v_time'] = df.visit_time.apply(lambda x: x[:2]).astype('int8')

    #по разрешению экрана: ширина, высота, пиксели
    df['w_scr'] = df.device_screen_resolution.apply(
            lambda x: x[:x.find('x')] if x.find('x') > 0 else 0
            ).astype('int')
    df['h_scr'] = df.device_screen_resolution.apply(
            lambda x: x[x.find('x') + 1:] if x.find('x') > 0 else 0
            ).astype('int')
    df['pix'] = (df['h_scr'] * df['w_scr'])

    # Россия / Заграница
    df['ru'] = 0
    df.loc[df.geo_country == 'Russia', 'ru'] = 1
    df['ru'] = df['ru'].astype('int8')

    # Трафик органический / НЕорганический
    df['organic'] = 0
    df.loc[df.utm_medium.isin(
            ['organic', 'referral', '(none)']),
            'organic'] = 1
    df['organic'] = df['organic'].astype('int8')

    # СоцСети / прочие
    df['sm'] = 0
    df.loc[df.utm_source.isin(
            ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs',
             'IZEXUFLARCUMynmHNBGo', 'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm'
            ]), 'sm'] = 1
    df['sm'] = df['sm'].astype('int8')

    return df

# выбросы высота, ширина экрана, пиксели
def outliers(data):
    df = data.copy()

    w_max = 2026
    h_max = 1236
    pix_max = 2396760

    df.loc[df['w_scr'] > w_max, 'w_scr'] = w_max
    df.loc[df['h_scr'] > h_max, 'h_scr'] = h_max
    df.loc[df['pix'] > pix_max, 'pix'] = pix_max
    #print(df.columns)
    return df

def df_filter(df):
    columns_to_drop =  ['session_id',
                        'client_id',
                        'visit_date',
                        'visit_time',
                        'utm_keyword',
                        'device_os',
                        'device_model',
                        'device_screen_resolution',
                        'v_date',
                        'geo_country',
                        'geo_city']
    return df.drop(columns_to_drop, axis=1)

def main():
    print('Prediction Pipeline')

    data_prep = Pipeline(steps=[
        ('regions', FunctionTransformer(regions)),
        ('new_features', FunctionTransformer(new_features)),
        ('df_filter', FunctionTransformer(df_filter)),
        ('outliers', FunctionTransformer(outliers))
        ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
        ])

    models = (
        LogisticRegression(C=10.0,
                           class_weight='balanced',
                           max_iter=200,
                           multi_class='ovr',
                           n_jobs=-1,
                           solver='newton-cholesky',
                           tol=0.001),
        MLPClassifier(activation='logistic',
                      alpha=0.001,
                      early_stopping=True,
                      random_state=12,
                      warm_start=True)
        )
    #df = pd.read_pickle('data\small_sess_hits.pkl')
    df = pd.read_pickle('E:/Code/Skillbox/sess_hits.pkl')

    df_posit = df[df.hit == 1]
    for i in range(11):
        df = pd.concat([df, df_posit])

    df = df.sample(frac=1)
    print(df.shape, df.columns)

    X = df.drop(['hit', 'hit_number'], axis=1)
    y = df['hit']

    best_score = .0
    best_pipe = None

    for model in models:
        print(datetime.datetime.now(), model)
        pipe = Pipeline(steps=[
            ('data_preparation',  data_prep),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    with open('hit_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Hit target prediction model',
                'author': 'AK',
                'version': 1,
                'data': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, file)


if __name__ == '__main__':
    main()
