BREAST_CANCER_PATH = './datasets/breast_cancer.csv'
AMAZON_PATH = './datasets/amazon_train.csv'

ADULT_TRAIN_PATH = './datasets/adult_train.csv'
ADULT_TEST_PATH = './datasets/adult_test.csv'

INTERNET_TRAIN_PATH = './datasets/internet_train.dat'
INTERNET_TEST_PATH = './datasets/internet_test.dat'

ALGO_PARAMS = {
    'dataset': [
        #{'type': 'holdout', 'problem': 'classification', 'name': "breast_cancer", 'data': BreastDataSet(BREAST_CANCER_PATH)},
        #{'type': 'holdout', 'problem': 'classification', 'name': "amazon", 'data': AmazonDataSet(AMAZON_PATH)},
        #{'type': 'train_test', 'problem': 'classification', 'name': "adult", 'data': AdultDataSet(ADULT_TRAIN_PATH, ADULT_TEST_PATH)},
        #{'type': 'train_test', 'problem': 'classification', 'name': "internet", 'data': InternetDataSet(INTERNET_TRAIN_PATH, INTERNET_TEST_PATH)},
    ],

    'decision_function': [
        {
            'preprocess': 'linear',
            'regression': ('lasso', {}),
            'classification': ('logreg', {'multi_class': 'auto', 'solver':'liblinear', 'penalty': 'l1', 'C': 1000}),
        },
        {
            'preprocess': 'linear',
            'regression': ('mlpr', {'hidden_layer_sizes': (5, 5), 'activation': 'tanh'}),
            'classification': ('mlpc', {'hidden_layer_sizes': (5, 5), 'activation': 'tanh'})
        },
        {
            'preprocess': None,
            'regression': ('lgbmr', {
                'boosting_type': 'rf',
                'learning_rate': 1e-2,
                'max_depth': 5,
                'subsample': 0.7,
                'n_estimators': 200,
                'verbose': -1,
                'subsample_freq': 5,
                'num_leaves': 2**5,
                'silent': True
            }),
        'classification': ('lgbmc', {
                'boosting_type': 'rf',
                'learning_rate': 1e-2,
                'max_depth': 5,
                'subsample': 0.7,
                'n_estimators': 200,
                'verbose': -1,
                'subsample_freq': 5,
                'num_leaves': 2 ** 5,
                'silent': True
            })
        }
    ]
}


HYPERPARAMS = {
    'bins': [2, 4, 8],
    'lambda': [.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
}