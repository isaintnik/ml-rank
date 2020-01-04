import pandas as pd
import numpy as np

from mlrank.datasets.dataset import dataframe_to_series_map, fit_encoder, SeparatedDataset, get_features_except


class InternetDataSet(SeparatedDataset):
    def __init__(self, train_folder: str, test_folder: str):
        super().__init__('internet', train_folder=train_folder, test_folder=test_folder)

        self.target_feature = 'Who_Pays_for_Access_Work'

        self.cat_features = {
            'Actual_Time', 'Community_Building', 'Country', 'Education_Attainment', 'Falsification_of_Information',
            'Household_Income', 'Major_Geographical_Location', 'Major_Occupation', 'Marital_Status',
            'Most_Import_Issue_Facing_the_Internet', 'Opinions_on_Censorship', 'Primary_Computing_Platform', 'Primary_Language',
            'Primary_Place_of_WWW_Access', 'Race', 'Registered_to_Vote', 'Sexual_Preference', 'Web_Ordering', 'Web_Page_Creation',
            'Willingness_to_Pay_Fees', 'Years_on_Internet'
        }

        self.bin_features = {'Community_Membership_Family',
            'Community_Membership_Hobbies',
            'Community_Membership_None',
            'Community_Membership_Other',
            'Community_Membership_Political',
            'Community_Membership_Professional',
            'Community_Membership_Religious',
            'Community_Membership_Support',
            'Disability_Cognitive',
            'Disability_Hearing',
            'Disability_Motor',
            'Disability_Not_Impaired',
            'Disability_Not_Say',
            'Disability_Vision',
            'Gender',
            'How_You_Heard_About_Survey_Banner',
            'How_You_Heard_About_Survey_Friend',
            'How_You_Heard_About_Survey_Mailing_List',
            'How_You_Heard_About_Survey_Others',
            'How_You_Heard_About_Survey_Printed_Media',
            'How_You_Heard_About_Survey_Remebered',
            'How_You_Heard_About_Survey_Search_Engine',
            'How_You_Heard_About_Survey_Usenet_News',
            'How_You_Heard_About_Survey_WWW_Page',
            'Not_Purchasing_Bad_experience',
            'Not_Purchasing_Bad_press',
            'Not_Purchasing_Cant_find',
            'Not_Purchasing_Company_policy',
            'Not_Purchasing_Easier_locally',
            'Not_Purchasing_Enough_info',
            'Not_Purchasing_Judge_quality',
            'Not_Purchasing_Never_tried',
            'Not_Purchasing_No_credit',
            'Not_Purchasing_Not_applicable',
            'Not_Purchasing_Not_option',
            'Not_Purchasing_Other',
            'Not_Purchasing_Prefer_people',
            'Not_Purchasing_Privacy',
            'Not_Purchasing_Receipt',
            'Not_Purchasing_Security',
            'Not_Purchasing_Too_complicated',
            'Not_Purchasing_Uncomfortable',
            'Not_Purchasing_Unfamiliar_vendor',
            'Who_Pays_for_Access_Dont_Know',
            'Who_Pays_for_Access_Other',
            'Who_Pays_for_Access_Parents',
            'Who_Pays_for_Access_School',
            'Who_Pays_for_Access_Self',
            'Who_Pays_for_Access_Work'
        }

        self.encoders = dict()

    def get_continuous_feature_names(self):
        return ['Age']

    def load_train_from_file(self):
        self.train = pd.read_csv(self.train_folder)
        self.train.drop(labels=['who'], axis=1, inplace=True)
        self.train = self.train[self.train.Age != 'Not_Say']
        self.train['Falsification_of_Information'].fillna('NaN', inplace=True)

    def load_test_from_file(self):
        self.test = pd.read_csv(self.test_folder)
        self.test.drop(labels=['who'], axis=1, inplace=True)
        self.test = self.test[self.test.Age != 'Not_Say']
        self.test['Falsification_of_Information'].fillna('NaN', inplace=True)
        #self.test.Age = self.test.Age.astype(np.float32)

    def process_features(self):
        encoders = dict()
        for feature in self.cat_features.union(self.bin_features):
            encoders[feature] = fit_encoder(pd.concat([self.train[feature], self.test[feature]]))

            self.train[feature] = encoders[feature].transform(self.train[feature])
            self.test[feature] = encoders[feature].transform(self.test[feature])

        self.features_ready = True

    def get_dummies(self, data_chunk: pd.DataFrame) -> dict:
        dummy_features = dict()

        for feature in self.cat_features.difference({self.target_feature}):
            if feature != 'ACTION':
                dummy_features[feature] = pd.get_dummies(data_chunk[feature]).values.T

        dummy_features.update({'Age': data_chunk['Age']})

        for feature in self.bin_features.difference({self.target_feature}):
            dummy_features.update({feature: data_chunk[feature].values})

        return dummy_features

    def cache_features(self):
        self.train_plain = dataframe_to_series_map(get_features_except(self.train, [self.target_feature]))
        self.train_transformed = self.get_dummies(get_features_except(self.train, [self.target_feature]))

        self.test_plain = dataframe_to_series_map(get_features_except(self.test, [self.target_feature]))
        self.test_transformed = self.get_dummies(get_features_except(self.test, [self.target_feature]))

    def get_train_target(self) -> pd.Series:
        return self.train[self.target_feature].values

    def get_test_target(self) -> np.array:
        return self.test[self.target_feature].values

    def get_train_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return self.train_plain
        else:
            return self.train_transformed

    def get_test_features(self, convert_to_linear: bool) -> dict:
        if not self.features_ready:
            raise Exception('call process_features')

        if not convert_to_linear:
            return self.test_plain
        else:
            return self.test_transformed
