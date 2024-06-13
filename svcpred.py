import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Results:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.importances: list = []
        self.accuracies: list = []
        self.precisions: list = []
        self.recalls: list = []
        self.f1_scores: list = []
        self.conf_matrices: list = []

    def calculate(self, method: str, key: str) -> float:
        method_dict = {"mean": np.mean, "median": np.median, "sum": sum}
        return method_dict[method](getattr(self, key))

    def get_dictionary(self, method: str) -> dict:

        attributes = [
            "accuracies",
            "precisions",
            "recalls",
            "f1_scores",
        ]

        output_dictionary = {}
        for attribute in attributes:
            output_dictionary[attribute] = self.calculate(method, attribute)

        return output_dictionary

TEST_TIMES = 100

model_results = {}
target_col = ["Attrition"]

df = pd.read_csv("train.csv")

drop_datas = ['user_id', 'BusinessTravel', 'Gender', 'Department',
              'EmployeeCount', 'EmployeeNumber', 'StockOptionLevel',
              'TrainingTimesLastYear', 'Education', 'EducationField', 'JobRole',
              'PercentSalaryHike', 'YearsWithCurrManager', 'Over18', 'DistanceFromHome',
              'MaritalStatus', 'OverTime', 'JobInvolvement', 'JobLevel', 'StandardHours']

input_datas = ['Age',' Attrition', 'DailyRate', 'EnvironmentSatisfaction', 'HourlyRate',
             'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PerformanceRating', 'RelationshipSatisfaction',
            'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion']

df = df.loc[:, ~df.columns.isin(drop_datas)]

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)  # set random state for reproducibility

smote.fit(df.loc[:, ~df.columns.isin(['Attrition'])], df['Attrition'])
X_resampled, y_resampled = smote.fit_resample(df.loc[:, ~df.columns.isin(['Attrition'])], df['Attrition'])

df = pd.concat([X_resampled, y_resampled], axis=1)

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

OE_datas = ['EnvironmentSatisfaction', 'JobSatisfaction', 'PerformanceRating',
            'RelationshipSatisfaction', 'WorkLifeBalance']

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([
    ("OE", OrdinalEncoder(), df.loc[:, df.columns.isin(OE_datas)].columns)
])

encoded_datas = ct.fit_transform(df)

le = LabelEncoder()
df.Attrition = le.fit_transform(df.Attrition)

# Get the column names of the encoded data
column_names = ct.get_feature_names_out()

# Concatenate the original data with the encoded data
df = pd.concat([df.loc[:, (~df.columns.isin(OE_datas))],
                      pd.DataFrame(encoded_datas, columns=column_names)], axis=1)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

X = df.drop(columns = target_col)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

def get_best_rf(X_train, y_train):

	rf = RandomForestClassifier(class_weight="balanced")

	grid = {
		"n_estimators": [70, 75, 80, 85, 90],
		"max_features": [2, 3, "sqrt", "log2"],
		"max_depth": [150, 200, 250],
		"min_samples_split": [5, 6, 7, 8],
		"min_samples_leaf": [1, 2, 3],
		'bootstrap': [True],
	}

	rf_grid = GridSearchCV(estimator = rf, param_grid = grid, cv = 8, verbose = 2, n_jobs = -1)

	rf_grid.fit(X_train, y_train)

	print(f"Random Forest Best Params: {rf_grid.best_params_}\n")

	return rf_grid.best_estimator_

from sklearn.svm import SVC

def get_best_svc(X_train, y_train):

	svc = SVC(class_weight="balanced")

	grid = {
		"kernel": ["linear", "rbf"],
		"C": [5, 10, 15, 20],
		"gamma": ["scale", "auto"],
	}

	svc_grid = GridSearchCV(estimator = svc, param_grid = grid, cv = 8, verbose = 2, n_jobs = -1)

	svc_grid.fit(X_train, y_train)

	print(f"SVC Best Params: {svc_grid.best_params_}\n")

	return svc_grid.best_estimator_

from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score, precision_score

def generate_results(results: Results, model: object, X, y, test_times = 10, set_random = 42):

    for _ in range(test_times):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = set_random)
        X_test_scaled = scaler.fit_transform(X_test)
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        try:
            results.importances.append(model.feature_importances_)
        except AttributeError:
            results.importances.append(0)

        results.accuracies.append(accuracy_score(y_test, y_pred))
        results.precisions.append(precision_score(y_test, y_pred))
        results.recalls.append(recall_score(y_test, y_pred))
        results.conf_matrices.append(confusion_matrix(y_test, y_pred))
        results.f1_scores.append(f1_score(y_test, y_pred))

def print_results(results_dictionary: dict, prepend = False):
    if prepend:
        print(prepend)
    for key, value in results_dictionary.items():
        if type(value) == np.float64:
            print(f"{key}: {(value * 100).round(2)}%")
        else:
            print(f"{key}: {value} Type: {type(value)}")
    print()

SVC_Results = Results("SVC")
pred_SVC_result = get_best_svc(X_train_scaled, y_train)
generate_results(SVC_Results, pred_SVC_result, X, y, TEST_TIMES)

SVC_averages: dict = SVC_Results.get_dictionary("mean")
print_results(SVC_averages, "SVC Averages: ")

from joblib import dump
dump(pred_SVC_result, 'SVC_model.joblib')


