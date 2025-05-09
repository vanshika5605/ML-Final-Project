from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_dataset():
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_y = digits[1]
    N = len(digits_dataset_X)
    return digits_dataset_X, digits_dataset_y

def load_from_csv(filename, target_column):
    df = pd.read_csv(filename)

    X = df.drop(columns=target_column).values
    y = df[target_column].values
    return X, y

def load_student_data_from_csv():
    df = pd.read_csv("../data/student_performance_dataset.csv")

    numerical_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce bad entries to NaN

    df.dropna(inplace=True)

    categorical_cols = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 
                    'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    X = df.drop(columns=['GradeClass', 'StudentID']).values  # Don't include target or ID
    y = df['GradeClass'].values
    return X, y