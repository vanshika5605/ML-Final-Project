import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_dataset import load_dataset, load_from_csv
from neural_network import evaluate_model, incremental_learning_curve, epoch_based_learning_curve
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def run_NN(X, y):
    input_size = X.shape[1]
    output_size = len(np.unique(y))
    print(input_size)
    print(output_size)
    architectures = [
        [input_size, 32, output_size],
        [input_size, 32, 16, output_size],
        [input_size, 100, 50, output_size],
        #[input_size, 32, 16, 8, output_size],
    ]

    lambdas = [0.0, 0.01, 0.1, 0.25]

    best_result = {"acc": 0, "f1": 0, "arch": None, "lam": None}

    for arch in architectures:
        for lam in lambdas:
            acc, f1 = evaluate_model(X, y, arch, lam=lam, alpha=0.5)
            print(f"Architecture: {arch}, Lambda: {lam:.2f} -> Acc: {acc:.4f}, F1: {f1:.4f}")

            if f1 > best_result["f1"]:
                best_result = {"acc": acc, "f1": f1, "arch": arch, "lam": lam}

    print("Best configuration:")
    print(f"Architecture: {best_result['arch']}, Lambda: {best_result['lam']}")
    print(f"Accuracy: {best_result['acc']:.4f}, F1 Score: {best_result['f1']:.4f}")
    return best_result['arch'], best_result['lam']

def transform_data(X_raw, y_raw, target_column, csv_path: str):
    # Re-read the DataFrame just to get column info
    df = pd.read_csv(csv_path)
    feature_cols = df.drop(columns=target_column).columns.tolist()
    df_features = df[feature_cols]

    # Detect categorical columns by type
    categorical_cols = df_features.select_dtypes(include='object').columns.tolist()
    categorical_indices = [df_features.columns.get_loc(col) for col in categorical_cols]

    # Define the transformation pipeline
    transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices),
        ('num', StandardScaler(), [i for i in range(X_raw.shape[1]) if i not in categorical_indices])
    ])

    X_transformed = transformer.fit_transform(X_raw)

    # Encode y if it's not already numeric
    if y_raw.dtype == object or len(np.unique(y_raw)) > 2:
        label_encoder = LabelEncoder()
        y_transformed = label_encoder.fit_transform(y_raw)
    else:
        y_transformed = y_raw

    return X_transformed, y_transformed



if __name__ == "__main__":
    # X_digit, y_digit = load_dataset()
    # Normalize input
    # X_digit = X_digit / 16.0
    #best_arch, best_lam = run_NN(X_digit, y_digit)
    # best_arch = [64, 32, 10]
    # best_lam = 0.01
    # epoch_based_learning_curve(X_digit, y_digit, best_arch, best_lam, alpha = 0.1)
    
    # X_credit, y_credit = load_from_csv("../data/credit_approval.csv", target_column="label")
    # X_trans, y_trans = transform_data(X_credit, y_credit, "label", csv_path="../data/credit_approval.csv")
    # best_arch, best_lam = run_NN(X_trans, y_trans)
    # epoch_based_learning_curve(X_trans, y_trans, best_arch, best_lam, alpha = 0.1)
    
    # X_parkinsons, y_parkinsons = load_from_csv("../data/parkinsons.csv", target_column="Diagnosis")
    # X_trans, y_trans = transform_data(X_parkinsons, y_parkinsons, "Diagnosis", csv_path="../data/parkinsons.csv")
    # best_arch, best_lam = run_NN(X_trans, y_trans)
    # epoch_based_learning_curve(X_trans, y_trans, best_arch, best_lam, alpha = 0.1)
    
    # X_rice, y_rice = load_from_csv("../data/rice.csv", target_column="label")
    # X_trans, y_trans = transform_data(X_rice, y_rice, "label", csv_path="../data/rice.csv")
    # best_arch, best_lam = run_NN(X_trans, y_trans)
    # epoch_based_learning_curve(X_trans, y_trans, best_arch, best_lam)

    X_student, y_student = load_from_csv("../data/student_performance_dataset.csv", target_column="GradeClass")
    X_trans, y_trans = transform_data(X_student, y_student, "GradeClass", csv_path="../data/student_performance_dataset.csv")
    # best_arch, best_lam = run_NN(X_trans, y_trans)
    best_arch = [31, 32, 16, 5]
    best_lam = 0.01
    epoch_based_learning_curve(X_trans, y_trans, best_arch, best_lam, alpha = 0.1)

