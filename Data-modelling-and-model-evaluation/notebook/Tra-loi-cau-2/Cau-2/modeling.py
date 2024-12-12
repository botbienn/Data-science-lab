    import joblib
import numpy as np
import pandas as pd
from pandas.core.common import random_state
from pandas.io.common import tarfile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

# from pandas.core.array_algos.replace import compare_or_regex_search
import sklearn.pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""------------------------------------------------------------------------"""


def get_file_list():
    # import required module
    import os

    # assign directory
    directory = "../data/"

    # iterate over files in that directory
    result = [f[:-4] for f in os.listdir(directory)]
    return result


def read_location_data(loc_list):
    result = [pd.read_csv(f"../data/{loc}.csv") for loc in loc_list]
    return result


"""------------------------------------------------------------------------"""


def plot_corr_matrix(corr_mat):
    # get module for plotting corr matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(corr_mat, cmap="coolwarm", linewidths=0.5)
    # un-comment this when use in ipynb
    # plt.show()
    # comment this when use in notebook
    plt.show(block=True)


"""------------------------------------------------------------------------"""


def get_high_curr_attri(corr_mat):
    threshold = 0.1
    temp_df = corr_mat["Humidity"]
    result = [
        temp_df.index[val]
        for val, i in enumerate(temp_df)
        if abs(i) >= threshold and i != 1.0
    ]

    return result


"""------------------------------------------------------------------------"""


"""------------------------------------------------------------------------"""


def calc_loss(pipe, X: pd.DataFrame, y: pd.DataFrame):
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=8412
    # )
    # MSE = mean_squared_error(
    #     y_pred=pipe.predict(X_test), y_true=np.array(y_test.to_list())
    # )
    cv_score = cross_val_score(pipe, X, y, cv=10, scoring="r2")
    cv_score = cv_score.round(4)

    return cv_score


# print(f"initial test MSE: {MSE}\n")
# print("cross validation result:")
# print("r2 scores:\n", ", ".join(map(str, cv_score)))
# print("mean r2_score", np.mean(cv_score))
# print("standard deviation r2_score", np.std(cv_score).round(4))
#
"""------------------------------------------------------------------------"""


def remove_list_analyze(pipe, attris):
    remove_list = []
    for attri in attris:
        print("removed", attri)
        attri_list_k = attris.copy()
        attri_list_k.remove(attri)
        score = calc_loss(pipe, X, y)
        if score >= 0.9:
            remove_list.append(attri)
    return remove_list


"""------------------------------------------------------------------------"""

LOCATIONS_LIST = get_file_list()

df_list = read_location_data(LOCATIONS_LIST)

result_df = pd.concat(
    [df.select_dtypes(exclude="object") for df in df_list], ignore_index=True
)

corr_mat = result_df.corr()

# plot_corr_matrix(corr_mat)
attri_list = get_high_curr_attri(corr_mat)
# print(type(attri_list))
print(attri_list)

"""------------------------------------------------------------------------"""
def modeling(result_df: pd.DataFrame, attri_list): 
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    X = result_df[attri_list]
    y = result_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=8412
    )
    pipe.fit(X_train, y_train)

    return pipe

X = result_df[attri_list]
y = result_df[TARGET]

pipe = modeling(result_df, attri_list)
rm_list = remove_list_analyze(pipe, attri_list)
print(rm_list)
for x in rm_list:
    attri_list.remove(x)
score = calc_loss(modeling(result_df, attri_list), X, y)
print(score)
print(np.mean(score))

"""------------------------------------------------------------------------"""

# ATTRI_LIST = ['Tempmax','Dew','Precipprob','']
