import copy

import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re

import RyStats
from RyStats import common
from RyStats import dimensionality
from RyStats import plots

from scipy.optimize import minimize
from math import sqrt

import factor_analyzer
from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.utils import partial_correlations
from factor_analyzer.rotator import Rotator
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

from bokeh.plotting import show

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
cmap = ["#FD6467", "#C6CDF7", "#9c1d18", "#D8A499", "#D67236"]

sns.set_palette(cmap)


#### general data handling functions####


def get_question_items(df, col_str: str):
    """
    This function gives a subset of the dataframe containing only the questions.
    ------
    Input:
    - ABCD behavioral data (e.g., Child behavior checklist data)
    ------
    Output:
    -  data frame
    """
    question_items = df.loc[:, [col for col in df.columns if col_str in col]]
    return question_items


def get_data_with_followups(df, followups="all"):
    """
    This function gives the subset of the ABCD data that also has the needed follow up data.
    ------
    Input:
    - ABCD behavioral data (e.g., Child behavior checklist data)
    ------
    Output:
    -  data frame
    """

    subject_ids = []
    events = sorted(list(df.eventname.unique()), key=lambda x: (x[0].isdigit(), x))

    for subject in df["src_subject_id"].unique():

        subject_rows = df[df.src_subject_id == subject]

        if followups == "all":
            events = pd.DataFrame(events[:])
            if (events.isin(list(subject_rows.eventname))).all().all():
                subject_ids.append(subject)

        elif followups == "1y_follow_up":
            events = pd.DataFrame(events[0:2])
            if (events.isin(list(subject_rows.eventname))).all().all():
                subject_ids.append(subject)

        elif followups == "2y_follow_up":
            events = pd.DataFrame(events[0:3])
            if (events.isin(list(subject_rows.eventname))).all().all():
                subject_ids.append(subject)

        elif followups == "3y_follow_up":
            events = pd.DataFrame(events[0:4])
            if (events.isin(list(subject_rows.eventname))).all().all():
                subject_ids.append(subject)

        else:
            raise NameError("This measurement point is not defined.")

    df_new = df[df.src_subject_id.isin(subject_ids)]

    return df_new


def keep_english(x: str, raiseWarning=False) -> str:
    """
    Splits a string into two and keeps the first half.
    The marking point is defined as the second capital letter in the string.
    This corresponds to the starting of the Spanish question in the dataset.
    The function is used to format the question names in the table

    Args:
        x (str): input string

    Returns:
        str: English string
    """
    try:
        letter_remove = [match.start() for match in re.finditer("[A-Z]", x)][1]
    except IndexError:
        if raiseWarning:
            print(
                f"Question '{x}' has no second capital letter in it. Time for manual inspection!"
            )
        return x
    return x[: letter_remove - 1]


#### functions for EFA ####


def abcd_dict(file):
    """
    This function returns a data dictionary containing variable names and variable descriptions
    ------
    Input:
    - ABCD file as string, e.g. 'abcd_mri01' without .txt extension
    ------
    Output:
    - abcd_dict
    """
    path = "data/{}.txt"
    f = path.format(file)
    header = pd.read_csv(f, header=None, sep="\t", nrows=2)
    abcd_dict = dict(zip(header.iloc[0, :], header.iloc[1, :]))

    for key in abcd_dict.keys():
        abcd_dict[key] = keep_english(abcd_dict[key])
    # print key, value per line
    # [print(key, ':', value) for key, value in abcd_dict.items()]
    return abcd_dict


def abcd_data(measurement="all", load=True, df=None):
    """
    This function loads data from the ABCD study and returns a subset based on the
    chosen measurement time point without duplicates.
    ------
    Parameters:
    measurement {'all', 'baseline', '1y_follow_up', '2y_follow_up', '3y_follow_up'}, (str): time point selection
    ------
    Output:
    - dataframe
    """

    if load:
        # load header info and store in dictionary
        header = pd.read_csv(
            "data/abcd_cbcl01.txt", header=None, sep="\t", nrows=2
        )  # read header
        # load data and use header info to name columns
        data = pd.read_csv(
            "data/abcd_cbcl01.txt", header=None, sep="\t", skiprows=2
        )  # read data
        data.columns = list(header.iloc[0, :])

    if load != bool(isinstance(df, pd.DataFrame)):
        if load:
            df = data

        # drop duplicates
        df = df.drop_duplicates(
            subset=["subjectkey", "interview_date", "interview_age"], ignore_index=False
        )

        # select only a specific measurememt time point
        if measurement == "baseline":
            data = df.loc[df["eventname"] == "baseline_year_1_arm_1"]
        elif measurement == "1y_follow_up":
            data = df.loc[df["eventname"] == "1_year_follow_up_y_arm_1"]
        elif measurement == "2y_follow_up":
            data = df.loc[df["eventname"] == "2_year_follow_up_y_arm_1"]
        elif measurement == "3y_follow_up":
            data = df.loc[df["eventname"] == "3_year_follow_up_y_arm_1"]
        elif measurement == "all":
            data = df
        else:
            data = None
            raise NameError("This measurement point is not defined.")

    else:
        raise ValueError("Please either provide a dataframe or load the ABCD data.")

    return data


def prepare_data(df, threshold=0.995, impute="drop"):
    """
    This function prepares the data by imputing missing data and dropping low frequency
    items below a given threshold.
    ------
    Input:
    - ABCD Child behavior checklist data
    ------
    Parameters:
    - threshold: between 0 and 1, default = .995, select threshold for low frequency items
    - impute ({'drop', 'mean', 'median'}), default = drop, select missing value handling option
    ------
    Output:
    -  dataframe
    """

    # check for missing values
    null_data = df[df.isnull().any(1)]
    id_nans = null_data["subjectkey"]

    # handling of missing values
    if impute == "drop":
        # remove the subjects with missing values
        df_nmv = df[
            ~df["subjectkey"].isin(id_nans)
        ]  # 11868 out of 11876 rows (subjects) remaining
    elif impute == "mean":
        # replace missing values with the mean
        df_nmv = df.fillna(df.mean())
    elif impute == "median":
        # replace missing values with the median
        df_nmv = df.fillna(df.median())
    else:
        df_nmv = None
        raise NameError("This method is not defined.")

    if 0 <= threshold <= 1:
        # remove items for which the frequency is too low (99.5% rated 0)
        count_zero = (get_question_items(df_nmv, "cbcl_q") == 0).sum()
        id_zeros = count_zero[count_zero.divide(len(df_nmv)) > threshold]
        sum(
            count_zero.divide(len(df_nmv)) > threshold
        )  # n = 5, matches with the original paper
        df_1 = df_nmv.drop(
            id_zeros.keys(), axis=1
        )  # 125 out of 130 columns (items) remaining
    else:
        raise ValueError("Value out of bound. Threshold must lie in between 0 and 1.")

    return df_1


def filter_polychoric_corr(df, threshold=0.75, print_corr=True):
    """
    This function performs a polychoric correlation to identify aggregates. Items with a
    correlation above a given threshold are identified as a cluster and returned.
    ------
    Input:
    - ABCD Child behavior checklist data
    ------
    Parameters:
    - threshold: default = .75, select threshold for correlated aggregates
    ------
    Output:
    -  List of items to aggregate (lists).
    """
    description = abcd_dict("abcd_cbcl01")
    polychoric_corr = common.polychoric.polychoric_correlation_serial(
        get_question_items(df, "cbcl_q").to_numpy().T, 0, 3
    )
    col_names = get_question_items(df, "cbcl_q").columns  # get the column names

    filtered_corr = np.where(polychoric_corr > threshold, 1, 0)
    # above filter the array so that those with high correlation are 1 and those with low are 0
    np.fill_diagonal(
        filtered_corr, 0
    )  # make diagonal entries 0 as item correlation with itself is 1 by definition and is useless
    filtered_corr = np.triu(
        filtered_corr
    )  # take the upper triangular to avoid repeating items

    a, b = np.nonzero(
        filtered_corr
    )  # get x and y coordinates of nonzero items in two 1d arrays
    drop_me = []
    for i in range(len(a)):
        if print_corr:
            print(
                f"{col_names[a[i]]} with {col_names[b[i]]}\n\n"
            )  # print pairs of correlated items
            print(
                f"{description[col_names[a[i]]]} with \n{description[col_names[b[i]]]}\n\n"
            )
        drop_me.append(col_names[a[i]])
        drop_me.append(col_names[b[i]])

    drop_me = set(drop_me)  # get unique entries only

    # cluster correlated columns to identify the ones to merge
    cluster = []
    cluster_list = []
    for i in range(len(a)):
        cluster = []
        cluster.append(col_names[a[i]])
        cluster.append(col_names[b[i]])
        cluster_list.append(set(cluster))

    new_cluster_list = []

    while len(cluster_list) > 0:
        check_set = cluster_list.pop()

        for i in cluster_list:
            if bool(check_set & i) == True:
                cluster_list.remove(i)
                check_set = check_set.union(i)

        new_cluster_list.append(list(check_set))

    return new_cluster_list


def aggregate_items(df, item_list, aggregate_dict):
    """
    This function takes a list of clustered items and aggregates them in new question
    items. The labelling process should be done manually, to create meaningful clusters.
    ------
    Input:
    - ABCD Child behavior checklist data, list of clusters
    ------
    Output:
    - dataframe
    """

    # average the columns and round to the nearest integer
    for key in aggregate_dict:
        df[key] = df[aggregate_dict[key]].mean(axis=1).round()

    # drop the columns of the aggregated items
    to_drop = sum(item_list, [])
    df_2 = df.drop(columns=to_drop)

    return df_2


def adequacy_test(df, corr_matrix):
    """
    This function tests if a dataframe is suitable for running an EFA using Bartlett's
    Test of Sphericty and the Kaiser-Meyer-Olkin Measure of Sampling Adequacy.
    ------
    Input:
    - dataframe
    ------
    Output:
    - dataframe

    NOTE: use a polychoric correlation matrix if the data is not normally distributed.
    """
    # Barlett's sphericity
    n, p = df.shape
    corr_det = np.linalg.det(corr_matrix)
    statistic = -np.log(abs(corr_det)) * (n - 1 - (2 * p + 5) / 6)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = scipy.stats.chi2.sf(statistic, degrees_of_freedom)
    statistic, p_value

    if p_value < 0.05:
        print(
            f"The data is suitable for an EFA. Sphericity test was significant with a p-value of {p_value}.\n\n"
        )
    elif p_value > 0.05:
        print(
            f"The data is not suitable for an EFA. Sphericity test was not significant with a p-value of {p_value}.\n\n"
        )
    else:
        raise ValueError("The Sphericity test was not conducted successfully")

    # KMO
    partial_corr = partial_correlations(get_question_items(df, "cbcl_q"))
    x_corr = corr_matrix
    np.fill_diagonal(x_corr, 0)
    np.fill_diagonal(partial_corr, 0)

    partial_corr = partial_corr**2
    x_corr = x_corr**2

    # calculate KMO per item
    partial_corr_sum = np.sum(partial_corr, axis=0)
    corr_sum = np.sum(x_corr, axis=0)
    kmo_per_item = corr_sum / (corr_sum + partial_corr_sum)

    # calculate KMO overall
    corr_sum_total = np.sum(x_corr)
    partial_corr_sum_total = np.sum(partial_corr)
    kmo_total = corr_sum_total / (corr_sum_total + partial_corr_sum_total)

    if kmo_total >= 0.5:
        print(
            f"The data is suitable for an EFA. KMO test resulted in a value of {kmo_total}.\n\n"
        )
    elif kmo_total < 0.5:
        print(
            f"The data is not suitable for an EFA. KMO test resulted in a value of {kmo_total}.\n\n"
        )
    else:
        raise ValueError("The KMO test was not conducted successfully")
    return None


def plot_diagrams(df, diagram, ev_df=None):
    """
    This function plots Matrix and Scree plots of the data.
    ------
    Input:
    - dataframe, String
    ------
    Parameters:
    - diagram ({'Matrix', 'Scree'}), select plotting option
    ------
    Output:

    """

    if diagram == "Matrix":
        fig, ax = plt.subplots()
        c = ax.pcolor(abs(df))
        fig.colorbar(c, ax=ax)
        ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
        ax.set_yticklabels(df.index.values)
        ax.set_xticklabels(df.columns.values)
        plt.show()

    elif diagram == "Scree":

        if ev_df is not None:
            plt.scatter(range(1, df.shape[1] + 1), ev_df.iloc[1, :])
            plt.plot(range(1, df.shape[1] + 1), ev_df.iloc[1, :])
            plt.title("Scree Plot")
            plt.xlabel("Factors")
            plt.ylabel("Eigenvalue")
            plt.axhline(y=1, c="k")

        else:
            raise ValueError("Dataframe for Eigenvalues is required.")

    else:
        raise NameError("Plots must either be 'Scree' or 'Matrix'.")

    return None


def color(val):
    if val >= 0.5 or val <= -0.5:
        color = "#2ecc71"
    elif val >= 0.35 or val <= -0.35:
        color = "#f1c40f"
    else:
        color = "white"
    return "background-color: %s" % color


def get_primary_loadings(df, delete=False):

    for col in df.columns:

        df = df.sort_values(col, ascending=False)

        high_loadings = df[col][df[col] >= 0.35]

        if len(high_loadings) > 3:
            count_vals = 0
            count_crossloadings = 0

            for item in high_loadings:

                row_vals = df.loc[df[col] == item].to_numpy()
                row_vals[row_vals == item] = 0
                row_vals = row_vals + 0.1

                if (item > row_vals).all():
                    count_vals = count_vals + 1

                else:
                    count_vals = count_vals
                    count_crossloadings = count_crossloadings + 1
                    if delete:
                        # drop the items with crossloadings
                        df.drop(df.loc[df[col] == item].index, inplace=True)

            if count_vals >= 4:
                df[col] = df[col]

            else:
                df.drop(col, axis=1, inplace=True)

        else:
            df.drop(col, axis=1, inplace=True)

    return df


def get_EFA_structure(df, n_factors: int, rotation: str, run_again=False):
    """Perform an EFA with the specified parameters. Return factor scores and loadings.
    Args:
        df (pd.DataFrame): DataFrame containing question of items
        n_factors (int): Number of factors to extract
        rotation (str): Rotation method to applied in the EFA. Must be one of geomin_obl, varimax, or promax
        run_again (bool, optional): If True, runs the EFA again after removing items that load lowly on all factors. Defaults to False.

    Raises:
        ValueError: If rotation method is invalid

    Returns:
        factor_scores: DataFrame with number of participants as number of rows and number of factors as number of columns.
        rotated_loadings: DataFrame with different question items as rows and different factors as columns
    """

    legal_rotation = ["geomin_obl", "varimax", "promax"]

    if rotation not in legal_rotation:
        raise ValueError(f"Rotation must be one of {legal_rotation}.")

    EFA_df = get_question_items(df, "cbcl_q")

    efa = FactorAnalyzer(n_factors, rotation, is_corr_matrix=False, method="principal")
    efa.fit(EFA_df)

    # loading matrix
    rotated_loadings = pd.DataFrame(efa.loadings_).abs()
    rotated_loadings.set_index(EFA_df.columns, inplace=True)

    # factor scores
    factor_scores = pd.DataFrame(efa.transform(EFA_df))

    if run_again:

        rotated_loadings = get_primary_loadings(rotated_loadings, delete=True)
        EFA_df = EFA_df[rotated_loadings.index]

        list_to_drop = []
        threshold = 0.35

        for row_index in rotated_loadings.index:
            row_vals = rotated_loadings.loc[row_index].to_numpy()

            if (row_vals < threshold).all():
                list_to_drop.append(row_index)

        EFA_df = EFA_df.drop(EFA_df[list_to_drop], axis=1)

        efa2 = FactorAnalyzer(
            n_factors, rotation, is_corr_matrix=False, method="principal"
        )
        efa2.fit(EFA_df)

        # loading matrix
        rotated_loadings = pd.DataFrame(efa2.loadings_).abs()
        rotated_loadings.set_index(EFA_df.columns, inplace=True)

        # factor scores
        factor_scores = pd.DataFrame(efa2.transform(EFA_df))

    return factor_scores, rotated_loadings


#### function hierarchy####
def plot_hierarchy(corr_dict: dict, name_dict: dict, threshold: float, save_as: str):
    """This is an incredibly messy function to quickly inspect the hierarchical factor structure.
    It is not written to be understood but a very hacky solution that barely works.

    Args:
        corr_dict (dict): Dictionary with one key per correlation structure between one level and the next (e.g. factor_1factor_2)
                          and values being i x j matrices where i being the number of factors in the smaller solution and j being
                          the number of factors in the larger solution. Entry [i,j] in this matrix should represent the correlation
                          between the two respective factors
        name_dict (dict): One key per level and values are lists containing the names of the factors in an ordered fashion (from factor 1 to factor max)
        threshold (float): Correlation threshold below which the correlations will not be annotated on the plot
        save_as (str): file name that you wish to save the plot to
    """
    G = nx.Graph()
    total_items = np.arange(1, len(name_dict) + 1).sum()
    labels = [x for x in name_dict.values()]
    labels = [item for sublist in labels for item in sublist]
    counter = 0
    x = 0
    y = 0
    for value in name_dict.values():
        x = 0
        for item in value:
            G.add_node(counter, pos=(x, y))
            x += 1
            counter += 1
        y += 1

    start = 1
    stop = 2
    swap_time = 0
    swap_every = 1
    to_add = 1
    to_remove = 1
    for i in range(total_items - len(name_dict)):

        for j in range(start, stop + 1):
            G.add_edge(i, j)

        swap_time += 1

        if swap_time == swap_every:
            swap_every += 1
            swap_time = 0
            old_start = start
            start = stop + 1
            stop = old_start + start + to_add
            to_add -= to_remove
            to_remove += 1

    edge_labels = {}
    corrs = [x.flatten() for x in corr_dict.values()]
    corrs = [item for sublist in corrs for item in sublist]
    for edge, item in zip(G.edges(), corrs):
        edge_labels[edge] = round(item, 2)

    to_remove = []
    iter_edge_labels = copy.copy(edge_labels)
    for key, value in iter_edge_labels.items():

        if np.abs(value) < threshold:
            to_remove.append(key)
            del edge_labels[key]

    G.remove_edges_from(to_remove)

    pos = nx.get_node_attributes(G, "pos")
    pos = {node: (x, -y) for (node, (x, y)) in pos.items()}
    nx.draw(
        G,
        pos,
        node_size=0.1,
        bbox=dict(facecolor="skyblue"),
        labels={x: labels[x] for x in range(total_items)},
        font_size=8,
    )

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(save_as, format="PNG")
    plt.show()

    return


def factor_extract(df, n_factors: int, rotation_opt: str):
    """Does an EFA on the given df with the specified number of factors and the specified rotation.
       Returns the factor scores and the loadings

    Args:
        df (pd.DataFrame): DataFrame containing the question items
        n_factors (int): Number of factors to extract
        rotation_opt (str): Rotation method applied. Must be one of geomin_obl, varimax, or promax

    Raises:
        ValueError: Error raised when an illegal rotation method is passed

    Returns:
        factor_scores_dict: Dictionary containing one key per factor (factor_x) where the values are DataFrames with
                            participants as rows and factors as columns
        corr_dict: Dictionary with one key per correlation structure between one level and the next (e.g. factor_1factor_2)
                   and values being i x j matrices where i being the number of factors in the smaller solution and j being
                   the number of factors in the larger solution. Entry [i,j] in this matrix should represent the correlation
                   between the two respective factors
    """

    legal_rotation = ["geomin_obl", "varimax", "promax"]

    if rotation_opt not in legal_rotation:
        raise ValueError(f"Rotation must be one of {legal_rotation}.")

    factor_scores_dict = {}
    corr_dict = {}

    for i in range(1, n_factors + 1):
        dict_values = []
        name = "factor_"
        rotation = None if i == 1 else rotation_opt

        factor_sol = FactorAnalyzer(i, rotation, method="principal")
        factor_sol.fit(get_question_items(df, "cbcl_q"))
        factor_scores = pd.DataFrame(
            factor_sol.transform(get_question_items(df, "cbcl_q"))
        )

        for col in factor_scores.columns:
            dict_values.append(factor_scores.iloc[:, col])

        name = name + str(i)
        factor_scores_dict[name] = dict_values

    for i in range(1, len(factor_scores_dict)):
        name = "factor_"
        name1 = name + str(i)
        name2 = name + str(i + 1)

        corrs = np.zeros([i, i + 1])
        for i1, m in enumerate(factor_scores_dict[name1]):
            for i2, n in enumerate(factor_scores_dict[name2]):
                corrs[i1, i2] = n.corr(m, method="pearson", min_periods=None)

        corr_dict[name1 + name2] = corrs

    return factor_scores_dict, corr_dict


#### functions sex differences####


def sex_corr(df, n_factors):
    """Do an EFA in a hierarchical manner up to and including the specified number of factors.
       Seperate factor scores by sex for each factor at each level of the hierarchy.
       Compute descriptive statistics and formal statistical comparisons between sexes for
       each factor and put all of these in a DataFrame. Additionally return the factor structure
       for each sex seperately.

       NOTE: p-values are not corrected for multiple comparisons inside this function.
    Args:
        df (pd.DataFrame): DataFrame containing all question items and responses
        n_factors (_type_): Number of factors to extract

    Returns:
        sex_diff_df: DataFrame containing descriptive and comparison statistics for all factors at each level
        factor_scores_F: Array containing factor scores (participants as rows and factors as columns) for female
        factor_scores_M: Array containing factor scores (participants as rows and factors as columns) for male
    """

    # check the different factor scores for different number of solutions

    EFA_df = get_question_items(df, "cbcl_q")

    factor_scores_dict = {}

    for i in range(1, n_factors + 1):
        dict_values = []
        name = "factor_"
        rotation = None if i == 1 else "geomin_obl"

        factor_sol = FactorAnalyzer(i, rotation, method="principal")
        factor_sol.fit(EFA_df)
        factor_scores = pd.DataFrame(factor_sol.transform(EFA_df))

        factor_scores.set_index(EFA_df.index, inplace=True)

        for col in factor_scores.columns:

            dict_values.append(pd.DataFrame(factor_scores[col]))

        name = name + str(i)
        factor_scores_dict[name] = dict_values

    factor_scores_dict

    sex_diff_df = pd.DataFrame(
        columns=["Mean_F", "SD_F", "Mean_M", "SD_M", "t-value", "Cohen's d"]
    )

    for key in factor_scores_dict.keys():

        index = 0
        for elem in factor_scores_dict[key]:
            name = key + "_" + str(index + 1)

            factor_scores = factor_scores_dict[key][index]
            factor_scores_F = factor_scores.loc[df[df["sex"] == "F"].index].to_numpy()
            factor_scores_M = factor_scores.loc[df[df["sex"] == "M"].index].to_numpy()

            p_val = scipy.stats.ttest_ind(factor_scores_F, factor_scores_M)[1]

            new_row = pd.DataFrame(
                {
                    "Mean_F": factor_scores_F.mean(),
                    "SD_F": np.std(factor_scores_F),
                    "Mean_M": np.mean(factor_scores_M),
                    "SD_M": np.std(factor_scores_M),
                    "t-value": scipy.stats.ttest_ind(factor_scores_F, factor_scores_M)[
                        0
                    ],
                    "Cohen's d": (np.mean(factor_scores_F) - np.mean(factor_scores_M))
                    / (
                        np.sqrt(
                            (
                                np.std(factor_scores_F) ** 2
                                + np.std(factor_scores_M) ** 2
                            )
                            / 2
                        )
                    ),
                    "adj. p-value": "ns" if p_val > 0.001 else "<.001",
                },
                index=[str(name)],
            )

            sex_diff_df = pd.concat([sex_diff_df, new_row])

            index = index + 1

    return sex_diff_df, factor_scores_F, factor_scores_M


def get_gender_congruency(report: str, timepoint="1_year_follow_up_y_arm_1"):
    """Computes the normalised gender congruency scores for the given questionnaire and time-point.
       Saves them as a csv file and returns them as a DataFrame.

       For boys and girls, the reversed version of the same questions are presented (e.g. does he act like
       a girl for boys, and does she act like a boy for girls). We align these questions for the two sexes.
       Then we normalise the questionnaire scores between 0 and 1 for each question


    Args:
        report (str): parent or youth. Specifies which questionnaire to pick
        timepoint (str, optional): From which year to pick the questionnaire. Defaults to '1_year_follow_up_y_arm_1'.

    Raises:
        ValueError: If illegal questionnaire is requested

    Returns:
        gender_congruency : DataFrame containing gender congruency scores where each row is a participant and each column is a (transformed) question item.
    """

    if report not in ["parent", "youth"]:
        raise ValueError("Report must be one of 'parent' or 'youth'.")

    elif report == "parent":
        df = pd.read_csv("data/abcd_pgi01.txt", sep="\t", low_memory=False)  # read data

    elif report == "youth":
        df = pd.read_csv("data/abcd_ygi01.txt", sep="\t", low_memory=False)  # read data

    data = df.loc[df["eventname"] == timepoint]
    # drop duplicates
    data = data.drop_duplicates(
        subset=["subjectkey", "interview_date", "interview_age"], ignore_index=False
    )

    # split into female data and preprocess
    data_F = data[data["sex"] == "F"]
    items_F = get_question_items(data_F, "gish_f")
    items_F = items_F.replace("777", None)
    items_F = items_F.replace("0", None)
    items_F = items_F.apply(pd.to_numeric)
    to_drop_F = items_F[items_F.isna().all(axis=1)].index
    gp_items_F = items_F.drop(to_drop_F, axis=0)

    # split into male data and preprocess
    data_M = data[data["sex"] == "M"]
    items_M = get_question_items(data_M, "gish_m")
    items_M = items_M.replace("777", None)
    items_M = items_M.replace("0", None)
    items_M = items_M.apply(pd.to_numeric)
    to_drop_M = items_M[items_M.isna().all(axis=1)].index
    gp_items_M = items_M.drop(to_drop_M, axis=0)

    # calculate female gender congruency scores
    gender_congruency_F = pd.DataFrame(columns=gp_items_F.columns)

    for row in gp_items_F.index:
        row_mean = gp_items_F.loc[[row]].mean(axis=1)

        # first fill nans with mean
        gp_items_F.loc[row].fillna(float(row_mean), inplace=True)

        # the recalculate to 0-1 score
        row_val = np.array(gp_items_F.loc[row].values)
        row_norm = (row_val - 1) / 4

        # add to GC df
        gender_congruency_F.loc[row] = row_norm

    # calculate male gender congruency scores
    gender_congruency_M = pd.DataFrame(columns=gp_items_M.columns)

    for row in gp_items_M.index:
        row_mean = gp_items_M.loc[[row]].mean(axis=1)

        # first fill nans with row mean -> should i do column mean?
        gp_items_M.loc[row].fillna(float(row_mean), inplace=True)

        # the recalculate to 0-1 score
        row_val = np.array(gp_items_M.loc[row].values)
        row_norm = (row_val - 1) / 4

        # add to GC df
        gender_congruency_M.loc[row] = row_norm

    # combine into shared dataframe
    GC_col = gender_congruency_M.columns.str.replace("m", "")
    gender_congruency_M.columns = GC_col
    gender_congruency_F.columns = GC_col

    # concat the dataframes (items correspond)
    frames = [gender_congruency_M, gender_congruency_F]
    gender_congruency = pd.concat(frames)

    # add the subject ID and sex
    gender_congruency["sex"] = data.loc[gender_congruency.index]["sex"]
    gender_congruency["src_subject_id"] = data.loc[gender_congruency.index][
        "src_subject_id"
    ]

    # export the data frame
    gender_congruency.to_csv(f"final_results/GC_{report}.csv", index=False)

    return gender_congruency


def predict_sex_diff(factor_scores, predictor_array, factor: str, cv_no=50):
    """Predict the given factor scores with the given predictors using linear regression in a cross-validated fashion

    Args:
        factor_scores (pd.DataFrame): DataFrame containing factor scores
        predictor_array (np.ndarray): Design matrix. Rows are observations and columns are features
        factor (str): Which factor to predict
        cv_no (int, optional): Number of cross-validation folds. Defaults to 50.
    Returns:
        test_scores: list of test scores for each fold
        weights: list of weights computed on all observations
    """

    allowed_factors = [
        "Externalizing",
        "Internalizing",
        "Neurodevelopmental",
        "Detachment",
        "Somatoform",
        "Antisocial Behavior",
    ]

    # prepare the dataframes and match their indices
    y = factor_scores.sort_values(by=["src_subject_id"])
    y_df = y.rename(
        columns={
            0: allowed_factors[0],
            1: allowed_factors[1],
            2: allowed_factors[2],
            3: allowed_factors[3],
            4: allowed_factors[4],
            5: allowed_factors[5],
        }
    )
    Y = y_df[factor].values

    my_model = LinearRegression()  # initialize the model
    scores = cross_validate(my_model, predictor_array, Y, cv=cv_no)
    test_scores = scores["test_score"]

    model_fit = my_model.fit(predictor_array, Y)
    weights = model_fit.coef_

    return test_scores, weights


def get_predictor_df(factor_list, predictor_dict, factor_scores, full = False):
    """Obtain a data frame of 

    Args:
        factor list: 
        predictor dict:
        factor scores:
        full (bool): if True table includes R^2 values from each individual fold
                     if False table includes mean R^2 values and significance results
        
    Returns:
        data frame
        
    """
    result_list = []
    mean_to_df = []

    for factor in factor_list:

        for predictor in predictor_dict:
            test_scores, weigths = predict_sex_diff(factor_scores, predictor_dict[predictor], factor, cv_no = 100)

            for score in test_scores:
                current_list = []
                current_list = [factor, predictor, score]
                result_list.append(current_list)


            mean = test_scores.mean()
            mean_to_df.append([factor, predictor, round(mean, 4)])
            
    result_df = pd.DataFrame(result_list, columns = ['Factor', 'Predictor', 'R^2'])

    if not full:
        # test for significance against null
        result_df['adj. p-values'] = result_df['R^2']
        tested_df = result_df.groupby(['Factor', 'Predictor'], as_index = False)['adj. p-values'].apply(lambda x: scipy.stats.ttest_1samp(x, 0, axis=0, nan_policy='propagate', alternative='greater')[1]*len(mean_to_df))
        
        # combine the significance and prediction results
        mean_df = pd.DataFrame(mean_to_df, columns = ['Factor', 'Predictor', 'R^2'])
        mean_df = mean_df.sort_values(by = ['Factor', 'Predictor'])
        mean_df['adj. p-values'] = tested_df['adj. p-values'].values
        result_df = mean_df


    return result_df



def display_p_val(x):
    """Display the p-value in a nice way

    Args:
        x (float): p-value
    """

    if x > 0.05:
        return " ns"
    elif x < 0.001:
        return "***"
    elif x < 0.01:
        return "**"
    elif x < 0.05:
        return "*"
    else:
        return str(np.round(x, 3))
