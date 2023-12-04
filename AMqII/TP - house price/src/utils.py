"""
utils.py

This script contains various utility functions and a custom transformer for data analysis 
and preprocessing.

Imports:
- pandas as pd
- numpy as np
- seaborn as sns
- matplotlib.pyplot as plt
- scipy.stats as stats
- train_test_split, GridSearchCV from sklearn.model_selection
- BaseEstimator, TransformerMixin from sklearn.base

Functions:
- calculate_outlier_percentage: Calculate the percentage of outliers for each column 
in a DataFrame.
- calculate_null_percentage: Calculate the percentage of null values for each column 
in a DataFrame.
- outlier_diagnostic_plots: Display diagnostic plots (histogram, QQ plot, box plot) for 
outlier analysis.
- feature_target_correlation_df: Calculate feature-target variable correlations 
in a DataFrame.
- boxplot: Create a customized box plot for visualization.
- load_and_split_data: Split data into features and target variables for training and 
validation sets.
- train_regressor: Train a regression model using GridSearchCV for hyperparameter tuning.
- CappingTransformer: A custom transformer for capping numerical data using Interquartile 
Range (IQR).

Classes:
- CappingTransformer: A transformer class for capping (winsorizing) numerical data.

Usage:
- Import this script to access utility functions and the CappingTransformer class for 
data analysis and preprocessing.

Note:
- Each function and class includes a docstring describing its purpose, parameters, and usage.


DESCRIPCIÓN: utils.py
AUTOR: Clara Bureu - Maximiliano Medina - Luis Pablo Segovia
FECHA: 01/12/2023
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV


def calculate_outlier_percentage(data, threshold=1.5):
    """
    Calculates the percentage of outliers in each column of a DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the data to analyze.
        threshold (float, optional): The threshold multiplier for identifying outliers. 
        Defaults to 1.5.

    Returns:
        pandas.DataFrame: A DataFrame containing the outlier percentages for each column.

    Description:
        This function calculates the percentage of outliers in each column of a DataFrame.
        It uses the Interquartile Range (IQR) method to identify outliers.

        For each column, the IQR is calculated and used to define the lower and upper bounds
        for potential outliers. Values that fall outside these bounds are considered outliers.

        The percentage of outliers in each column is then calculated and stored in a DataFrame.
        The DataFrame is sorted in descending order by the "Outlier Percentage" column.

    """
    outlier_percentages = []

    for column in data.columns:
        # Extract the column data as a NumPy array
        column_data = data[column].values

        # Calculate the IQR and the lower and
        # upper bounds for potential outliers
        Q1 = np.percentile(column_data, 25)
        Q3 = np.percentile(column_data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify the outliers
        num_outliers = len([x for x in column_data if x <
                           lower_bound or x > upper_bound])

        # Calculate the percentage of outliers
        percentage = (num_outliers / len(column_data)) * 100

        outlier_percentages.append((column, percentage))

    # Create a new DataFrame with the outlier percentages
    result_df = pd.DataFrame(outlier_percentages,
                             columns=['Column', 'Outlier Percentage [%]'])

    # Sort the DataFrame in descending order by "Outlier Percentage"
    result_df = result_df.sort_values(by='Outlier Percentage [%]',
                                      ascending=False)
    return result_df


def calculate_null_percentage(data):
    """
    Calculates the percentage of null values in each column of a DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the data to analyze.

    Returns:
        pandas.DataFrame: A DataFrame containing the null percentages for each 
        column, sorted by highest percentage first.

    Description:
        This function calculates the percentage of null values in each column 
        of a DataFrame.

        For each column, it calculates the percentage of missing values and stores 
        the results in a DataFrame.
        The DataFrame is then sorted in descending order by the "Null Percentage" 
        column, giving the highest percentage of null values first.

    """
    # Calculate the percentage of null values for each column
    null_percentages = (data.isnull().mean() * 100).round(2)

    # Create a DataFrame to store the results
    result_df = pd.DataFrame({'Column': null_percentages.index,
                              'Null Percentage [%]': null_percentages.values})

    # Sort the DataFrame in descending order by "Null Percentage"
    result_df = result_df.sort_values(
        by='Null Percentage [%]', ascending=False)

    return result_df


def outlier_diagnostic_plots(df, variable):
    """
    Creates three diagnostic plots to assess potential outliers in a DataFrame column:

    * Histogram
    * QQ plot
    * Box plot

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        variable (str): The name of the column to analyze.

    Returns:
        None

    Description:
        This function creates three diagnostic plots to visually assess potential 
        outliers in a DataFrame column.
        The plots are:

        1. Histogram: A histogram of the column's values.
        2. QQ plot: A quantile-quantile (QQ) plot to compare the distribution of 
        the column's values to a normal distribution.
        3. Box plot: A box plot to show the median, quartiles, and range of the 
        column's values.

        The plots are displayed in a single figure with a title indicating the 
        analyzed variable.

    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))

    # Histogram
    sns.histplot(df[variable], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Histogram')

    # QQ plot
    stats.probplot(df[variable], dist="norm", plot=axes[1])
    axes[1].set_title('QQ Plot')

    # Box plot
    sns.boxplot(y=df[variable], ax=axes[2])
    axes[2].set_title('Box & Whiskers')

    fig.suptitle(variable, fontsize=16)
    plt.show()


def feature_target_correlation_df(df, target_column):
    """
    Calculates the correlation coefficients between features and a target variable 
    in a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        target_column (str): The name of the target variable column.

    Raises:
        ValueError: If the target column is not found in the DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing feature-target correlation coefficients 
        sorted by absolute correlation in descending order.

    Description:
        This function calculates the correlation coefficients between each feature (column) 
        and a specified target variable in a DataFrame.
        It utilizes the `corrwith()` method to compute the correlation coefficients for each 
        feature-target pair.

        The correlation coefficients are stored in a DataFrame with columns named 
        "Correlation". Additionally, a new column named "Absolute Correlation" is created,
        which contains the absolute values of the correlation coefficients.

        The DataFrame is sorted in descending order of absolute correlations, effectively 
        ranking features based on their strength of association with the target variable.
        The resulting DataFrame provides insights into the relationship between features 
        and the target variable.

    """
    if target_column not in df.columns:
        raise ValueError("Target column not found in the DataFrame.")

    feature_columns = [col for col in df.columns if col != target_column]
    correlations = df[feature_columns].corrwith(df[target_column])
    correlation_df = pd.DataFrame({'Correlation': correlations})
    correlation_df['Absolute Correlation'] = correlation_df['Correlation'].abs()

    # Sort the DataFrame in descending order of absolute correlations
    correlation_df = correlation_df.sort_values(
        by='Absolute Correlation', ascending=False)
    correlation_df.drop(columns=['Absolute Correlation'], inplace=True)

    return correlation_df


def boxplot(x, y, **kwargs):
    """
    Creates a boxplot to visualize the distribution of a target variable across 
    different categories of an independent variable.

    Parameters:
        x (str): The name of the independent variable column in the DataFrame.
        y (str): The name of the target variable column in the DataFrame.
        **kwargs: Additional keyword arguments to pass to the `sns.boxplot()` function.

    Returns:
        None

    Description:
        This function creates a boxplot to visualize the distribution of a target 
        variable across different categories of an independent variable.
        The boxplot provides a quick overview of how the target variable is distributed 
        within each category of the independent variable.

        The function utilizes the `sns.boxplot()` function from the Seaborn library to 
        generate the boxplot.
        Additional keyword arguments can be passed to the `sns.boxplot()` function using 
        the `**kwargs` parameter.

    """
    sns.boxplot(x=x, y=y, palette="husl")
    plt.xticks(rotation=90)


def load_and_split_data(data_frame, target_col):
    """
    Splits a DataFrame into training and validation sets for machine learning tasks.

    Parameters:
        data_frame (pandas.DataFrame): The DataFrame containing the data to split.
        target_col (str): The name of the target variable column in the DataFrame.

    Returns:
        tuple: A tuple containing the training and validation sets:
            X_train (pandas.DataFrame): The training set features (independent variables).
            X_val (pandas.DataFrame): The validation set features (independent variables).
            y_train (pandas.Series): The training set target variable (dependent variable).
            y_val (pandas.Series): The validation set target variable (dependent variable).

    Description:
        This function splits a DataFrame into training and validation sets for machine 
        learning tasks.
        The training set is used to train the machine learning model, while the validation 
        set is used to evaluate the model's performance.

        The function splits the DataFrame into training and validation sets using the 
        `train_test_split()` function from the scikit-learn library.
        The `test_size` parameter controls the proportion of data allocated to the 
        validation set.
        The random_state parameter ensures reproducibility of the split.

        The returned tuple contains the training set features (X_train), validation 
        set features (X_val), training set target variable (y_train), and validation set 
        target variable (y_val).
    """
    
    # Split the dataset into features (X) and the target variable (y)
    X = data_frame.drop(target_col, axis=1)
    y = data_frame[target_col]

    # Split the data into a training and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val


def train_regressor(model, X_train, y_train, param_grid):
    """
    Performs grid search to find the optimal hyperparameters for a regression model.

    Parameters:
        model (object): The regression model object.
        X_train (pandas.DataFrame): The training set features (independent variables).
        y_train (pandas.Series): The training set target variable (dependent variable).
        param_grid (dict): A dictionary of hyperparameters to tune.

    Returns:
        object: The trained regression model with the optimal hyperparameters.

    Description:
        This function performs grid search to find the optimal hyperparameters for a 
        regression model.
        Grid search is a method for efficiently evaluating multiple combinations of 
        hyperparameter values and selecting the set that results in the best model performance.

        The function utilizes the `GridSearchCV()` function from the scikit-learn 
        library to perform grid search.
        The `cv` parameter specifies the number of cross-validation folds to use for 
        evaluating the model performance on different partitions of the training data.
        The `scoring` parameter specifies the metric used to evaluate the model 
        performance. In this case, `neg_mean_squared_error` is used, which minimizes 
        the mean squared error.

        The function returns the trained regression model with the optimal hyperparameters.

    """
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model


class CappingTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for capping (winsorizing) numerical data using
    the Interquartile Range (IQR) and a threshold.

    Parameters:
    -----------
    threshold : float, optional (default=1.5)
        The threshold multiplier for the IQR. Determines how far beyond
        the IQR the limits should extend.

    Attributes:
    -----------
    lower_limit : float
        The lower limit for capping, calculated during fitting.
    upper_limit : float
        The upper limit for capping, calculated during fitting.

    Methods:
    --------
    fit(X, y=None):
        Calculate the lower and upper limits based on
        the data distribution during training.

    transform(X, y=None):
        Apply capping to the input data using the
        precomputed lower and upper limits.

    Example Usage:
    -------------
    # Create a capping transformer with a threshold of 1.5
    capper = CappingTransformer(threshold=1.5)

    # Fit the transformer on data
    capper.fit(data)

    # Transform the data using the calculated limits
    capped_data = capper.transform(data)
    """

    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.lower_limit = None
        self.upper_limit = None

    def fit(self, X, y=None):
        """
        Fit the capping transformer to the input data and calculate
        lower and upper limits.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data for fitting the transformer.
        y : array-like, optional (default=None)
            Ignored. There is no need for a target variable.

        Returns:
        --------
        self : object
            Returns self for method chaining.
        """
        Q1 = np.percentile(X, 25)
        Q3 = np.percentile(X, 75)
        IQR = Q3 - Q1
        self.lower_limit = Q1 - self.threshold * IQR
        self.upper_limit = Q3 + self.threshold * IQR
        return self

    def transform(self, X, y=None):
        """
        Apply capping to the input data using the precomputed lower and upper limits.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to be capped.

        y : array-like, optional (default=None)
            Ignored. There is no need for a target variable.

        Returns:
        --------
        capped_X : ndarray, shape (n_samples, n_features)
            The capped input data.
        """
        capped_X = np.copy(X)
        capped_X[capped_X < self.lower_limit] = self.lower_limit
        capped_X[capped_X > self.upper_limit] = self.upper_limit
        return capped_X

    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for transformed data. In this case,
        the names are preserved.

        Parameters:
        -----------
        input_features : array-like, shape (n_features,),
        optional (default=None)
            Names of the input features.

        Returns:
        --------
        output_feature_names : array, shape (n_features,)
            The feature names, which are the same as the input feature names.
        """
        return input_features
    