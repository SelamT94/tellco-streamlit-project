import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Ridge

# def fill_missing_values(data):
#     # Fill missing values in numeric columns with the mean
#     numeric_columns = data.select_dtypes(include=np.number).columns
#     data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
#     # Fill missing values in datetime columns with the most frequent value if available
#     datetime_columns = data.select_dtypes(include='datetime').columns
#     if not data[datetime_columns].empty:
#         mode_values = data[datetime_columns].mode().iloc[0]
#         data[datetime_columns] = data[datetime_columns].fillna(mode_values)
    
#     return data
def fill_missing_values(data):
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    return data

def perform_data_transformation(data):
    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=np.number)
    
    # Perform Min-Max scaling on the numeric data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    return scaled_data


def assign_engagement_score(user_data, cluster_data):
    distances = euclidean_distances(user_data, cluster_data[:, :-1])  # Exclude the last column from cluster_data
    least_engaged_cluster_index = distances.sum(axis=1).argmin()
    engagement_scores = distances[:, least_engaged_cluster_index]
    return engagement_scores

def assign_experience_score(user_data, cluster_data):
    distances = euclidean_distances(user_data, cluster_data[:, :-1])  # Exclude the last column from cluster_data
    worst_experience_cluster_index = distances.sum(axis=1).argmax()
    experience_scores = distances[:, worst_experience_cluster_index]
    return experience_scores

def get_top_satisfied_customers(data, n=10):
    data['Satisfaction Score'] = (data['Engagement Score'] + data['Experience Score']) / 2
    sorted_data = data.sort_values(by='Satisfaction Score', ascending=False)
    top_satisfied_customers = sorted_data.head(n)
    return top_satisfied_customers


# def build_regression_model(data, target_col, test_size=0.2, random_state=42):
#     numeric_data = data.select_dtypes(include=np.number)
#     X = numeric_data.drop(columns=[target_col])
#     y = numeric_data[target_col]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
#     # Define the regression model
#     model = LinearRegression()
    
#     # Define hyperparameters for grid search
#     parameters = {'fit_intercept': [True, False]}
    
#     # Perform grid search to find the best hyperparameters
#     grid_search = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
#     grid_search.fit(X_train, y_train)
    
#     # Get the best model from grid search
#     best_model = grid_search.best_estimator_
    
#     # Train the best model
#     best_model.fit(X_train, y_train)
    
#     # Predict target variable on the testing set
#     y_pred = best_model.predict(X_test)
    
#     # Calculate Mean Squared Error (MSE) on the testing set
#     mse = mean_squared_error(y_test, y_pred)
    
#     return best_model, mse


def build_regression_model(X_train, y_train):
    model = Ridge()
    parameters = {'alpha': [0.1, 1, 10]}
    grid_search = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, rmse, r2

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    user_data = pd.read_csv("results/cleaned_data.csv")
    clustered_data = pd.read_csv("results/experience_clusters.csv")
    

    user_data = fill_missing_values(user_data)
    user_data_transformed = perform_data_transformation(user_data.drop('MSISDN/Number', axis=1))
    numeric_columns = user_data.select_dtypes(include=np.number).columns
    cluster_data = clustered_data.groupby('Cluster')[numeric_columns].mean().values
 
    # Assign engagement scores
    engagement_scores = assign_engagement_score(user_data_transformed, cluster_data)
    
    # Assign experience scores
    experience_scores = assign_experience_score(user_data_transformed, cluster_data)
    
    user_data_with_scores = user_data.copy()
    user_data_with_scores['Engagement Score'] = engagement_scores
    user_data_with_scores['Experience Score'] = experience_scores
    user_data_with_scores['Satisfaction Score'] = (user_data_with_scores['Engagement Score'] + user_data_with_scores['Experience Score']) / 2

    user_data_with_scores.to_csv("results/user_data_with_scores.csv", index=False)

    top_10_satisfied_customers = get_top_satisfied_customers(user_data_with_scores, n=10)
    print("Top 10 Satisfied Customers:")
    print(top_10_satisfied_customers)

    # Regression model

    X = user_data_transformed
    y = user_data_with_scores['Satisfaction Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = build_regression_model(X_train, y_train)

 # Evaluation
    mse, mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2):", r2)

    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    mean_cv_mse = -cv_scores.mean()
    print("Mean Cross-Validation MSE:", mean_cv_mse)

    # target_column = 'Satisfaction Score'
    # trained_model, mse = build_regression_model(user_data_with_scores, target_col=target_column)
    # print("Mean Squared Error (MSE):", mse)
    
    # # Split data into training and testing sets
    # numeric_data = user_data_with_scores.select_dtypes(include=np.number)
    # X = numeric_data.drop(columns=[target_column])
    # y = numeric_data[target_column]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Perform 5-fold cross-validation
    # cv_scores = cross_val_score(trained_model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')

    # # Calculate mean MSE across folds
    # mean_mse = -cv_scores.mean()

    # # Print the results
    # print("Mean Squared Error (MSE):", mean_mse)

if __name__ == "__main__":
    main()
