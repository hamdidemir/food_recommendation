import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Define the path to the CSV file
csv_file_path = "data/interactions_train.csv"

# Read the CSV file using pandas
df = pd.read_csv(csv_file_path)
df = df.drop('date', axis=1) #Dropping timestamp

# Let's take a subset of the dataset (by only keeping the users who have given 50 or more ratings) to make the dataset less sparse and easy to work with.
counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= 50].index)]

#Creating the interaction matrix of products and users based on ratings and replacing NaN value with 0
interactions_matrix = df_final.pivot(index = 'user_id', columns ='recipe_id', values = 'rating').fillna(0)

#Finding the number of non-zero entries in the interaction matrix
given_num_of_ratings = np.count_nonzero(interactions_matrix)

#Finding the possible number of ratings as per the number of users and products
possible_num_of_ratings = interactions_matrix.shape[0] * interactions_matrix.shape[1]

#Density of ratings
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100

# Calculate the average rating for each recipe
average_rating = df_final.groupby('recipe_id').mean()['rating']

#Calculate the count of ratings for each recipe
count_rating = df_final.groupby('recipe_id').count()['rating']

#Create a dataframe with calculated average and count of ratings
final_rating = pd.DataFrame({'avg_rating':average_rating, 'rating_count':count_rating})

#Sort the dataframe by average of ratings
final_rating = final_rating.sort_values(by='avg_rating',ascending=False)

interactions_matrix['user_index'] = np.arange(0, interactions_matrix.shape[0])
interactions_matrix.set_index(['user_index'], inplace=True)

# defining a function to get the top n products based on highest average rating and minimum interactions
def rank_based_recommendation(num_recipes, min_interaction):
    # Finding products with minimum number of interactions
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]

    # Sorting values w.r.t average rating
    recommendations = recommendations.sort_values('avg_rating', ascending=False)

    return list(recommendations.index[:num_recipes])


# defining a function to get similar users
def similar_users(user_index):
    similarity = []
    for user in range(0, interactions_matrix.shape[0]):  # .shape[0] gives number of rows

        # finding cosine similarity between the user_id and each user
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])

        # Appending the user and the corresponding similarity score with user_id as a tuple
        similarity.append((user, sim))

    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]  # Extract the user from each tuple in the sorted list
    similarity_score = [tup[1] for tup in
                        similarity]  ##Extracting the similarity score from each tuple in the sorted list

    # Remove the original user and its similarity score and keep only other similar users
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])

    return most_similar_users, similarity_score


# defining the recommendations function to get recommendations by using the similar users' preferences
def user_based_recommendations(user_index, num_of_products):
    #Saving similar users using the function similar_users defined above
    most_similar_users = similar_users(user_index)[0]

    # Finding product IDs with which the user_id has interacted
    prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)]))
    recommendations = []

    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:

            # Finding 'n' products which have been rated by similar users but not by the user_id
            similar_user_prod_ids = set(
                list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break

    return recommendations[:num_of_products]

"""
interactions_matrix = df_final.pivot(index = 'user_id', columns ='recipe_id', values = 'rating').fillna(0)
interactions_matrix['user_index'] = np.arange(0, interactions_matrix.shape[0])
interactions_matrix.set_index(['user_index'], inplace=True)

final_ratings_sparse = csr_matrix(interactions_matrix.values)

# Singular Value Decomposition
U, s, Vt = svds(final_ratings_sparse, k = 50) # here k is the number of latent features

# Construct diagonal array in SVD
sigma = np.diag(s)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Predicted ratings
preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns = interactions_matrix.columns)
preds_matrix = csr_matrix(preds_df.values)

def model_based_recommendations(user_index, num_recommendations):
    # Get the user's ratings from the actual and predicted interaction matrices
    user_ratings = interactions_matrix[user_index, :].toarray().reshape(-1)
    user_predictions = preds_matrix[user_index, :].toarray().reshape(-1)

    # Creating a dataframe with actual and predicted ratings columns
    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['Recommended Recipes'] = np.arange(len(user_ratings))
    temp = temp.set_index('Recommended Recipes')

    # Filtering the dataframe where actual ratings are 0 which implies that the user has not interacted with that product
    temp = temp.loc[temp.user_ratings == 0]

    # Recommending products with top predicted ratings
    temp = temp.sort_values('user_predictions',
                            ascending=False)  # Sort the dataframe by user_predictions in descending order
    print('\nBelow are the recommended recipes for user(user_id = {}):\n'.format(user_index))
    print(temp['user_predictions'].head(num_recommendations))


def calculate_rmse():
    # Calculate average actual ratings
    average_rating = interactions_matrix.mean()

    # Calculate average predicted ratings
    avg_preds = preds_df.mean()

    # Combine average actual and predicted ratings into a DataFrame
    rmse_df = pd.concat([average_rating, avg_preds], axis=1)
    rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']

    # Calculate RMSE
    RMSE = mean_squared_error(rmse_df['Avg_actual_ratings'], rmse_df['Avg_predicted_ratings'], squared=False)

    return RMSE
"""
