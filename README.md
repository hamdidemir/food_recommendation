# Food Recipe Recommendation System

The Food Recipe Recommendation System is a machine learning-based project that offers personalized recipe recommendations to users based on their browsing and purchase history. The system leverages rank-based filtering and user-based collaborative filtering algorithms to analyze user behavior and generate relevant suggestions. It's implemented as a RESTful API using Django, with Nginx for load balancing to enhance application performance.

## Dataset

The dataset used for this project comprises recipes and user interactions from food.com. You can access the dataset [here](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

- **Exploratory Data Analysis**: Initial data exploration is conducted in Jupyter Notebook, and the findings are translated into Python code for the RESTful API.

## Rank Based Product Recommendation (Shortened)

### Objective

- Recommend products with the highest number of ratings.
- Target new customers with the most popular products.
- Solve the Cold Start Problem.

### Outputs

- Recommend the top 5 products with a minimum of 50/100 ratings/interactions.

### Approach

1. Calculate the average rating for each product.
2. Determine the total number of ratings for each product.
3. Create a DataFrame using these values and sort it by average.
4. Develop a function to retrieve the top 'n' products with a specified minimum number of interactions.

## User-based Collaborative Filtering (Shortened)

### Objective

- Provide personalized and relevant recommendations to users.

### Outputs

- Recommend the top 5 products based on interactions of similar users.

### Approach

1. Convert user_id to integers for convenience.
2. Find similar users using cosine similarity.
3. Recommend products based on interactions of similar users.

## RESTful API

The API offers two endpoints for recommendation: user-based and rank-based. 

- **User-based recommendation**: Users can provide a user id and an optional number of products (default is 10).
- **Rank-based recommendation**: Users can specify the number of products and interactions, primarily for new users (Cold Start Problem).

## Load Balancing

Nginx is incorporated into the project for performance optimization. The default ports are 8000 and 9000. Developers can build the project using these ports and test multiporting.

## Model-based Recommendation and Evaluation

The project includes a commented-out model-based recommendation system and error checking using Root Mean Square Error (RMSE). However, due to resource constraints, the csr_matrix method from the scipy.sparse library with this dataset resulted in memory errors in my computer however they are working in the jupiter notebook in kaggle servers. These sections are temporarily disabled and commented out.
