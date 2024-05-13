from django.http import JsonResponse
from .recommenders.recommenders import (rank_based_recommendation, user_based_recommendations)
# from .recommenders.recommenders import (model_based_recommendations, calculate_rmse)


def rank_based_recommendation_view(request):
    # Parse request parameters
    num_recipes = int(request.GET.get('num_recipes', 10))
    min_interaction = int(request.GET.get('min_interaction', 5))

    # Call the method from your recommender module
    recommendations = rank_based_recommendation(num_recipes, min_interaction)

    # Return the recommendations as JSON response
    return JsonResponse({'recommendations': recommendations})



def user_based_recommendations_view(request, user_index):
    # Parse request parameters
    num_of_products = int(request.GET.get('num_of_products', 10))

    # Call the method from your recommender module
    recommendations = user_based_recommendations(user_index, num_of_products)

    # Return the recommendations as JSON response
    return JsonResponse({'recommendations': recommendations})



"""
def model_based_recommendations_view(request, user_index):
    # Parse request parameters if needed
    num_recommendations = int(request.GET.get('num_recommendations', 10))

    # Call the method from your recommender module
    recommendations = model_based_recommendations(user_index, num_recommendations)

    # Return the recommendations as JSON response
    return JsonResponse({'recommendations': recommendations})


def calculate_rmse_view(request):
    # Call the method from your recommender module
    rmse = calculate_rmse()

    # Return the RMSE value as JSON response
    return JsonResponse({'RMSE': rmse})
"""