"""
URL configuration for ecommerce project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from .views import rank_based_recommendation_view, user_based_recommendations_view
# from .views import model_based_recommendations_view, calculate_rmse_view


urlpatterns = [
    path('rank-based-recommendation/', rank_based_recommendation_view, name='rank_based_recommendation'),
    path('user-based-recommendations/<int:user_index>/', user_based_recommendations_view, name='user_based_recommendations'),
    # path('model-based-recommendations/<int:user_index>/', model_based_recommendations_view, name='model_based_recommendations'),
    # path('calculate-rmse/', calculate_rmse_view, name='calculate_rmse'),
]


