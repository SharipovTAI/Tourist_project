from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('ask-llama/', views.ask_llama, name='ask_llama'),
    path('debug_search/', views.debug_search, name='debug_search'),
]
