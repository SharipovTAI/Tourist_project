from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('ask-llama/', views.ask_llama),
]