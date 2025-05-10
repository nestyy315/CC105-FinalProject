from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('predictionForm/', views.predictionForm, name='predictionForm'),
    path('predictionResult/', views.predictionResult, name='predictionResult'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout_view, name='logout'),  
     
]

