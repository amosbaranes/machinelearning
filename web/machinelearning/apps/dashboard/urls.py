from django.urls import path
from .views import dashboard_with_pivot, pivot_data

app_name = "dashboard"

urlpatterns = [
    path('', dashboard_with_pivot, name='dashboard_with_pivot'),
    path('data', pivot_data, name='pivot_data'),
]