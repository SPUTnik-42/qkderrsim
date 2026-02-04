from django.urls import path
from webapp import views

urlpatterns = [
    path("", views.home, name="home"),
    path("simulation/", views.run_simulation, name="simulation"),
]