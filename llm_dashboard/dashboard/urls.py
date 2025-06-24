from django.urls import path
from .views import dashboard_view, upload_csv

urlpatterns = [
    path("", dashboard_view, name="dashboard"),
    path("upload-csv/", upload_csv, name="upload_csv"),
]
