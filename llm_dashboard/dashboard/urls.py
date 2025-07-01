from django.urls import path
from .views import dashboard_view, upload_csv, chatbot_view, upload_dataset

urlpatterns = [
    path("", dashboard_view, name="dashboard"),
    path("upload-csv/", upload_csv, name="upload_csv"),
    path('chatbot/', chatbot_view, name='chatbot'),
    path('upload-dataset/', upload_dataset),
]
