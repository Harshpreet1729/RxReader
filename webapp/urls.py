from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("prescription/<str:identifier>/", views.detail, name="detail"),
    path("prescription/<str:identifier>/status/", views.status, name="status"),
    path("prescription/<str:identifier>/image/", views.picture, name="picture"),
    path(
        "prescription/<str:identifier>/crop/<str:region_id>/", views.crop, name="crop"
    ),
    path("prescription/<str:identifier>/review/", views.review, name="review"),
    path(
        "prescription/<str:identifier>/read-empty/", views.read_empty, name="read_empty"
    ),
    path("prescription/<str:identifier>/add/", views.add_region, name="add_region"),
    path(
        "prescription/<str:identifier>/read/<str:region_id>/",
        views.read_region,
        name="read_region",
    ),
    path(
        "prescription/<str:identifier>/remove/<str:region_id>/",
        views.remove_region,
        name="remove_region",
    ),
    path("prescription/<str:identifier>/delete/", views.delete, name="delete"),
    path("prescription/<str:identifier>/export/", views.export, name="export"),
    path(
        "prescription/<str:identifier>/lookup/<str:region_id>/",
        views.lookup,
        name="lookup",
    ),
]
