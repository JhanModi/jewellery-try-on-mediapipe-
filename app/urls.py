from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('necklace/', views.necklace_tryon, name='necklace_tryon'),
    path('validate_face_for_necklace/', views.validate_face_for_necklace, name='validate_face_for_necklace'),
    path('apply_necklace/', views.apply_necklace, name='apply_necklace'),
    path('rings/', views.ring_tryon, name='ring_tryon'),
    path('earrings/', views.earrings, name='earrings'),
    path('validate_face/', views.validate_face, name='validate_face'),
    path('apply_earring/', views.apply_earring, name='apply_earring'),
    path('validate_hand/', views.validate_hand, name='validate_hand'),
    path('apply_ring/', views.apply_ring, name='apply_ring'),
]