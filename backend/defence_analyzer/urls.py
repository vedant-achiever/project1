from django.urls import path, include
from ai_engine import views as ai_views

urlpatterns = [
    path('', ai_views.index_view, name='root_index'),
    path('api/', include('ai_engine.urls')),
]
