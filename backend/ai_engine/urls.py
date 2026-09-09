from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('health', views.health_check, name='health_check'),
    path('health/', views.health_check, name='health_check_slash'),
    path('gemini/analyze', views.analyze_telemetry, name='analyze_telemetry'),
    path('gemini/analyze/', views.analyze_telemetry, name='analyze_telemetry_slash'),
    path('gemini/chat', views.chat_telemetry, name='chat_telemetry'),
    path('gemini/chat/', views.chat_telemetry, name='chat_telemetry_slash'),
    path('gemini/vision', views.vision_telemetry, name='vision_telemetry'),
    path('gemini/vision/', views.vision_telemetry, name='vision_telemetry_slash'),
    path('ai/semantic-search', views.semantic_search, name='semantic_search'),
    path('ai/semantic-search/', views.semantic_search, name='semantic_search_slash'),
]
