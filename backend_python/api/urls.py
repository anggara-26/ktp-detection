from django.urls import path
from .views import HealthCheckView, IDCardOcrView


urlpatterns = [
    path('healthcheck/', HealthCheckView.as_view(), name='healthcheck'),
    path('ocr/', IDCardOcrView.as_view(), name='ocr'),
]