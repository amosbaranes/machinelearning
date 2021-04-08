from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path(r'', include('machinelearning.apps.main.urls')),
    path(r'core/', include('machinelearning.apps.core.urls')),
    path(r'intro/', include('machinelearning.apps.introml.urls')),
    path(r'dashboard/', include('machinelearning.apps.dashboard.urls')),
    path(r'bi/', include('machinelearning.apps.bi.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



