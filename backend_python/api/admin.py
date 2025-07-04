from django.contrib import admin
from .models import ImageUpload, Detections, Countries, Provinces, Cities, Districts, SubDistricts

admin.site.register(ImageUpload)
admin.site.register(Detections)
admin.site.register(Countries)
admin.site.register(Provinces)
admin.site.register(Cities)
admin.site.register(Districts)
admin.site.register(SubDistricts)

