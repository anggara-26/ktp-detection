from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class ImageUpload(models.Model):
    STATUS_PENDING = 'pending'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    image_file = models.ImageField(upload_to='media/images/')
    upload_timestamp = models.DateTimeField(auto_now_add=True)
    confidence_threshold = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    status = models.CharField(
        max_length=50,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )


class Detections(models.Model):
    object_detection = models.ForeignKey(ImageUpload, on_delete=models.CASCADE, related_name='detections')
    label = models.CharField(max_length=100)
    confidence = models.FloatField()
    x_min = models.FloatField()
    x_max = models.FloatField()
    y_min = models.FloatField()
    y_max = models.FloatField()
    detection_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} ({self.confidence})"

class Countries(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name
    
class Provinces(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True)
    country = models.ForeignKey(Countries, on_delete=models.CASCADE, related_name='provinces')

    def __str__(self):
        return self.name
    
class Cities(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True)
    province = models.ForeignKey(Provinces, on_delete=models.CASCADE, related_name='cities')

    def __str__(self):
        return self.name

class Districts(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True)
    city = models.ForeignKey(Cities, on_delete=models.CASCADE, related_name='districts')

    def __str__(self):
        return self.name
    
class SubDistricts(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)
    code = models.CharField(max_length=20, unique=True)
    district = models.ForeignKey(Districts, on_delete=models.CASCADE, related_name='subdistricts')

    def __str__(self):
        return self.name