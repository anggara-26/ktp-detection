# Backend OCR KTP - Django API

Indonesian ID Card (KTP) OCR Backend using Django REST Framework, YOLO object detection, and TrOCR text recognition.

## Features

- **ID Card Detection**: Automatically detects and segments Indonesian ID cards from images
- **Text Extraction**: Uses TrOCR (Transformer-based OCR) to extract text from detected regions
- **Multi-field Recognition**: Extracts multiple fields including:
  - NIK (ID Number)
  - Nama (Name)
  - TTL (Place and Date of Birth)
  - Jenis Kelamin (Gender)
  - Alamat (Address)
  - RT/RW
  - Kelurahan/Desa (Village)
  - Kecamatan (District)
  - Agama (Religion)
  - Status Perkawinan (Marital Status)
  - Pekerjaan (Occupation)
  - Kewarganegaraan (Nationality)
  - Foto (Photo)
- **REST API**: Django REST Framework for easy integration
- **CORS Support**: Cross-origin requests enabled for frontend integration
- **Admin Interface**: Django admin for managing data

## Tech Stack

- **Framework**: Django 5.2 + Django REST Framework
- **Computer Vision**: YOLOv8 (Ultralytics) for object detection
- **OCR**: Microsoft TrOCR for text recognition
- **Image Processing**: OpenCV, PIL
- **Database**: SQLite (default, configurable)
- **Other Libraries**:
  - EasyOCR (alternative OCR)
  - Matplotlib for visualization
  - python-Levenshtein for text similarity

## Project Structure

```
backend_python/
├── api/                          # Main API app
│   ├── yolo/                     # YOLO detection module
│   │   ├── yolo.py              # Main detection logic
│   │   └── weights/             # Model weights
│   ├── models.py                # Database models
│   ├── views.py                 # API endpoints
│   ├── serializers.py           # DRF serializers
│   ├── urls.py                  # API URL routing
│   └── admin.py                 # Admin configuration
├── backend_python/              # Django project settings
│   ├── settings.py              # Main settings
│   ├── urls.py                  # Root URL configuration
│   └── wsgi.py                  # WSGI configuration
├── media/                       # Uploaded images storage
└── db.sqlite3                   # SQLite database
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Backend-OCR-KTP
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv env

   # On Windows
   .\env\Scripts\activate

   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO weights**

   - Place your trained YOLO model weights in `backend_python/api/yolo/weights/`
   - Required files:
     - `best.pt` (attributes detection model)
     - `segment_model.pt` (card segmentation model)

5. **Database setup**

   ```bash
   cd backend_python
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser (optional)**

   ```bash
   python manage.py createsuperuser
   ```

7. **Run the development server**
   ```bash
   python manage.py runserver
   ```

The API will be available at `http://localhost:8000/`

## API Endpoints

### Health Check

```
GET /api/health
```

Returns server status.

### OCR Processing

```
POST /api/ocr/
```

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `image_file`: Image file (jpg, png, etc.)
  - `confidence_threshold`: Float (0.0-1.0, default: 0.5)

**Response:**

```json
{
  "extracted_data": {
    "nik": "1234567890123456",
    "nama": "JOHN DOE",
    "ttl": "JAKARTA, 01-01-1990",
    "jeniskelamin": "LAKI-LAKI",
    "alamat": "JL. CONTOH NO. 123",
    "rtrw": "001/002",
    "keldesa": "KELURAHAN CONTOH",
    "kecamatan": "KECAMATAN CONTOH",
    "agama": "ISLAM",
    "statuskawin": "KAWIN",
    "pekerjaan": "PEGAWAI SWASTA",
    "kewarganegaraan": "WNI"
  },
  "foto_image": "base64_encoded_photo",
  "ktp_image": "base64_encoded_result_visualization",
  "list_of_images": ["base64_image1", "base64_image2", ...]
}
```

## Configuration

### Settings

Key settings in `backend_python/settings.py`:

```python
# CORS Configuration
CORS_ALLOW_ALL_ORIGINS = True  # For development only

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# REST Framework
REST_FRAMEWORK = {
    'DEFAULT_PARSER_CLASSES': (
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.MultiPartParser',
    ),
}
```

### Model Configuration

Adjust model parameters in `api/yolo/yolo.py`:

```python
class YOLODetector:
    def __init__(self,
                 attributes_model_path="path/to/best.pt",
                 card_model_path="path/to/segment_model.pt"):
        # Model initialization
```

## Development

### Adding New Fields

1. Update the detection labels in `yolo.py`
2. Modify the response structure in `views.py`
3. Update the frontend to handle new fields

### Database Models

The project includes models for:

- `ImageUpload`: Stores uploaded images and processing status
- `Detections`: Stores detection results
- Geographic models: `Countries`, `Provinces`, `Cities`, `Districts`, `SubDistricts`

### Admin Interface

Access the Django admin at `http://localhost:8000/admin/` to:

- View uploaded images
- Check detection results
- Manage geographic data

## Deployment

### Production Settings

1. **Environment Variables**

   ```bash
   export DEBUG=False
   export SECRET_KEY='your-secret-key'
   export ALLOWED_HOSTS='your-domain.com'
   ```

2. **Database**

   - Configure PostgreSQL or MySQL for production
   - Update `DATABASES` setting in `settings.py`

3. **Static Files**

   ```bash
   python manage.py collectstatic
   ```

4. **CORS**
   - Update `CORS_ALLOWED_ORIGINS` with specific domains
   - Remove `CORS_ALLOW_ALL_ORIGINS = True`

### Docker (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

## Troubleshooting

### Common Issues

1. **CORS Errors**

   - Ensure `corsheaders` is installed and configured
   - Check `CORS_ALLOWED_ORIGINS` settings

2. **Model Loading Errors**

   - Verify YOLO weight files are in correct location
   - Check file permissions

3. **Memory Issues**

   - Large images may cause memory issues
   - Consider resizing images before processing

4. **TrOCR Performance**
   - First run downloads model weights (may be slow)
   - Consider using GPU for faster inference

### Performance Optimization

1. **Image Preprocessing**

   - Resize images to optimal size
   - Apply image enhancement techniques

2. **Batch Processing**

   - Process multiple text regions in batches
   - Implement caching for repeated requests

3. **Model Optimization**
   - Use quantized models for faster inference
   - Consider using ONNX runtime

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:

- Create an issue in the repository
- Contact the development team

---

**Note**: This is a development setup. For production deployment, ensure proper security configurations, environment variables, and production-grade database setup.
