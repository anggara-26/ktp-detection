from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import ImageUpload, Detections
from .yolo.yolo import YOLODetector
from .serializers import ImageUploadSerializer


class HealthCheckView(APIView):
    def get(self, request):
        return Response({"status": 200, "message": "It's Running :D"}, status=200)

class IDCardOcrView(APIView):
    detector = YOLODetector()

    def post(self, request, format=None):
        print("Request data:", request.data)
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_upload = serializer.save()

            try:
                extracted_data, foto_image, image, list_of_images = self.detector.run_detection(image_path=image_upload.image_file.path, confidence_threshold=image_upload.confidence_threshold)
                image_upload.status = ImageUpload.STATUS_COMPLETED
                image_upload.save()
            except Exception as e:
                image_upload.status = ImageUpload.STATUS_FAILED
                image_upload.save()
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # for detection in detection_result:
            #     Detections.objects.create(
            #         object_detection=image_upload,
            #         label=detection['label'],
            #         confidence=detection['confidence'],
            #         x_min=detection['x_min'],
            #         x_max=detection['x_max'],
            #         y_min=detection['y_min'],
            #         y_max=detection['y_max']
            #     )

            return Response({ "data": {"extracted_data": extracted_data, "foto_image": foto_image, "ktp_image": image}, "code": status.HTTP_200_OK }, status=status.HTTP_200_OK)
        return Response({ "code": status.HTTP_400_BAD_REQUEST, "message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)