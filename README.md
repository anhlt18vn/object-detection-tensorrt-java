#object detection tensorrt java

Thanks to javacpp projects that build the bridge between jvm and native C/C++ libraries.
TensorRT, OpenCV are available for Java developers.
This project is an example that builds object detection application on Java using javacpp-preset tensorrt.

Install 
   - CUDA 10.1
   - TensorRT 6.0.1.5

To create pre-trained model yolov3 in tensort, please follow the guide from official TensorRT:
[Object Detection With The ONNX TensorRT Backend In Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#yolov3_onnx)     

To run the demo:
   - arg[0] path to the generated yolov3.trt file.
   - arg[1] path to the video file (e.g video.mp4)
   
[Demo](https://www.youtube.com/watch?v=RGnpiJWMPN8)
