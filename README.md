#object detection tensorrt java

Thanks to javacpp projects that build the bridge between jvm and native C/C++ libraries.
TensorRT, OpenCV are available for Java developers.
This project is an example that builds object detection application on Java using javacpp-preset tensorrt.

It is required to install the following libraries:
   - CUDA 10.1
   - TensorRT 6.0.1.5

The demo use pre-trained model yolov3 in tensorRT format.
This is created from the sample  



To run the demo:
   - arg[0] path to yolov3.trt file.
   - arg[1] path to the video file (e.g video.mp4)
   
   