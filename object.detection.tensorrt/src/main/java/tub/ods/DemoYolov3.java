package tub.ods;

/*
 * Created by Anh Le-Tuan
 * Email: anh.letuan@tu-berlin.de
 * Date: 01.04.20
 */


import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import tub.ods.detection.DetectedObjectImage;
import tub.ods.detection.DetectorTensorRT;
import tub.ods.detection.DetectorYolov3;
import tub.ods.source.Source;
import tub.ods.source.SourceVideoFromDisk;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static tub.ods.detection.Draw.render;


public class DemoYolov3 {

    public static void main(String[] args) {
        String path2yolov3 = args[0];
        String path2VideoFile = args[1];

        DetectorTensorRT detectorYolov3 = new DetectorYolov3(path2yolov3);

        try {

            Source<Mat> source = new SourceVideoFromDisk(path2VideoFile);

            while (!source.end()) {
                Mat mat = source.source();
                DetectedObjectImage[] dois = detectorYolov3.detect(mat);

                namedWindow("detected", WINDOW_NORMAL );
                render(mat, dois, "detected");
                int key = waitKey(10);
                if (key == 'q') {
                    System.exit(0);
                }
            }
            detectorYolov3.shutdown();

        } catch (FrameGrabber.Exception e) {
            e.printStackTrace();
        }
    }
}
