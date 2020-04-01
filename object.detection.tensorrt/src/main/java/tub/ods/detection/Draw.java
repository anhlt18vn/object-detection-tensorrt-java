package tub.ods.detection;

/*
 * Created by Anh Le-Tuan
 * Email: anh.letuan@tu-berlin.de
 * Date: 01.04.20
 */


import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;

import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Draw {

    public static Mat drawBox(Mat mat, Box box, Scalar color) {
        int left = (int) box.getLeft();
        int top = (int) box.getTop();
        int right = (int) box.getRight();
        int bottom = (int) box.getBottom();

        Point leftTop = new Point(left, top);
        Point rightBottom = new Point(right, bottom);

        rectangle(mat, leftTop, rightBottom, color);
        return mat;
    }


    public static void render(Mat mat, Box box, String windowName) {
        mat = drawBox(mat, box, Scalar.GREEN);
        imshow(windowName, mat);
    }

    public static void render(Mat mat, DetectedObjectImage[] detectedObjectImages, String windowName) {
        for (DetectedObjectImage doi : detectedObjectImages) {
            Box box = doi.getBox();
            mat = drawBox(mat, box, Scalar.RED);
            Point textPoint = new Point((int) box.getLeft(), (int) box.getTop());
            String conf = String.format("%.2f", doi.getConfidence());
            String label = doi.getLabel();
            putText(mat, label + " " + conf, textPoint, FONT_HERSHEY_DUPLEX, 1, Scalar.RED);
        }
        imshow(windowName, mat);
    }
}
