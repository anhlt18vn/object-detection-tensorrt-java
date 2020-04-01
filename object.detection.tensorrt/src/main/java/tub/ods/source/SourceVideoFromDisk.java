package tub.ods.source;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class SourceVideoFromDisk implements Source<Mat> {

    Logger logger = LoggerFactory.getLogger(SourceVideoFromDisk.class);

    private Frame frame;
    private FFmpegFrameGrabber grabber;
    private boolean hasNext = true;
    private boolean checkNext;
    private OpenCVFrameConverter.ToMat toMat;

    public SourceVideoFromDisk(String path2VideoFile) throws FrameGrabber.Exception {
        File file = new File(path2VideoFile);
        this.grabber = new FFmpegFrameGrabber(file);
        grabber.start();
        toMat = new OpenCVFrameConverter.ToMat();
    }

    @Override
    public Mat source() {
        if (checkNext && hasNext) {
            this.checkNext = false;
            Mat mat = toMat.convertToMat(this.frame);
            return mat;
        }
        throw new IllegalStateException("FrameGrabberIterator has ended");
    }


    @Override
    public boolean end() {
        return !hasNext();
    }

    private void checkNext() {
        try {
            this.frame = this.grabber.grabImage();
            this.checkNext = true;
        } catch (FrameGrabber.Exception e) {
            this.checkNext = true;
            this.hasNext = false;
        }
    }

    public boolean hasNext() {
        if (!checkNext) checkNext();
        this.hasNext = checkNext && this.frame != null;
        return hasNext;
    }
}
