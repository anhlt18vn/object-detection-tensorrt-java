package tub.ods.detection;

/*
 * Created by Anh Le-Tuan
 * Email: anh.letuan@tu-berlin.de
 * Date: 01.04.20
 */


import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.tensorrt.global.nvinfer;
import tub.ods.tensorrt.TensorRTEngine;

import java.util.Map;

public abstract class DetectorTensorRT {
    protected TensorRTEngine engine;


    protected DetectorTensorRT(String path2TRTFile) {
        Loader.load(nvinfer.class);
        this.engine = new TensorRTEngine(path2TRTFile);
    }

    public DetectedObjectImage[] detect(Mat in) {
        Map<String, Pointer> input = processInput(in);
        Map<String, Pointer> output = engine.infer(input, 1);
        return processOutput(output);
    }

    protected abstract Map<String, Pointer> processInput(Mat mat);

    protected abstract DetectedObjectImage[] processOutput(Map<String, Pointer> output);

    public void shutdown() {
        engine.shutdown();
    }
}
