package tub.ods.detection;

/*
 * Created by Anh Le-Tuan
 * Email: anh.letuan@tu-berlin.de
 * Date: 01.04.20
 */


import javafx.util.Pair;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;


public class DetectorYolov3 extends DetectorTensorRT {

    protected static final int boxesIdx = 0;
    protected static final int classesIdx = 1;
    protected static final int confidencesIdx = 2;

    private static String _000_NET = "000_net";
    private static String CONV_082 = "082_convolutional";
    private static String CONV_094 = "094_convolutional";
    private static String CONV_106 = "106_convolutional";
    private final int[][] outputShapes = {{1, 255, 19, 19}, {1, 255, 38, 38}, {1, 255, 76, 76}};
    private final int[][] mMasks = {{6, 7, 8}, {3, 4, 5}, {0, 1, 2}};
    private final int[][] mAnchors = {{10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45}, {59, 119}, {116, 90}, {156, 198}, {373, 326}};
    private final int numberClasses = 80;

    private static int h = 608;
    private static int w = 608;

    private int orgH, orgW;

    private static int ch = 3;


    private final String[] classesLabels = new String[]{
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
            "toilet", "tvmonitor", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster",
            "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
    };
    protected float objThreshold = 0.5f;
    protected float nmsThreshold = 0.4f;


    public DetectorYolov3(String path2TRTFile) {
        super(path2TRTFile);
    }

    @Override
    protected Map<String, Pointer> processInput(Mat input) {
        input = input.clone();
        orgH = input.rows();
        orgW = input.cols();
        resize(input, input, new Size(h, w));

        ByteBuffer byteBuffer = input.createBuffer();
        FloatBuffer floatBuffer = FloatBuffer.allocate(byteBuffer.capacity());

        for (int c = 0; c < ch; c++) {
            for (int wh = 0; wh < w * h; wh++) {
                int idxOutput = c * w * h + wh;
                int idxInput = wh * ch + c;
                byte b = byteBuffer.get(idxInput);
                float f = (float) (b & 0xFF) / 255;
                floatBuffer.put(idxOutput, f);
            }
        }

        FloatPointer pointer = new FloatPointer(floatBuffer);

        Map<String, Pointer> map = new HashMap<>();
        map.put(_000_NET, pointer);
        return map;
    }


    @Override
    protected DetectedObjectImage[] processOutput(Map<String, Pointer> output) {

        float[] conv_082 = toFloatArray(output.get(CONV_082).asByteBuffer().asFloatBuffer());
        float[] conv_094 = toFloatArray(output.get(CONV_094).asByteBuffer().asFloatBuffer());
        float[] conv_106 = toFloatArray(output.get(CONV_106).asByteBuffer().asFloatBuffer());

        INDArray[] tensorsINDArrays = new INDArray[3];
        tensorsINDArrays[0] = reshapeOutputTensor(Nd4j.create(conv_082, outputShapes[0]), this.numberClasses);
        tensorsINDArrays[1] = reshapeOutputTensor(Nd4j.create(conv_094, outputShapes[1]), this.numberClasses);
        tensorsINDArrays[2] = reshapeOutputTensor(Nd4j.create(conv_106, outputShapes[2]), this.numberClasses);

        ArrayList<INDArray> listBoxesINDArrays = new ArrayList<>();
        ArrayList<INDArray> listClassesINDArrays = new ArrayList<>();
        ArrayList<INDArray> listConfidencesINDArrays = new ArrayList<>();


        for (int i = 0; i < tensorsINDArrays.length; i++) {

            INDArray[] detectedBoxesClassAndConfidence = filterBoxes(processFeats(tensorsINDArrays[i], this.mMasks[i]), this.objThreshold);

            if (detectedBoxesClassAndConfidence != null) {
                listBoxesINDArrays.add(detectedBoxesClassAndConfidence[0]);
                listClassesINDArrays.add(detectedBoxesClassAndConfidence[1]);
                listConfidencesINDArrays.add(detectedBoxesClassAndConfidence[2]);
            }
        }

        INDArray[] boxesINDArray = new INDArray[listBoxesINDArrays.size()];
        INDArray[] classesINDArray = new INDArray[listClassesINDArrays.size()];
        INDArray[] confidencesINDArray = new INDArray[listConfidencesINDArrays.size()];

        listBoxesINDArrays.toArray(boxesINDArray);
        listClassesINDArrays.toArray(classesINDArray);
        listConfidencesINDArrays.toArray(confidencesINDArray);

        if (listBoxesINDArrays.size() == 0 || listClassesINDArrays.size() == 0 || listConfidencesINDArrays.size() == 0) {
            System.out.println("boxes: " + listBoxesINDArrays.size());
            System.out.println("classes: " + listClassesINDArrays.size());
            System.out.println("conf: " + listConfidencesINDArrays.size());
            return new DetectedObjectImage[0];
        }

        INDArray[] indArrays = new INDArray[3];
        indArrays[boxesIdx] = Nd4j.concat(0, boxesINDArray);
        indArrays[classesIdx] = Nd4j.concat(0, classesINDArray);
        indArrays[confidencesIdx] = Nd4j.concat(0, confidencesINDArray);

        float[][] boxes = indArrays[boxesIdx].toFloatMatrix();
        float[] classesFinal = indArrays[classesIdx].toFloatVector();
        float[] confidencesFinal = indArrays[confidencesIdx].toFloatVector();

        Box[] boxesFinal = new Box[boxes.length];

        for (int idx = 0; idx < boxesFinal.length; idx++) {
            Box box = new Box();
            float left = boxes[idx][0] * orgW;
            float top = boxes[idx][1] * orgH;
            float width = boxes[idx][2] * orgW;
            float height = boxes[idx][3] * orgH;
            box.setBoxLeftTopWidthHeight(left, top, width, height);
            boxesFinal[idx] = box;
        }

        return processFinalResult(boxesFinal, classesFinal, confidencesFinal, this.nmsThreshold);
    }

    private INDArray[] filterBoxes(INDArray[] bcp, float objThreshold) {
        INDArray boxes = bcp[0];
        INDArray confidences = bcp[1];
        INDArray probs = bcp[2];

        INDArray scores = confidences.mul(probs);
        INDArray classes = Nd4j.argMax(scores, -1);
        INDArray classScores = Nd4j.max(scores, -1);

        INDArray mask = classScores.match(1, Conditions.greaterThanOrEqual(objThreshold));

        INDArray[] pos = Nd4j.where(mask, null, null);

        if (pos.length == 0) {
            return null;
        }

        if (pos[0].length() == 0) {
            return null;
        }

        INDArray[] boxesInd = new INDArray[(int) pos[0].length()];
        INDArray[] classInd = new INDArray[(int) pos[0].length()];
        INDArray[] scoreInd = new INDArray[(int) pos[0].length()];

        for (int i = 0; i < pos[0].size(0); i++) {
            int d1 = (int) pos[0].getFloat(i);
            int d2 = (int) pos[1].getFloat(i);
            int d3 = (int) pos[2].getFloat(i);

            INDArray box = boxes.get(NDArrayIndex.point(d1), NDArrayIndex.point(d2), NDArrayIndex.point(d3), NDArrayIndex.all());
            box = box.reshape(new long[]{1, box.size(0)});
            boxesInd[i] = box;

            INDArray clazz = classes.get(NDArrayIndex.point(d1), NDArrayIndex.point(d2), NDArrayIndex.point(d3));
            classInd[i] = clazz;

            INDArray score = classScores.get(NDArrayIndex.point(d1), NDArrayIndex.point(d2), NDArrayIndex.point(d3));
            scoreInd[i] = score;
        }

        INDArray boxesResult = Nd4j.concat(0, boxesInd);
        INDArray clazzResult = Nd4j.concat(0, classInd);
        INDArray scoreResult = Nd4j.concat(0, scoreInd);

        return new INDArray[]{boxesResult, clazzResult, scoreResult};
    }

    private INDArray reshapeOutputTensor(INDArray output, int num_classes) {
        output = output.permute(0, 2, 3, 1);
        long[] shape = output.shape();

        int height = (int) shape[1];
        int width = (int) shape[2];

        int dim1 = height;
        int dim2 = width;
        int dim3 = 3;
        int dim4 = (4 + 1 + num_classes);

        output = output.reshape(new int[]{dim1, dim2, dim3, dim4});
        return output;
    }

    private INDArray[] processFeats(INDArray outputTensor, int[] mask) {
        long gridH = outputTensor.shape()[0];
        long gridW = outputTensor.shape()[1];

        float[] anchors = new float[mask.length * 2];
        for (int i = 0; i < mask.length; i++) {
            anchors[i * 2] = this.mAnchors[mask[i]][0];
            anchors[i * 2 + 1] = this.mAnchors[mask[i]][1];
        }
        INDArray anchorTensor = Nd4j.create(anchors, new int[]{1, 1, mask.length, 2});

        INDArray boxXY = outputTensor.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 2));
        boxXY = Nd4j.getExecutioner().exec(new Sigmoid(boxXY));

        INDArray boxWH = outputTensor.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(2, 4));
        boxWH = Nd4j.getExecutioner().exec(new Exp(boxWH)).mul(anchorTensor);

        INDArray confidence = outputTensor.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(4));
        confidence = Nd4j.getExecutioner().exec(new Sigmoid(confidence));
        confidence = Nd4j.expandDims(confidence, 3);

        INDArray probs = outputTensor.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(5, 85));
        probs = Nd4j.getExecutioner().exec(new Sigmoid(probs));

        INDArray col = Nd4j.arange(0, gridW);
        col = Nd4j.tile(col, (int) gridW);
        col = col.reshape(new long[]{-1, gridW});
        INDArray row = col.transpose();

        col = col.reshape(gridH, gridW, 1, 1).repeat(-2, 3);
        row = row.reshape(gridH, gridW, 1, 1).repeat(-2, 3);

        INDArray grid = Nd4j.concat(-1, col, row);

        boxXY = boxXY.add(grid);
        boxXY = boxXY.div(Nd4j.create(new float[]{gridW, gridH}));

        boxWH = boxWH.div(Nd4j.create(new float[]{w, h}));
        boxXY = boxXY.sub(boxWH.div(2));

        INDArray boxes = Nd4j.concat(-1, boxXY, boxWH);

        return new INDArray[]{boxes, confidence, probs};
    }

    protected DetectedObjectImage constructDetectedObject(String label, float confidence, Box box) {
        DetectedObjectImage detectedObject = new DetectedObjectImage(label, confidence, box);
        return detectedObject;
    }

    protected DetectedObjectImage[] processFinalResult(Box[] boxes, float[] classes, float[] confidences, float nmsThreshold) {
        ArrayList<Integer> keepList = filterOverlappedBoxes(boxes, confidences, nmsThreshold);
        ArrayList<DetectedObjectImage> detectedObjects = new ArrayList<>();

        for (int keepIdx : keepList) {
            float conf = confidences[keepIdx];
            String label = classesLabels[((int) classes[keepIdx])];
            DetectedObjectImage detectedObject = constructDetectedObject(label, conf, boxes[keepIdx]);
            detectedObjects.add(detectedObject);
        }

        DetectedObjectImage[] dois = new DetectedObjectImage[detectedObjects.size()];
        detectedObjects.toArray(dois);

        return dois;
    }

    private ArrayList<Integer> filterOverlappedBoxes(Box[] boxes, float[] confidences, float mnsThreshold) {

        ArrayList<Pair<Float, Integer>> scoreIndex = new ArrayList<>();
        for (int i = 0; i < confidences.length; i++) {
            scoreIndex.add(new Pair<>(confidences[i], i));
        }
        scoreIndex.sort(Comparator.comparing(Pair::getKey));
        return nmsBoxes(scoreIndex, boxes, mnsThreshold);
    }

    private ArrayList<Integer> nmsBoxes(ArrayList<Pair<Float, Integer>> scoreIndex, Box[] boxes, float nmsThreshold) {
        ArrayList<Integer> keepList = new ArrayList<>();
        for (Pair<Float, Integer> si : scoreIndex) {
            int idx = si.getValue();
            boolean kept = true;
            for (int keptIdx : keepList) {
                if (kept) {
                    float overlapped = iou(boxes[idx], boxes[keptIdx]);
                    kept = overlapped < nmsThreshold;
                } else {
                    break;
                }
            }
            if (kept) {
                keepList.add(idx);
            }
        }
        return keepList;
    }

    private float iou(Box b1, Box b2) {
        float area1 = b1.getWidth() * b1.getHeight();
        float area2 = b2.getWidth() * b2.getHeight();

        float xx1 = Math.max(b1.getLeft(), b2.getLeft());
        float yy1 = Math.max(b1.getTop(), b2.getTop());
        float xx2 = Math.min(b1.getLeft() + b1.getWidth(), b2.getLeft() + b2.getWidth());
        float yy2 = Math.min(b1.getTop() + b1.getHeight(), b2.getTop() + b2.getHeight());

        float width = Math.max(0.0f, xx2 - xx1 + 1);
        float height = Math.max(0.0f, yy2 - yy1 + 1);

        float intersection = width * height;
        float union = area1 + area2 - intersection;
        return intersection / union;
    }

    public static float[] toFloatArray(FloatBuffer buffer) {
        if (buffer.hasArray()) {
            return buffer.array();
        } else {
            float[] array = new float[buffer.capacity()];
            for (int i = 0; i < buffer.capacity(); i++) {
                array[i] = buffer.get(i);
            }
            return array;
        }
    }
}
