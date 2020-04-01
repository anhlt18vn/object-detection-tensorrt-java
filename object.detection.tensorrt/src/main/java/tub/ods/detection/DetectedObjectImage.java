package tub.ods.detection;

/*
 * Created by Anh Le-Tuan
 * Email: anh.letuan@tu-berlin.de
 * Date: 01.04.20
 */


import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class DetectedObjectImage {
    private Box box;
    private String label;
    private float confidence;

    public DetectedObjectImage() {
    }

    public DetectedObjectImage(String label, float confidence, Box box) {
        this.box = box;
        this.label = label;
        this.confidence = confidence;
    }

    public Box getBox() {
        return box;
    }

    public void setBox(Box box) {
        this.box = box;
    }


    @Override
    public String toString() {
        String s = "{}";

        ObjectMapper objectMapper = new ObjectMapper();

        try {
            s = objectMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }

        return s;
    }

    public float getConfidence() {
        return confidence;
    }

    public void setConfidence(float confidence) {
        this.confidence = confidence;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }
}
