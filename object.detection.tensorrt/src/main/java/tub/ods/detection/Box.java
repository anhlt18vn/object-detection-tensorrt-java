package tub.ods.detection;

/*
 * Created by Anh Le-Tuan
 * Email: anh.letuan@tu-berlin.de
 * Date: 01.04.20
 */

public class Box {
    private float left;
    private float top;
    private float width;
    private float height;
    private float right;
    private float bottom;
    private int boxId;

    public Box() {
    }

    public void setBoxTopLeftBottomRight(float top, float left, float bottom, float right) {
        this.top = top;
        this.left = left;
        this.width = right - left;
        this.height = bottom - top;
        this.bottom = bottom;
        this.right = right;
    }

    public void setBoxLeftTopWidthHeight(float left, float top, float width, float height) {
        this.top = top;
        this.left = left;
        this.width = width;
        this.height = height;
        this.right = left + width;
        this.bottom = top + height;
    }

    public void setXYAH(float centerX, float centerY, float a, float height) {
        this.height = height;
        this.width = a * height;
        this.top = centerY - height / 2;
        this.bottom = centerY + height / 2;
        this.left = centerX - this.width / 2;
        this.right = centerX + this.width / 2;
    }

    public float getLeft() {
        return left;
    }

    public void setLeft(float left) {
        this.left = left;
    }

    public float getTop() {
        return top;
    }

    public void setTop(float top) {
        this.top = top;
    }

    public float getWidth() {
        return width;
    }

    public void setWidth(float width) {
        this.width = width;
    }

    public float getHeight() {
        return height;
    }

    public void setHeight(float height) {
        this.height = height;
    }

    public float getBottom() {
        return bottom;
    }

    public void setBottom(float bottom) {
        this.bottom = bottom;
    }

    public float getRight() {
        return right;
    }

    public void setRight(float right) {
        this.right = right;
    }

    public float[] getTLBR() {
        return new float[]{top, left, bottom, right};
    }

    public float[] getTLWH() {
        return new float[]{top, left, width, height};
    }

    public float[] getXYAH() {
        float X = left + width / 2;
        float Y = top + height / 2;
        float AH = width / height;
        return new float[]{X, Y, AH, height};
    }

    @Override
    public String toString() {
        return "[box: " + left + " " + top + " " + right + " " + bottom + " ]";
    }

    public int getBoxId() {
        return boxId;
    }

    public void setBoxId(int boxId) {
        this.boxId = boxId;
    }
}