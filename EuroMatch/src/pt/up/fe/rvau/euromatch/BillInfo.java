/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.features2d.KeyPoint;
import pt.up.fe.rvau.euromatch.jutils.MatConverter;

/**
 *
 * @author luiscubal
 */
public class BillInfo {
	private int value;
	private Mat billImage;
	private MatOfKeyPoint keyPoints;
	private Mat descriptor;
	
	public BillInfo(Mat billImage, int value) {
		this.billImage = billImage;
		this.value = value;
	}
	
	public int getValue() {
		return value;
	}
	
	public Mat getBillImage() {
		return billImage;
	}
	
	public int getImageWidth() {
		return billImage.cols();
	}
	
	public int getImageHeight() {
		return billImage.rows();
	}
	
	public void setKeyPoints(MatOfKeyPoint keyPoints, int[] relevantParts) {
		List<KeyPoint> keyPointsList = keyPoints.toList();
		List<KeyPoint> validKeyPoints = new ArrayList<>();
		for (KeyPoint keyPoint : keyPointsList) {
			Point point = keyPoint.pt;
			if (isInRelevantRegion(point, relevantParts)) {
				validKeyPoints.add(keyPoint);
			}
		}
		
		this.keyPoints = MatConverter.convertToMatOfKeyPoint(validKeyPoints);
	}
	
	public MatOfKeyPoint getKeyPoints() {
		return keyPoints;
	}
	
	public List<KeyPoint> getKeyPointsAsList() {
		return keyPoints.toList();
	}
	
	public void setDescriptor(Mat descriptor) {
		this.descriptor = descriptor;
	}
	
	public Mat getDescriptor() {
		return descriptor;
	}

	private boolean isInRelevantRegion(Point point, int[] relevantParts) {
		for (int i = 0; i < relevantParts.length; i += 4) {
			if (point.x >= relevantParts[i] && point.y >= relevantParts[i + 1]
					&& point.x < relevantParts[i + 2] && point.y < relevantParts[i + 3]) {
				return true;
			}
		}
		return false;
	}
}
