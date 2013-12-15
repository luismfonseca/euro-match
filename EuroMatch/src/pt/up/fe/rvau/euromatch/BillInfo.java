/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;

/**
 *
 * @author luiscubal
 */
public class BillInfo {
	private Mat billImage;
	private MatOfKeyPoint keyPoints;
	private Mat descriptor;
	
	public BillInfo(Mat billImage) {
		this.billImage = billImage;
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
	
	public void setKeyPoints(MatOfKeyPoint keyPoints) {
		this.keyPoints = keyPoints;
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
}
