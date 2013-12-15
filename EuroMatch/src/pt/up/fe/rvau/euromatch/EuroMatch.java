/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch;

import pt.up.fe.rvau.euromatch.jutils.PointUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import pt.up.fe.rvau.euromatch.jutils.ImageConverter;
import pt.up.fe.rvau.euromatch.jutils.MatConverter;
import pt.up.fe.rvau.euromatch.jutils.VisualHelper;

/**
 *
 * @author luiscubal
 */
public class EuroMatch {
    static {
        System.loadLibrary("opencv_java247");
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Starting");
        
        Mat object = Highgui.imread("50eu_r.jpg", Highgui.CV_LOAD_IMAGE_COLOR);
        Mat scene = Highgui.imread("scenes.png", Highgui.CV_LOAD_IMAGE_COLOR);
        
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		
		BillInfo billInfo = new BillInfo(object);
		
		MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
        featureDetector.detect(billInfo.getBillImage(), objectKeypoints);
		Mat objectDescriptor = new Mat();
		extractor.compute(billInfo.getBillImage(), objectKeypoints, objectDescriptor);
		
		billInfo.setKeyPoints(objectKeypoints);
		billInfo.setDescriptor(objectDescriptor);
		
		Mat outputImage = scene.clone();
		List<DetectedBill> detectedBills = detectBills(outputImage,
				billInfo, scene, featureDetector, extractor);
		
        for (int i = 0; i < detectedBills.size(); ++i) {
			DetectedBill detectedBill = detectedBills.get(i);
			drawBill(outputImage, detectedBill, 0, getColorForIndex(i));
		}
		System.out.println("Matched " + detectedBills.size() + " bills");
		
        try {
            VisualHelper.showImageFrame(ImageConverter.convert(outputImage, 640, 480));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

	private static List<DetectedBill> detectBills(Mat outImage, BillInfo billInfo, Mat scene, FeatureDetector featureDetector, DescriptorExtractor extractor) {
		
		List<DetectedBill> detectedBills = new ArrayList<>();
		
        MatOfKeyPoint sceneKeypoints = new MatOfKeyPoint();
        featureDetector.detect(scene, sceneKeypoints);
		//We need to cast sceneKeyPoints.toList() to an ArrayList
		// because the "normal" implementation of sceneKeyPoints.toList().iterator()
		// does not support remove.
        List<KeyPoint> sceneKeypointsList = new ArrayList<>(sceneKeypoints.toList());
		
		Features2d.drawKeypoints(scene, sceneKeypoints, outImage);
		
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
		for (int billIndex = 0;;++billIndex) {
		
			Mat sceneDescriptor = new Mat();
			extractor.compute(scene, sceneKeypoints, sceneDescriptor);

			MatOfDMatch matches = new MatOfDMatch();
			matcher.match(billInfo.getDescriptor(), sceneDescriptor, matches);
			List<DMatch> matchesList = matches.toList();

			float maxDistance = 0;
			float minDistance = 100;
			for (int i = 0; i < billInfo.getDescriptor().rows(); ++i) {
				float distance = matchesList.get(i).distance;
				if (distance > maxDistance) maxDistance = distance;
				if (distance < minDistance) minDistance = distance;
			}

			System.out.println("Min: " + minDistance);
			System.out.println("Max: " + maxDistance);

			List<DMatch> goodMatches = new ArrayList<>();
			for (DMatch match : matchesList) {
				if (match.distance < 2 * minDistance) {
					goodMatches.add(match);
				}
			}
			List<Point> objPoints = new ArrayList<>();
			List<Point> scenePoints = new ArrayList<>();
			for (DMatch goodMatch : goodMatches) {
				objPoints.add(billInfo.getKeyPointsAsList().get(goodMatch.queryIdx).pt);
				scenePoints.add(sceneKeypointsList.get(goodMatch.trainIdx).pt);
			}
			MatOfPoint2f objPointsMat = MatConverter.convertToMatOfPoint2f(objPoints);
			MatOfPoint2f scenePointsMat = MatConverter.convertToMatOfPoint2f(scenePoints);

			Mat homography = Calib3d.findHomography(objPointsMat, scenePointsMat, Calib3d.RANSAC, 2);
			List<Point> cornersInObject = new ArrayList<>();
			cornersInObject.add(new Point(0, 0));
			cornersInObject.add(new Point(billInfo.getImageWidth(), 0));
			cornersInObject.add(new Point(billInfo.getImageWidth(), billInfo.getImageHeight()));
			cornersInObject.add(new Point(0, billInfo.getImageHeight()));
			Mat cornersInObjectMat = MatConverter.convertToMatOfType(cornersInObject, CvType.CV_32FC2);

			Mat cornersInSceneMat = new Mat(4, 1, CvType.CV_32FC2);
			Core.perspectiveTransform(cornersInObjectMat, cornersInSceneMat, homography);
			List<Point> cornersInScene = MatConverter.toPointList(cornersInSceneMat);

			DetectedBill detectedBill = new DetectedBill(cornersInScene);
			
			List<KeyPoint> removedKeyPoints = new ArrayList<>();
			removeUsedKeypoints(sceneKeypointsList, removedKeyPoints, detectedBill);
			int removedPoints = removedKeyPoints.size();
			System.out.println("Removed " + removedPoints);
			if (removedPoints > 10) {
				detectedBills.add(detectedBill);
			} else {
				break;
			}
			
			sceneKeypoints = MatConverter.convertToMatOfKeyPoint(sceneKeypointsList);
		}
		
		return detectedBills;
	}
	
	private static void removeUsedKeypoints(List<KeyPoint> keyPoints,
			List<KeyPoint> keyPointSink,
			DetectedBill detectedBill)
	{
		
		MatOfPoint2f contour = MatConverter.convertToMatOfPoint2f(detectedBill.getPoints());
		Iterator<KeyPoint> it = keyPoints.iterator();
		while (it.hasNext()) {
			KeyPoint keyPoint = it.next();
			Point point = keyPoint.pt;
			if (Imgproc.pointPolygonTest(contour, point, false) > 0) {
				//Point contained in object
				keyPointSink.add(keyPoint);
				it.remove();
			}
		}
	}
	
	private static void drawBill(Mat outputImage, DetectedBill detectedBill, int offset, Scalar color) {
		List<Point> points = detectedBill.getPoints();
		
		Core.line(
                outputImage,
                PointUtils.add(points.get(0), new Point(offset, 0)),
                PointUtils.add(points.get(1), new Point(offset, 0)),
                color,
                4);
        Core.line(
                outputImage,
                PointUtils.add(points.get(1), new Point(offset, 0)),
                PointUtils.add(points.get(2), new Point(offset, 0)),
                color,
                4);
        Core.line(
                outputImage,
                PointUtils.add(points.get(2), new Point(offset, 0)),
                PointUtils.add(points.get(3), new Point(offset, 0)),
                color,
                4);
        Core.line(
                outputImage,
                PointUtils.add(points.get(3), new Point(offset, 0)),
                PointUtils.add(points.get(0), new Point(offset, 0)),
                color,
                4);
	}
    
	private static Scalar getColorForIndex(int i) {
		switch (i) {
			case 0: return new Scalar(255, 0, 0);
			case 1: return new Scalar(0, 255, 0);
			case 2: return new Scalar(0, 0, 255);
			case 3: return new Scalar(255, 255, 0);
			case 4: return new Scalar(255, 0, 255);
			case 5: return new Scalar(0, 255, 255);
			default: return new Scalar(255, 255, 255);
		}
	}
}
