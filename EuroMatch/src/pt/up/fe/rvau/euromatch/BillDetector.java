/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch;

import com.google.gson.Gson;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
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
import pt.up.fe.rvau.euromatch.jutils.PointUtils;
import pt.up.fe.rvau.euromatch.jutils.VisualHelper;

/**
 *
 * @author luiscubal
 */
public class BillDetector {
	private static final int IMAGE_FORMAT = Highgui.CV_LOAD_IMAGE_COLOR;
	private static final int RANSAC_THRESHOLD = 3;
	
	private FeatureDetector featureDetector;
	private DescriptorExtractor extractor;
	private DescriptorMatcher matcher;
	private List<BillInfo> billInfo;
	
	public BillDetector(
			int featureDetectorType,
			int descriptorExtractorType,
			int descriptorMatcherType) {
		
		featureDetector = FeatureDetector.create(featureDetectorType);
		extractor = DescriptorExtractor.create(descriptorExtractorType);
		matcher = DescriptorMatcher.create(descriptorMatcherType);
		
		billInfo = loadBills(featureDetector, extractor);
	}
	
	public DetectedBillsResult performDetection(String filename) {
		Mat scene = Highgui.imread(filename, IMAGE_FORMAT);
		return performDetection(scene);
	}
	
	public DetectedBillsResult performDetection(Mat scene) {
		Mat outputImage = scene.clone();
		
		List<DetectedBill> detectedBills = detectBills(outputImage,
				billInfo, scene, featureDetector, extractor, matcher);
		
		//System.out.println("Matched " + detectedBills.size() + " bills");
		//System.out.println("Total Value: " + totalValue);/**/
		
		int totalValue = 0;
        for (int i = 0; i < detectedBills.size(); ++i) {
			DetectedBill detectedBill = detectedBills.get(i);
			drawBill(outputImage, detectedBill, 0, getColorForIndex(i));
			totalValue += detectedBill.getValue();
		}
		
		DetectedBillsResult result = new DetectedBillsResult();
		result.detectedBills = detectedBills;
		result.outputImage = outputImage;
		result.totalValue = totalValue;
		return result;
    }

	private List<BillInfo> loadBills(FeatureDetector featureDetector,
			DescriptorExtractor extractor) {
		List<BillInfo> billInfo = new ArrayList<>();
		//billInfo.add(loadBill(featureDetector, extractor, "5eu_r.jpg", 5,
				//new int[] { 184, 3, 230, 70 }));
		
		Gson gson = new Gson();
		File folder = new File("./bills");
		for (File file : folder.listFiles()) {
			if (!file.getName().endsWith(".json"))
				continue;
			
			String jsonFileName = file.getAbsolutePath();
			String imageFileName = jsonFileName.substring(0, jsonFileName.length() - ".json".length()) + ".jpg";
			
			try (Reader reader = new FileReader(jsonFileName)) {
				BillDescription description = gson.fromJson(reader, BillDescription.class);
				
				billInfo.add(loadBill(featureDetector, extractor, imageFileName,
						description.value,
						description.relevant_parts));
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		return billInfo;
	}
	
	private BillInfo loadBill(FeatureDetector featureDetector,
			DescriptorExtractor extractor, String billName, int value,
			int[] relevantParts) {
		
		if (relevantParts.length % 4 != 0)
			throw new IllegalArgumentException();
		
		Mat object = Highgui.imread(billName, IMAGE_FORMAT);
		if (object.empty())
			throw new RuntimeException("Could not load " + billName);
		BillInfo billInfo = new BillInfo(object, value);
		MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
        featureDetector.detect(billInfo.getBillImage(), objectKeypoints);
		billInfo.setKeyPoints(objectKeypoints, relevantParts);
		
		Mat objectDescriptor = new Mat();
		extractor.compute(billInfo.getBillImage(), billInfo.getKeyPoints(), objectDescriptor);
		billInfo.setDescriptor(objectDescriptor);
		
		return billInfo;
	}
	
	private List<DetectedBill> detectBills(
			Mat outImage,
			List<BillInfo> billTypes,
			Mat scene,
			FeatureDetector featureDetector,
			DescriptorExtractor extractor,
			DescriptorMatcher matcher) {
		
		List<DetectedBill> detectedBills = new ArrayList<>();
		
        
		MatOfKeyPoint sceneKeypoints = new MatOfKeyPoint();
        featureDetector.detect(scene, sceneKeypoints);
		//We need to cast sceneKeyPoints.toList() to an ArrayList
		// because the "normal" implementation of sceneKeyPoints.toList().iterator()
		// does not support remove.
        List<KeyPoint> sceneKeypointsList = new ArrayList<>(sceneKeypoints.toList());
		Features2d.drawKeypoints(scene, sceneKeypoints, outImage);
		
		for (BillInfo billInfo : billTypes) {
			//System.out.println("=====================");
			//System.out.println("Testing " + billInfo.getValue());
			for (int billIndex = 0;;++billIndex) {

				//Features2d.drawKeypoints(scene, sceneKeypoints, outImage);
				
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
					if (distance < minDistance) {
						minDistance = distance;
					}
				}

				if (minDistance < 1e-4) {
					minDistance = 0.001f;
				}
				//System.out.println("Min: " + minDistance);
				//System.out.println("Max: " + maxDistance);
				
				//Features2d.drawMatches(billInfo.getBillImage(), billInfo.getKeyPoints(), scene, sceneKeypoints, matches, outImage);
				
				List<DMatch> goodMatches = new ArrayList<>();
				for (DMatch match : matchesList) {
					if (match.distance < 2 * minDistance) {
						goodMatches.add(match);
					}
				}
				//System.out.println("GM=" + goodMatches.size());
				List<Point> objPoints = new ArrayList<>();
				List<Point> scenePoints = new ArrayList<>();
				for (DMatch goodMatch : goodMatches) {
					objPoints.add(billInfo.getKeyPointsAsList().get(goodMatch.queryIdx).pt);
					scenePoints.add(sceneKeypointsList.get(goodMatch.trainIdx).pt);
				}
				MatOfPoint2f objPointsMat = MatConverter.convertToMatOfPoint2f(objPoints);
				MatOfPoint2f scenePointsMat = MatConverter.convertToMatOfPoint2f(scenePoints);
				//System.out.println("SP=" + scenePoints.size());
				
				//MatOfDMatch goodMatchesMat = MatConverter.convert(goodMatches);
				//Features2d.drawMatches(billInfo.getBillImage(), billInfo.getKeyPoints(), scene, sceneKeypoints, goodMatchesMat, outImage);
				
				if (scenePoints.size() < 4)
					break;
				Mat homography = Calib3d.findHomography(objPointsMat, scenePointsMat, Calib3d.RANSAC, RANSAC_THRESHOLD);
				List<Point> cornersInObject = new ArrayList<>();
				cornersInObject.add(new Point(0, 0));
				cornersInObject.add(new Point(billInfo.getImageWidth(), 0));
				cornersInObject.add(new Point(billInfo.getImageWidth(), billInfo.getImageHeight()));
				cornersInObject.add(new Point(0, billInfo.getImageHeight()));
				Mat cornersInObjectMat = MatConverter.convertToMatOfType(cornersInObject, CvType.CV_32FC2);

				Mat cornersInSceneMat = new Mat(4, 1, CvType.CV_32FC2);
				Core.perspectiveTransform(cornersInObjectMat, cornersInSceneMat, homography);
				List<Point> cornersInScene = MatConverter.toPointList(cornersInSceneMat);

				if (!isBillValid(cornersInScene)) {
					//System.out.println("Rejected impossible bill");
					break;
				}
				
				DetectedBill detectedBill = new DetectedBill(billInfo, cornersInScene);

				List<KeyPoint> removedKeyPoints = new ArrayList<>();
				removeUsedKeypoints(sceneKeypointsList, removedKeyPoints, detectedBill, homography);
				int removedPoints = removedKeyPoints.size();
				//System.out.println("Removed " + removedPoints);
				if (removedPoints > 10) {
					detectedBills.add(detectedBill);
					sceneKeypoints = MatConverter.convertToMatOfKeyPoint(sceneKeypointsList);
				} else {
					sceneKeypointsList.addAll(removedKeyPoints);
					sceneKeypoints = MatConverter.convertToMatOfKeyPoint(sceneKeypointsList);
					break;
				}
			}
		}
		
		return detectedBills;
	}
	
	private void removeUsedKeypoints(
			List<KeyPoint> keyPoints,
			List<KeyPoint> keyPointSink,
			DetectedBill detectedBill,
			Mat homography)
	{
		
		MatOfPoint2f contour = MatConverter.convertToMatOfPoint2f(detectedBill.getPoints());
		Iterator<KeyPoint> it = keyPoints.iterator();
		while (it.hasNext()) {
			KeyPoint keyPoint = it.next();
			Point point = keyPoint.pt;
			if (Imgproc.pointPolygonTest(contour, point, false) > 0) {
				//Point contained in object
				if (isPointInRelevantArea(point, detectedBill, homography)) {
					keyPointSink.add(keyPoint);
					it.remove();
				}
			}
		}
	}
	
	private boolean isPointInRelevantArea(Point point, DetectedBill detectedBill, Mat homography) {
		BillInfo billInfo = detectedBill.getBillInfo();
		List<Point> relevantPoints = billInfo.getRelevantPoints();
		for (int i = 0; i < relevantPoints.size(); i += 2) {
			List<Point> srcPoints = new ArrayList<Point>();
			Point p1 = relevantPoints.get(i);
			Point p3 = relevantPoints.get(i + 1);
			
			Point p2 = new Point(p1.x, p3.y);
			Point p4 = new Point(p3.x, p1.y);
			
			srcPoints.add(p1);
			srcPoints.add(p2);
			srcPoints.add(p3);
			srcPoints.add(p4);
			
			MatOfPoint2f pointsMat = MatConverter.convertToMatOfPoint2f(srcPoints);
			MatOfPoint2f destPoints = new MatOfPoint2f();
			Core.perspectiveTransform(pointsMat, destPoints, homography);
			
			if (Imgproc.pointPolygonTest(destPoints, point, false) >= 0) {
				return true;
			}
		}
		return false;
	}
	
	private void drawBill(Mat outputImage, DetectedBill detectedBill, int offset, Scalar color) {
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
    
	

	private boolean isBillValid(List<Point> points) {
		final double EPSILON = 0.1;
		
		double angleSum = 0;
		for (int i = 0; i < 4; ++i) {
			double angle = computeAngle(points.get(i % 4),
					points.get((i + 1) % 4),
					points.get((i + 2) % 4));
			//angle = Math.abs(angle);
			if (angle > Math.PI) {
				//System.out.println("Rejected: Bad angle " + angle);
				return false;
			}
			angleSum += angle;
		}
		return true;
		/*
		angleSum = Math.abs(angleSum);
		final double zero = Math.abs(angleSum - 2 * Math.PI);
		return zero < EPSILON;*/
	}
	
	private double computeAngle(Point previous, Point center, Point next) {
		double angle = Math.atan2(next.x - center.x,next.y - center.y)-
                        Math.atan2(previous.x- center.x,previous.y- center.y);
		if (angle < 0) angle += 2 * Math.PI;
		return angle;
	}
	
	private static Scalar getColorForIndex(int i) {
		switch (i % 8) {
			case 0: return new Scalar(255, 0, 0); //Blue
			case 1: return new Scalar(0, 255, 0); //Green
			case 2: return new Scalar(0, 0, 255); //Red
			case 3: return new Scalar(255, 255, 0); //Cyan
			case 4: return new Scalar(255, 0, 255); //Magenta
			case 5: return new Scalar(0, 255, 255); //Yellow
			case 6: return new Scalar(255, 128, 0); //Cyan-ish blue
			case 7: return new Scalar(0, 128, 255); //Orange
			default: return new Scalar(255, 255, 255); //White
		}
	}
}
