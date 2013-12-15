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
    
	private static final int IMAGE_FORMAT = Highgui.CV_LOAD_IMAGE_COLOR;
	
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Starting");
        
        Mat scene = Highgui.imread("scenes.png", IMAGE_FORMAT);
        
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.DYNAMIC_SURF);
		DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
		
		List<BillInfo> billInfo = loadBills(featureDetector, extractor);
		
		Mat outputImage = scene.clone();
		List<DetectedBill> detectedBills = detectBills(outputImage,
				billInfo, scene, featureDetector, extractor, matcher);
		
		int totalValue = 0;
        for (int i = 0; i < detectedBills.size(); ++i) {
			DetectedBill detectedBill = detectedBills.get(i);
			drawBill(outputImage, detectedBill, 0, getColorForIndex(i));
			System.out.println("Bill: " + detectedBill.getValue());
			totalValue += detectedBill.getValue();
		}
		System.out.println("Matched " + detectedBills.size() + " bills");
		System.out.println("Total Value: " + totalValue);
		
        try {
            VisualHelper.showImageFrame(ImageConverter.convert(outputImage, 640, 480));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

	private static List<BillInfo> loadBills(FeatureDetector featureDetector,
			DescriptorExtractor extractor) {
		List<BillInfo> billInfo = new ArrayList<>();
		billInfo.add(loadBill(featureDetector, extractor, "5eu_r.jpg", 5,
				new int[] { 184, 3, 230, 70 }));
		billInfo.add(loadBill(featureDetector, extractor, "5eu_v.jpg", 5,
				new int[] { 207, 104, 264, 137,
							64, 37, 70, 141,
							5, 8, 35, 32,
							40, 40, 78, 72}));
		billInfo.add(loadBill(featureDetector, extractor, "10eu_r.jpg", 10,
				new int[] { 164, 3, 260, 60,
							3, 103, 97, 141,
							0, 0, 63, 32}));
		billInfo.add(loadBill(featureDetector, extractor, "10eu_v.jpg", 10,
				new int[] { 164, 3, 260, 60,
							3, 103, 97, 141,
							0, 0, 63, 32}));
		billInfo.add(loadBill(featureDetector, extractor, "20eu_r.jpg", 20,
				new int[] { 138, 3, 264, 60,
							5, 107, 37, 135,
							0, 0, 30, 30,
							135, 53, 178, 90}));
		billInfo.add(loadBill(featureDetector, extractor, "20eu_v.jpg", 20,
				new int[] { 204, 116, 264, 143,
							3, 113, 77, 141,
							0, 0, 90, 64,
							230, 0, 260, 30}));
		billInfo.add(loadBill(featureDetector, extractor, "50eu_r.jpg", 50,
				new int[] { 222, 78, 258, 116,
							164, 4, 230, 55,
							0, 113, 77, 141}));
		billInfo.add(loadBill(featureDetector, extractor, "50eu_v.jpg", 50,
				new int[] { 214, 107, 260, 145,
							0, 114, 85, 146}));
		return billInfo;
	}
	
	private static BillInfo loadBill(FeatureDetector featureDetector,
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
	
	private static List<DetectedBill> detectBills(
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
			System.out.println("=====================");
			System.out.println("Testing " + billInfo.getValue());
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
					if (distance < minDistance) {
						minDistance = distance;
					}
				}

				System.out.println("Min: " + minDistance);
				System.out.println("Max: " + maxDistance);
				
				//Features2d.drawMatches(billInfo.getBillImage(), billInfo.getKeyPoints(), scene, sceneKeypoints, matches, outImage);
				
				List<DMatch> goodMatches = new ArrayList<>();
				for (DMatch match : matchesList) {
					if (match.distance < 2 * minDistance) {
						goodMatches.add(match);
					}
				}
				System.out.println("GM=" + goodMatches.size());
				List<Point> objPoints = new ArrayList<>();
				List<Point> scenePoints = new ArrayList<>();
				for (DMatch goodMatch : goodMatches) {
					objPoints.add(billInfo.getKeyPointsAsList().get(goodMatch.queryIdx).pt);
					scenePoints.add(sceneKeypointsList.get(goodMatch.trainIdx).pt);
				}
				MatOfPoint2f objPointsMat = MatConverter.convertToMatOfPoint2f(objPoints);
				MatOfPoint2f scenePointsMat = MatConverter.convertToMatOfPoint2f(scenePoints);
				System.out.println("SP=" + scenePoints.size());
				
				//MatOfDMatch goodMatchesMat = MatConverter.convert(goodMatches);
				//Features2d.drawMatches(billInfo.getBillImage(), billInfo.getKeyPoints(), scene, sceneKeypoints, goodMatchesMat, outImage);
				
				if (scenePoints.size() < 4)
					break;
				Mat homography = Calib3d.findHomography(objPointsMat, scenePointsMat, Calib3d.RANSAC, 3);
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
					System.out.println("Rejected impossible bill");
					break;
				}
				
				DetectedBill detectedBill = new DetectedBill(billInfo.getValue(), cornersInScene);

				List<KeyPoint> removedKeyPoints = new ArrayList<>();
				removeUsedKeypoints(sceneKeypointsList, removedKeyPoints, detectedBill);
				int removedPoints = removedKeyPoints.size();
				System.out.println("Removed " + removedPoints);
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
	
	private static void removeUsedKeypoints(
			List<KeyPoint> keyPoints,
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

	private static boolean isBillValid(List<Point> points) {
		final double EPSILON = 0.1;
		
		double angleSum = 0;
		for (int i = 0; i < 4; ++i) {
			double angle = computeAngle(points.get(i % 4),
					points.get((i + 1) % 4),
					points.get((i + 2) % 4));
			//angle = Math.abs(angle);
			if (angle > Math.PI) {
				System.out.println("Rejected: Bad angle " + angle);
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
	
	private static double computeAngle(Point previous, Point center, Point next) {
		double angle = Math.atan2(next.x - center.x,next.y - center.y)-
                        Math.atan2(previous.x- center.x,previous.y- center.y);
		if (angle < 0) angle += 2 * Math.PI;
		return angle;
	}
}
