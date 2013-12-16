package pt.up.fe.rvau.euromatch;

import com.google.gson.Gson;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import pt.up.fe.rvau.euromatch.jutils.PointUtils;
import java.io.IOException;
import java.io.Reader;
import java.io.StringWriter;
import java.io.Writer;
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
	private static final int RANSAC_THRESHOLD = 3;
	private static final String RESULT_FOLDER = "./results/";
	
	public static final Integer[] FEATURE_DETECTORS = new Integer [] {
			FeatureDetector.SURF, FeatureDetector.DYNAMIC_SURF,
			FeatureDetector.FAST, FeatureDetector.BRISK,
			FeatureDetector.SIFT
		};
	public static final String[] FEATURE_DETECTORS_STRING = new String [] {
			"FeatureDetector.SURF", "FeatureDetector.DYNAMIC_SURF",
			"FeatureDetector.FAST", "FeatureDetector.BRISK",
			"FeatureDetector.SIFT"
		};
	public static final Integer[] DESCRIPTOR_EXTRACTORS = new Integer [] {
			DescriptorExtractor.SURF, DescriptorExtractor.BRIEF,
			DescriptorExtractor.BRISK, DescriptorExtractor.FREAK,
			DescriptorExtractor.ORB, DescriptorExtractor.SIFT,
			DescriptorExtractor.OPPONENT_SURF
		};
	public static final String[] DESCRIPTOR_EXTRACTORS_STRING = new String [] {
			"DescriptorExtractor.SURF", "DescriptorExtractor.BRIEF",
			"DescriptorExtractor.BRISK", "DescriptorExtractor.FREAK",
			"DescriptorExtractor.ORB", "DescriptorExtractor.SIFT",
			"DescriptorExtractor.OPPONENT_SURF"
		};
	public static final Integer[] DESCRIPTOR_MATCHERS = new Integer [] {
			DescriptorMatcher.BRUTEFORCE,
			DescriptorMatcher.FLANNBASED
		};
	public static final String[] DESCRIPTOR_MATCHERS_STRING = new String [] {
			"DescriptorMatcher.BRUTEFORCE",
			"DescriptorMatcher.FLANNBASED"
		};
	
	static long endTime;
    /**
     * @param args the command line arguments
     */
    public static void xmain(String[] args) {
        System.out.println("Starting");
		
		String imageFiles[] = new String[] { "scenes.png", "scenesx.png", "scenesy20r.png",  "scenesy50r.png"};
		
		for (int d = 0; d < FEATURE_DETECTORS.length; ++d)
		{
			for (int e = 0; e < DESCRIPTOR_EXTRACTORS.length; ++e)
			{
				for (int m = 0; m < DESCRIPTOR_MATCHERS.length; ++m)
				{
					for (String imageFileName : imageFiles)
					{
						int featureDetector = FEATURE_DETECTORS[d];
						int descriptorExtractor = DESCRIPTOR_EXTRACTORS[e];
						int descriptorMatcher = DESCRIPTOR_MATCHERS[m];
						String algorithmsCombination =
								FEATURE_DETECTORS_STRING[d] + "-" +
								DESCRIPTOR_EXTRACTORS_STRING[e] + "-" +
								DESCRIPTOR_MATCHERS_STRING[m];
						try {
							testPerformDetection(imageFileName, algorithmsCombination, featureDetector, descriptorExtractor, descriptorMatcher);
						} catch(Exception ex)
						{
							System.out.println("Error: " + ex.getMessage());
							ex.printStackTrace();
						}
					}
				}
			}
		}
	}
	
	public static void testPerformDetection(
			String imageFileName, String algorithmsCombination, int featureDetector, int descriptorExtractor, int descriptorMatcher)
	{
		long startTime = System.nanoTime();
		List<DetectedBill> detectedBills =
				performDetection(imageFileName, algorithmsCombination, featureDetector, descriptorExtractor, descriptorMatcher);

		double elapsedTime = (endTime - startTime) / 1000000000.0;
		String textResultFilename = RESULT_FOLDER + imageFileName + '.' + algorithmsCombination + ".result.txt";
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(textResultFilename)))
		{
			int totalValue = 0;
			for (int i = 0; i < detectedBills.size(); ++i) {
				DetectedBill detectedBill = detectedBills.get(i);
				writer.write("Bill: " + detectedBill.getValue() + "\r\n");
				totalValue += detectedBill.getValue();
			}
			writer.write("Matched " + detectedBills.size() + " bills" + "\r\n");
			writer.write("Total Value: " + totalValue + "\r\n");
			writer.write("Total Elapsed Time: " + elapsedTime + "\r\n");
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}
	
	public static List<DetectedBill> performDetection(
			String name, String algorithmsCombination,
			int featureDetectorType, int descriptorExtractorType, int descriptorMatcherType) {
        Mat scene = Highgui.imread(name, IMAGE_FORMAT);
		
        FeatureDetector featureDetector = FeatureDetector.create(featureDetectorType);
		DescriptorExtractor extractor = DescriptorExtractor.create(descriptorExtractorType);
		DescriptorMatcher matcher = DescriptorMatcher.create(descriptorMatcherType);
		
		List<BillInfo> billInfo = loadBills(featureDetector, extractor);
		
		Mat outputImage = scene.clone();
		List<DetectedBill> detectedBills = detectBills(outputImage,
				billInfo, scene, featureDetector, extractor, matcher);
		
		
		int totalValue = 0;
        for (int i = 0; i < detectedBills.size(); ++i) {
			DetectedBill detectedBill = detectedBills.get(i);
			drawBill(outputImage, detectedBill, 0, getColorForIndex(i));
			//System.out.println("Bill: " + detectedBill.getValue());
			totalValue += detectedBill.getValue();
		}
		//System.out.println("Matched " + detectedBills.size() + " bills");
		//System.out.println("Total Value: " + totalValue);/**/
		
		endTime = System.nanoTime();
		
		//Highgui.imwrite(RESULT_FOLDER + name + '.' + algorithmsCombination + ".result.jpg", outputImage);
        try {
            VisualHelper.showImageFrame(ImageConverter.convert(outputImage, 640, 480), totalValue);
        } catch (IOException e) {
            e.printStackTrace();
        }
		return detectedBills;
    }

	private static List<BillInfo> loadBills(FeatureDetector featureDetector,
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
					System.out.println("Rejected impossible bill");
					break;
				}
				
				DetectedBill detectedBill = new DetectedBill(billInfo, cornersInScene);

				List<KeyPoint> removedKeyPoints = new ArrayList<>();
				removeUsedKeypoints(sceneKeypointsList, removedKeyPoints, detectedBill, homography);
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
	
	private static boolean isPointInRelevantArea(Point point, DetectedBill detectedBill, Mat homography) {
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
