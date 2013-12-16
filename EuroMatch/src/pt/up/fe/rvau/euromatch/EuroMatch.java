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
							//System.out.println("Error: " + ex.getMessage());
							//ex.printStackTrace();
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
		DetectedBillsResult result =
				new BillDetector(featureDetector, descriptorExtractor, descriptorMatcher).performDetection(imageFileName);

		Highgui.imwrite(RESULT_FOLDER + imageFileName + '.' + algorithmsCombination + ".result.jpg", result.outputImage);
		long endTime = System.nanoTime();
		
		double elapsedTime = (endTime - startTime) / 1000000000.0;
		
		String textResultFilename = RESULT_FOLDER + imageFileName + '.' + algorithmsCombination + ".result.txt";
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(textResultFilename)))
		{
			for (int i = 0; i < result.detectedBills.size(); ++i) {
				DetectedBill detectedBill = result.detectedBills.get(i);
				writer.write("Bill: " + detectedBill.getValue() + "\r\n");
			}
			writer.write("Matched " + result.detectedBills.size() + " bills" + "\r\n");
			writer.write("Total Value: " + result.totalValue + "\r\n");
			writer.write("Total Elapsed Time: " + elapsedTime + "\r\n");
		} catch (IOException e)
		{
			e.printStackTrace();
		}
	}
}
