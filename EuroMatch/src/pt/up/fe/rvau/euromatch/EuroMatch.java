package pt.up.fe.rvau.euromatch;

import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.Mongo;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.net.UnknownHostException;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;

/**
 * This class is used for testing and performing bills matching.
 * @author luiscubal, luisfonseca
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
			DescriptorMatcher.FLANNBASED,
            DescriptorMatcher.BRUTEFORCE_HAMMING
		};
	public static final String[] DESCRIPTOR_MATCHERS_STRING = new String [] {
			"DescriptorMatcher.BRUTEFORCE",
			"DescriptorMatcher.FLANNBASED",
            "DescriptorMatcher.BRUTEFORCE_HAMMING"
		};
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws UnknownHostException {
        System.out.println("Starting tests");
		testSingleNoteResillience();
	}
    
    public static void testSingleNoteResillience() throws UnknownHostException {
        
        Mongo mongo = new Mongo();
        DB db = mongo.getDB("testset");
        DBCollection dbCollection = db.getCollection("singleNotes");
        String[] imageFiles = new String[] {
            "test5.png",
            "test5r.png",
            "test10.png", "test10r.png",
            "test20.png", "test20r.png",
            "test50.png", "test50r.png"
        };
        int[] imageFilesNoteValue = new int[] {
            5, 5,
            10, 10,
            20, 20,
            50, 50
        };

        for (int d = 0; d < FEATURE_DETECTORS.length; ++d) {
            for (int e = 0; e < DESCRIPTOR_EXTRACTORS.length; ++e) {
                for (int m = 0; m < DESCRIPTOR_MATCHERS.length; ++m) {
                    for (int i = 0; i < imageFiles.length; ++i) {
                        String imageFileName = imageFiles[i];
                        int featureDetector = FEATURE_DETECTORS[d];
                        int descriptorExtractor = DESCRIPTOR_EXTRACTORS[e];
                        int descriptorMatcher = DESCRIPTOR_MATCHERS[m];
                        String algorithmsCombination
                           = FEATURE_DETECTORS_STRING[d] + "-"
                           + DESCRIPTOR_EXTRACTORS_STRING[e] + "-"
                           + DESCRIPTOR_MATCHERS_STRING[m];
                        try {
                            DetectedBillsResult result
                               = testPerformDetection(imageFileName, algorithmsCombination, featureDetector, descriptorExtractor, descriptorMatcher);
                            
                            int totalValue = 0;
                            for (DetectedBill bill : result.detectedBills)
                            {
                                totalValue += bill.getValue();
                            }
                            
                            BasicDBObject document = new BasicDBObject("file", imageFiles[i]).
                                append("featureDetectors", FEATURE_DETECTORS_STRING[d]).
                                append("descriptorExtractor", DESCRIPTOR_EXTRACTORS_STRING[e]).
                                append("descriptorMatcher", DESCRIPTOR_MATCHERS_STRING[m]).
                                append("notesValue", imageFilesNoteValue[i]).
                                append("notesFoundCount", result.detectedBills.size()).
                                append("elapsedTime", result.elapsedTime).
                                append("totalValue", totalValue);
                            
                            dbCollection.insert(document);
                        } catch (Exception ex) {
                            System.out.println("Error: " + ex.getMessage());
                            ex.printStackTrace();
                        }
                    }
                }
            }
        }
    }
	
	public static DetectedBillsResult testPerformDetection(
			String imageFileName, String algorithmsCombination, int featureDetector, int descriptorExtractor, int descriptorMatcher)
	{
		long startTime = System.nanoTime();
		DetectedBillsResult result =
				new BillDetector(featureDetector, descriptorExtractor, descriptorMatcher).performDetection(imageFileName);

		Highgui.imwrite(RESULT_FOLDER + imageFileName + '.' + algorithmsCombination + ".result.jpg", result.outputImage);
		long endTime = System.nanoTime();
		
        result.elapsedTime = (endTime - startTime) / 1000000000.0;
		
		String textResultFilename = RESULT_FOLDER + imageFileName + '.' + algorithmsCombination + ".result.txt";
		try (BufferedWriter writer = new BufferedWriter(new FileWriter(textResultFilename)))
		{
			for (int i = 0; i < result.detectedBills.size(); ++i) {
				DetectedBill detectedBill = result.detectedBills.get(i);
				writer.write("Bill: " + detectedBill.getValue() + "\r\n");
			}
			writer.write("Matched " + result.detectedBills.size() + " bills" + "\r\n");
			writer.write("Total Value: " + result.totalValue + "\r\n");
			writer.write("Total Elapsed Time: " + result.elapsedTime + "\r\n");
		} catch (IOException e)
		{
			e.printStackTrace();
		}
        return result;
	}
}
