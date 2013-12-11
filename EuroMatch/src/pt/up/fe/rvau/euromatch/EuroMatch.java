/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import pt.up.fe.rvau.euromatch.jutils.ImageConverter;
import pt.up.fe.rvau.euromatch.jutils.VisualHelper;

/**
 *
 * @author luiscubal
 */
public class EuroMatch {

    private static final String BASE_LIBRARY_PATH = "C:\\Users\\luiscubal\\Downloads\\opencv\\build\\java\\x64\\";
    
    static {
        System.loadLibrary("opencv_java247");
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Starting");
        
        Mat object = Highgui.imread("50eu_r.jpg", Highgui.CV_LOAD_IMAGE_COLOR);
        Mat scene = Highgui.imread("scene.png", Highgui.CV_LOAD_IMAGE_COLOR);
        
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
        
        MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
        featureDetector.detect(object, objectKeypoints);
        MatOfKeyPoint sceneKeypoints = new MatOfKeyPoint();
        featureDetector.detect(scene, sceneKeypoints);
        
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        
        Mat objectDescriptor = new Mat();
        extractor.compute(object, objectKeypoints, objectDescriptor);
        Mat sceneDescriptor = new Mat();
        extractor.compute(scene, sceneKeypoints, sceneDescriptor);
        
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(objectDescriptor, sceneDescriptor, matches);
        List<DMatch> matchesList = matches.toList();
        
        float maxDistance = 0;
        float minDistance = 100;
        for (int i = 0; i < objectDescriptor.rows(); ++i) {
            float distance = matchesList.get(i).distance;
            if (distance > maxDistance) maxDistance = distance;
            if (distance < minDistance) minDistance = distance;
        }
        
        System.out.println("Min: " + minDistance);
        System.out.println("Max: " + maxDistance);
        
        try {
            VisualHelper.showImageFrame(ImageConverter.convert(object, 640, 480));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}
