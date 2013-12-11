/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch;

import pt.up.fe.rvau.euromatch.jutils.PointUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
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
        
        Mat object = Highgui.imread("10eu_r.jpg", Highgui.CV_LOAD_IMAGE_COLOR);
        Mat scene = Highgui.imread("scene.png", Highgui.CV_LOAD_IMAGE_COLOR);
        
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
        
        MatOfKeyPoint objectKeypoints = new MatOfKeyPoint();
        featureDetector.detect(object, objectKeypoints);
        List<KeyPoint> objectKeypointsList = objectKeypoints.toList();
        MatOfKeyPoint sceneKeypoints = new MatOfKeyPoint();
        featureDetector.detect(scene, sceneKeypoints);
        List<KeyPoint> sceneKeypointsList = sceneKeypoints.toList();
        
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
        
        List<DMatch> goodMatches = new ArrayList<>();
        for (DMatch match : matchesList) {
            if (match.distance < 3 * minDistance) {
                goodMatches.add(match);
            }
        }
        MatOfDMatch goodMatchesMat = MatConverter.convert(goodMatches);
        
        Mat imageWithMatches = new Mat();
        Features2d.drawMatches(object, objectKeypoints, scene, sceneKeypoints,
                goodMatchesMat, imageWithMatches, Scalar.all(-1), Scalar.all(-1),
                new MatOfByte(), Features2d.NOT_DRAW_SINGLE_POINTS);
        
        
        List<Point> objPoints = new ArrayList<>();
        List<Point> scenePoints = new ArrayList<>();
        for (DMatch goodMatch : goodMatches) {
            objPoints.add(objectKeypointsList.get(goodMatch.queryIdx).pt);
            scenePoints.add(sceneKeypointsList.get(goodMatch.trainIdx).pt);
        }
        MatOfPoint2f objPointsMat = MatConverter.convertToMatOfPoint2f(objPoints);
        MatOfPoint2f scenePointsMat = MatConverter.convertToMatOfPoint2f(scenePoints);
        
        Mat homography = Calib3d.findHomography(objPointsMat, scenePointsMat, Calib3d.RANSAC, 3);
        List<Point> cornersInObject = new ArrayList<>();
        cornersInObject.add(new Point(0, 0));
        cornersInObject.add(new Point(object.cols(), 0));
        cornersInObject.add(new Point(object.cols(), object.rows()));
        cornersInObject.add(new Point(0, object.rows()));
        Mat cornersInObjectMat = MatConverter.convertToMatOfType(cornersInObject, CvType.CV_32FC2);
        
        Mat cornersInSceneMat = new Mat(4, 1, CvType.CV_32FC2);
        Core.perspectiveTransform(cornersInObjectMat, cornersInSceneMat, homography);
        List<Point> cornersInScene = MatConverter.toPointList(cornersInSceneMat);
        
        Core.line(
                imageWithMatches,
                PointUtils.add(cornersInScene.get(0), new Point(object.cols(), 0)),
                PointUtils.add(cornersInScene.get(1), new Point(object.cols(), 0)),
                new Scalar(0, 0, 255),
                4);
        Core.line(
                imageWithMatches,
                PointUtils.add(cornersInScene.get(1), new Point(object.cols(), 0)),
                PointUtils.add(cornersInScene.get(2), new Point(object.cols(), 0)),
                new Scalar(0, 255, 0),
                4);
        Core.line(
                imageWithMatches,
                PointUtils.add(cornersInScene.get(2), new Point(object.cols(), 0)),
                PointUtils.add(cornersInScene.get(3), new Point(object.cols(), 0)),
                new Scalar(255, 0, 255),
                4);
        Core.line(
                imageWithMatches,
                PointUtils.add(cornersInScene.get(3), new Point(object.cols(), 0)),
                PointUtils.add(cornersInScene.get(0), new Point(object.cols(), 0)),
                new Scalar(255, 0, 0),
                4);
        
        try {
            VisualHelper.showImageFrame(ImageConverter.convert(imageWithMatches, 640, 480));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}
