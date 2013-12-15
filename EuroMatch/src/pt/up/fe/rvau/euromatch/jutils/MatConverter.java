/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch.jutils;

import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.KeyPoint;

/**
 *
 * @author luiscubal
 */
public class MatConverter {
    private MatConverter() {}
    
    public static MatOfDMatch convert(List<DMatch> src) {
        return new MatOfDMatch(src.toArray(new DMatch[src.size()]));
    }
    
    public static MatOfPoint2f convertToMatOfPoint2f(List<Point> src) {
        return new MatOfPoint2f(src.toArray(new Point[src.size()]));
    }
	
	public static MatOfKeyPoint convertToMatOfKeyPoint(List<KeyPoint> src) {
		return new MatOfKeyPoint(src.toArray(new KeyPoint[src.size()]));
	}
    
    public static Mat convertToMatOfType(List<Point> points, int cvType) {
        Mat mat = new Mat(4, 1, cvType);
        for (int i = 0; i < points.size(); ++i) {
            Point point = points.get(i);
            mat.put(i, 0, point.x, point.y);
        }
        return mat;
    }
    
    public static List<Point> toPointList(Mat srcMat) {
        return new MatOfPoint2f(srcMat).toList();
    }
}
