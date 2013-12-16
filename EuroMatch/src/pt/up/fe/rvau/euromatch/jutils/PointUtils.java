package pt.up.fe.rvau.euromatch.jutils;


import org.opencv.core.Point;

/**
 *
 * @author luiscubal, luisfonseca
 */
public class PointUtils {
    private PointUtils() {}
    
    public static Point add(Point point1, Point point2) {
        return new Point(point1.x + point2.x, point1.y + point2.y);
    }
}
