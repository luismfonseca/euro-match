package pt.up.fe.rvau.euromatch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.opencv.core.Point;

/**
 *
 * @author luiscubal
 */
public class DetectedBill {
	private final int value;
	private final List<Point> points;
	
	public DetectedBill(int value, List<Point> points) {
		if (points == null) {
			throw new IllegalArgumentException("points must not be null");
		}
		if (points.size() != 4) {
			throw new IllegalArgumentException("points must have size 4");
		}
		this.value = value;
		this.points = new ArrayList<>(points);
	}
	
	public int getValue() {
		return value;
	}
	
	public List<Point> getPoints() {
		return Collections.unmodifiableList(points);
	}
}
