package pt.up.fe.rvau.euromatch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.opencv.core.Point;

/**
 *
 * @author luiscubal, luisfonseca
 */
public class DetectedBill {
	private final BillInfo billInfo;
	private final List<Point> points;
	
	public DetectedBill(BillInfo billInfo, List<Point> points) {
		if (points == null) {
			throw new IllegalArgumentException("points must not be null");
		}
		if (points.size() != 4) {
			throw new IllegalArgumentException("points must have size 4");
		}
		this.billInfo = billInfo;
		this.points = new ArrayList<>(points);
	}
	
	public int getValue() {
		return billInfo.getValue();
	}
	
	public BillInfo getBillInfo() {
		return billInfo;
	}
	
	public List<Point> getPoints() {
		return Collections.unmodifiableList(points);
	}
}
