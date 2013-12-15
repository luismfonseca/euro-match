/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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
	private final List<Point> points;
	
	public DetectedBill(List<Point> points) {
		if (points == null) {
			throw new IllegalArgumentException("points must not be null");
		}
		if (points.size() != 4) {
			throw new IllegalArgumentException("points must have size 4");
		}
		this.points = new ArrayList<>(points);
	}
	
	public List<Point> getPoints() {
		return Collections.unmodifiableList(points);
	}
}
