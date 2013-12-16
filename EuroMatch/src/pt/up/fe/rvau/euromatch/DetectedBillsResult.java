package pt.up.fe.rvau.euromatch;

import java.util.List;
import org.opencv.core.Mat;

/**
 *
 * @author luiscubal
 */
public class DetectedBillsResult {
	public List<DetectedBill> detectedBills;
	public Mat outputImage;
	public int totalValue;
    public double elapsedTime;
}
