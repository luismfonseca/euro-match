package pt.up.fe.rvau.euromatch;

import java.util.List;
import org.opencv.core.Mat;

/**
 * This class holds the result of testing to match notes to an image.
 * It contains a list of detected bills, resulting image, total value and the elapsed time.
 * @author luiscubal, luisfonseca
 */
public class DetectedBillsResult {
	public List<DetectedBill> detectedBills;
	public Mat outputImage;
	public int totalValue;
    public double elapsedTime;
}
