/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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
}
