package pt.up.fe.rvau.euromatch.jutils;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.imageio.ImageIO;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

/**
 * This is an auxiliary to help with the Java bindings with OpenCV.
 * @author luiscubal, luisfonseca
 */
public class ImageConverter {
    private ImageConverter() {}
    
    public static BufferedImage convert(Mat image, int width, int height) throws IOException {
        Mat dest = new Mat();
        Imgproc.resize(image, dest, new Size(width, height));
        
        return convert(dest);
    }
    
    public static BufferedImage convert(Mat image) throws IOException {
        MatOfByte outputMat = new MatOfByte();
        Highgui.imencode(".png", image, outputMat);
        byte[] bytes = outputMat.toArray();
        try (InputStream input = new ByteArrayInputStream(bytes)) {
            return ImageIO.read(input);
        }
    }
}
