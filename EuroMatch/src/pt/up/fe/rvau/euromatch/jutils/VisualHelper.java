package pt.up.fe.rvau.euromatch.jutils;

import java.awt.BorderLayout;
import java.awt.image.BufferedImage;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

/**
 *
 * @author luiscubal, luisfonseca
 */
public class VisualHelper {
    public static void showImageFrame(BufferedImage image, int totalMoney) {
        JFrame frame = new JFrame();
		frame.setLayout(new BorderLayout());
		frame.add(new JLabel("Total: " + totalMoney), BorderLayout.NORTH);
        frame.add(new JLabel(new ImageIcon(image)), BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);
    }
}
