/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch.jutils;

import java.awt.image.BufferedImage;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

/**
 *
 * @author luiscubal
 */
public class VisualHelper {
    public static void showImageFrame(BufferedImage image) {
        JFrame frame = new JFrame();
        frame.add(new JLabel(new ImageIcon(image)));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }
}
