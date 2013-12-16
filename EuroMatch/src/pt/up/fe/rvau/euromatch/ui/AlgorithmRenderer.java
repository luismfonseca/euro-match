/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package pt.up.fe.rvau.euromatch.ui;

import java.awt.Component;
import java.util.ArrayList;
import java.util.Arrays;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.ListCellRenderer;
import pt.up.fe.rvau.euromatch.EuroMatch;

/**
 *
 * @author luiscubal
 */
class AlgorithmRenderer extends JLabel implements ListCellRenderer {

	private String[] strings;
	
	public AlgorithmRenderer(String[] strings) {
		this.strings = strings;
		
		setOpaque(true);
	}

	@Override
	public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
		if (isSelected) {
			setBackground(list.getSelectionBackground());
			setForeground(list.getSelectionForeground());
		} else {
			setBackground(list.getBackground());
			setForeground(list.getForeground());
		}
		
		int listIndex = indexOf(list, (int) value);
		setText(listIndex == -1 ? null : strings[listIndex]);
		
		return this;
	}
	
	private int indexOf(JList list, int value) {
		for (int i = 0; i < list.getModel().getSize(); ++i) {
			if ((int) list.getModel().getElementAt(i) == value) {
				return i;
			}
		}
		return -1;
	}
	
}
