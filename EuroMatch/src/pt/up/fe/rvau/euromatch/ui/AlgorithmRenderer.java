package pt.up.fe.rvau.euromatch.ui;

import java.awt.Component;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.ListCellRenderer;

/**
 *
 * @author luiscubal, luisfonseca
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
