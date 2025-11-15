import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
import numpy as np
import sys


class RangeAngleHeatmap(QtWidgets.QMainWindow):
    """
    Live-updating range-angle heatmap (e.g., RAI map visualization).
    """

    def __init__(self, range_res: float, angle_bins: int):
        super().__init__()

        self.range_res = range_res
        self.setWindowTitle("Range–Angle Heatmap")

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Create image item for fast 2D updates
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)

        self.plot_widget.setLabel("left", "Range (m)")
        self.plot_widget.setLabel("bottom", "Angle Bin")
        self.plot_widget.invertY(False)  # top = near range
        self.plot_widget.showGrid(x=True, y=True)

        # Set a color map (you can try 'plasma', 'viridis', etc.)
        cmap = pg.colormap.get("inferno")
        self.img_item.setLookupTable(cmap.getLookupTable())

        # Optionally scale axes to real units later
        self.angle_bins = angle_bins

    def update(self, rai_map: np.ndarray, distances: np.ndarray):
        """
        Args:
            rai_map (np.ndarray): 2D array [range_bins, angle_bins].
        """
        if rai_map is None or rai_map.size == 0:
            return

        # Convert to dB for visibility
        magnitude = 20 * np.log10(np.abs(rai_map) + 1e-6)

        # Define bounding box: [x0, y0, width, height]
        angle_start = -90
        angle_end = 90
        range_start = distances[0]
        range_end = distances[-1]

        self.img_item.setImage(
            magnitude.T,
            autoLevels=True,
            rect=QtCore.QRectF(angle_start, range_start, angle_end - angle_start, range_end - range_start)
        )

        # Adjust axes
        self.plot_widget.setXRange(-90, 90)  # assuming angle bins correspond to -90 to +90 degrees
        self.plot_widget.setYRange(np.min(distances) - 0.1, np.max(distances) + 0.1)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    heatmap = RangeAngleHeatmap(range_res=0.1, angle_bins=64)
    heatmap.resize(800, 600)
    heatmap.show()

    # Simulate incoming frames
    timer = QtCore.QTimer()

    def update_frame():
        # Generate mock range-angle data
        rng_bins = 128
        ang_bins = 64
        data = np.abs(np.random.randn(rng_bins, ang_bins)) ** 2
        heatmap.update(data)

    timer.timeout.connect(update_frame)
    timer.start(100)  # update every 100 ms

    sys.exit(app.exec())
