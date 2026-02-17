# lfa/base.py
import numpy as np
import cv2


class SimpleLFAAnalyzer:
    """
    Simple LFA Analyzer for cropped images (detection zone only, no wick)

    Works by:
    1. Splitting image in half (top = control, bottom = test)
    2. Finding darkest line in each half (pink lines are darker than white background)
    3. Measuring intensity and calculating relative intensity

    NOTE:
    - Heavy image processing is moved to image_processing.py
    - The analysis pipeline is moved to analysis.py (run_analysis)
    - Visualization is moved to viz.py
    """

    def __init__(self, image_path):
        """Initialize with image path"""
        self.image_path = image_path
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.gray_image = None
        self.inverted_image = None
        self.top_half = None
        self.bottom_half = None
        self.control_line_pos = None
        self.test_line_pos = None
        self.control_intensity = None
        self.test_intensity = None
        self.background = None
        self.relative_intensity = None
        self.is_negative = False

        self.background_image = None
        self.corrected_image = None
        self.otsu_threshold = None
        self.binary_mask = None
        self._rowwise_debug = None

    # -------------------------------------------------------------------------
    # BASE UTILS (kept here because they’re “core” and used by analysis/viz)
    # -------------------------------------------------------------------------

    def split_halves(self, use_corrected=False):
        """Split image into top (control) and bottom (test) halves"""

        if use_corrected:
            if self.corrected_image is None:
                raise ValueError("Run subtract_background() first or set use_corrected=False.")
            img = self.corrected_image
        else:
            img = self.inverted_image

        height = img.shape[0]
        mid_point = height // 2
        # Skip first 3 pixels of each half to avoid any marking lines
        skip = 3

        self.top_half = img[skip:mid_point - skip, :]
        self.bottom_half = img[mid_point + skip:height - skip, :]

        return self.top_half, self.bottom_half

    def find_darkest_line(self, image_half, half_name=""):
        """
        Find the darkest horizontal line (highest intensity after inversion)

        Returns: (line_position, line_intensity, intensity_profile)
        """
        # Average intensity across each row
        intensity_profile = np.mean(image_half, axis=1)

        # Find maximum intensity (darkest line in original)
        max_pos = np.argmax(intensity_profile)
        max_intensity = intensity_profile[max_pos]

        return max_pos, max_intensity, intensity_profile

    # -------------------------------------------------------------------------
    # DELEGATES (keep the nice class API, but do the work in other files)
    # -------------------------------------------------------------------------

    def analyze(self, *args, **kwargs):
        """Delegates to lfa/analysis.py so analysis() stays in one place."""
        from .analysis import run_analysis
        return run_analysis(self, *args, **kwargs)

    def visualize(self, *args, **kwargs):
        """Delegates to lfa/viz.py (keeps your old an.visualize() call)."""
        from .visualization import visualize_summary
        return visualize_summary(self, *args, **kwargs)