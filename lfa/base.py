# lfa/base.py
import cv2


class SimpleLFAAnalyzer:
    """
    LFA Analyzer using the rowwise pipeline.

    Pipeline:
        preprocess() -> subtract_background() -> rowwise_binarize_corrected() -> classify_two_band_top_bottom()

    All heavy processing lives in image_processing.py.
    Analysis pipeline lives in analysis.py.
    Visualization lives in viz.py.
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Set by preprocess()
        self.gray_image = None
        self.inverted_image = None

        # Set by subtract_background()
        self.background_image = None
        self.corrected_image = None

        # Set by rowwise_binarize_corrected()
        self.binary_mask = None
        self._rowwise_debug = None

        # Not used by rowwise but kept for viz compat
        self.otsu_threshold = None

    def analyze(self, *args, **kwargs):
        """Delegates to lfa/analysis.py"""
        from .analysis import run_analysis
        return run_analysis(self, *args, **kwargs)

    def visualize(self, *args, **kwargs):
        """Delegates to lfa/viz.py"""
        from .visualization import visualize_summary
        return visualize_summary(self, *args, **kwargs)
