# LFA-imaging

Python-based image analysis pipeline developed as part of a bioengineering capstone project in the Kamei Lab at UCLA. The repository processes lateral flow assay (LFA) strip images captured from a smartphone and performs automated quantitative analysis to extract relative signal intensities from test bands. The project was designed to support rapid, low-cost point-of-care diagnostics by integrating image processing, cloud computing, and automated hardware control.

The pipeline uses OpenCV and NumPy-based image processing techniques including image inversion, background correction via morphological opening, row-wise thresholding, band detection, and intensity quantification. The repository also includes visualization tools for debugging and assay validation, allowing users to inspect corrected images, binary masks, and detected regions of interest.

This repo serves as the backend analytical engine for a larger end-to-end diagnostic workflow involving:
1. A SwiftUI iOS application for image capture and user interaction
2. A FastAPI backend deployed on a Google Cloud VM
3. BLE communication with an ESP32-controlled automated assay device
4. Firebase integration for assay result storage and patient tracking

The goal of the project is to create an accessible and scalable platform for automated biomarker detection using smartphone-based imaging and cloud-enabled analysis.
