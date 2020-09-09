Kinetic branch of camera_calibration package in image_pipeline from ROS

Improvements
1. Added manual saving mode to reduce motion blur and control chessboard target distribution.
2. Added burst saving mode to increase image number.
3. Added show board pose in image view.
4. Corrected the error in refine corners.
5. print reprojection error and output board pose into yaml
6. threshold the motion blur by image different within board corners.
7. show 3d board in real-time
8. Added image saving mode, save all images within threshold.

Possible todos
1. Add a save image mode to save all original images.