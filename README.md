Kinetic branch of camera_calibration package in image_pipeline from ROS

Added manual saving mode to reduce motion blur and control chessboard target distribution.
Added burst saving mode to increase image number.
Added show board pose in image view.

Corrected the error in refine corners.
print reprojection error and output board pose into yaml
threshold the motion blur by image different within board corners.