#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import yaml

from numpy import linspace
import matplotlib.pyplot as plt

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1,1] = 0
    M[1,2] = 1
    M[2,1] = -1
    M[2,2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))

def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4,5))
    X_img_plane[0:3,0] = [-width, height, f_scale]
    X_img_plane[0:3,1] = [width, height, f_scale]
    X_img_plane[0:3,2] = [width, -height, f_scale]
    X_img_plane[0:3,3] = [-width, -height, f_scale]
    X_img_plane[0:3,4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4,3))
    X_triangle[0:3,0] = [-width, -height, f_scale]
    X_triangle[0:3,1] = [0, -2*height, f_scale]
    X_triangle[0:3,2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4,2))
    X_center1[0:3,0] = [0, 0, 0]
    X_center1[0:3,1] = [-width, height, f_scale]

    X_center2 = np.ones((4,2))
    X_center2[0:3,0] = [0, 0, 0]
    X_center2[0:3,1] = [width, height, f_scale]

    X_center3 = np.ones((4,2))
    X_center3[0:3,0] = [0, 0, 0]
    X_center3[0:3,1] = [width, -height, f_scale]

    X_center4 = np.ones((4,2))
    X_center4[0:3,0] = [0, 0, 0]
    X_center4[0:3,1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

def create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
    width = board_width*square_size
    height = board_height*square_size

    # draw calibration board
    X_board = np.ones((4,5))
    #X_board_cam = np.ones((extrinsics.shape[0],4,5))
    X_board[0:3,0] = [0,0,0]
    X_board[0:3,1] = [width,0,0]
    X_board[0:3,2] = [width,height,0]
    X_board[0:3,3] = [0,height,0]
    X_board[0:3,4] = [0,0,0]

    # draw board frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [height/2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, height/2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, height/2]

    if draw_frame_axis:
        return [X_board, X_frame1, X_frame2, X_frame3]
    else:
        return [X_board]

def draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scale_focal,
                       extrinsics, board_width, board_height, square_size,
                       patternCentric):
    from matplotlib import cm

    min_values = np.zeros((3,1))
    min_values = np.inf
    max_values = np.zeros((3,1))
    max_values = -np.inf

    if patternCentric:
        X_moving = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
        X_static = create_board_model(extrinsics, board_width, board_height, square_size)
    else:
        X_static = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
        X_moving = create_board_model(extrinsics, board_width, board_height, square_size)

    cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [ cm.jet(x) for x in cm_subsection ]

    for i in range(len(X_static)):
        X = np.zeros(X_static[i].shape)
        for j in range(X_static[i].shape[1]):
            X[:,j] = transform_to_matplotlib_frame(np.eye(4), X_static[i][:,j])
        ax.plot3D(X[0,:], X[1,:], X[2,:], color='r')
        min_values = np.minimum(min_values, X[0:3,:].min(1))
        max_values = np.maximum(max_values, X[0:3,:].max(1))

    for idx in range(extrinsics.shape[0]):
        R, _ = cv.Rodrigues(extrinsics[idx,0:3])
        cMo = np.eye(4,4)
        cMo[0:3,0:3] = R
        cMo[0:3,3] = extrinsics[idx,3:6]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4,j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4,j], patternCentric)
            ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3,:].min(1))
            max_values = np.maximum(max_values, X[0:3,:].max(1))

    return min_values, max_values

def main():
    import argparse
    

    parser = argparse.ArgumentParser(description='Plot camera calibration extrinsics.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--calibration', type=str, default='left_intrinsics.yml',
                        help='YAML camera calibration file.')
    parser.add_argument('--cam_width', type=float, default=0.064/2,
                        help='Width/2 of the displayed camera.')
    parser.add_argument('--cam_height', type=float, default=0.048/2,
                        help='Height/2 of the displayed camera.')
    parser.add_argument('--scale_focal', type=float, default=40,
                        help='Value to scale the focal length.')
    parser.add_argument('--patternCentric', action='store_true',
                        help='The calibration board is static and the camera is moving.')
    args = parser.parse_args()

    with open(args.calibration) as fp:
        calibration_file = yaml.load(fp)
    image_width = int(calibration_file['image_width'])
    image_height = int(calibration_file['image_height'])
    board_width = 6
    board_height = 9
    square_size = 0.74
    cam_mat = calibration_file['camera_matrix']
    camera_matrix = np.array(cam_mat['data']).reshape((cam_mat['rows'], cam_mat['cols']))
    r_mat = calibration_file['rvecs']
    rvecs = np.array(r_mat['data']).reshape((r_mat['rows'], r_mat['cols']))
    t_mat = calibration_file['tvecs']
    tvecs = np.array(t_mat['data']).reshape((t_mat['rows'], t_mat['cols']))
    extrinsics = np.concatenate([rvecs, tvecs], axis=1)
    assert extrinsics.shape[1] == 6
    
    # fs = cv.FileStorage(cv.samples.findFile(args.calibration), cv.FILE_STORAGE_READ)
    # board_width = int(fs.getNode('board_width').real())
    # board_height = int(fs.getNode('board_height').real())
    # square_size = fs.getNode('square_size').real()
    # camera_matrix = fs.getNode('camera_matrix').mat()
    # extrinsics = fs.getNode('extrinsic_parameters').mat()

    cam_width = args.cam_width
    cam_height = args.cam_height
    scale_focal = args.scale_focal
    pattern_centric = args.patternCentric
    fig = plt.figure(figsize=(10,12))
    plot_extrinsics(fig, camera_matrix, extrinsics, board_width, board_height, square_size)

def plot_extrinsics(fig, camera_matrix, extrinsics, board_width, board_height, square_size,
                    cam_width = 0.064/2, cam_height=0.048/2, scale_focal=40, pattern_centric=False, get_image=False):
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    # fig = plt.figure(figsize=(10,12))
    # 
    ax = fig.gca(projection='3d')
    ax.cla()
    ax.set_aspect("equal")

    min_values, max_values = draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, pattern_centric)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')

    if get_image:
        canvas = FigureCanvas(fig)
        ax.view_init(elev=90., azim = -90.)

        canvas.draw()       # draw the canvas, cache the renderer
        s, (width, height) = canvas.print_to_buffer()
        image1 = np.fromstring(s, np.uint8).reshape((height, width, 4))
        

        # winname = 'temp'
        # cv.namedWindow(winname, cv.WINDOW_GUI_NORMAL)
        # cv.imshow(winname, image)
        # cv.waitKey(0)
        # cv.destroyWindow(winname)

        ax.view_init(elev=0., azim = 0.)

        canvas.draw()       # draw the canvas, cache the renderer
        s, (width, height) = canvas.print_to_buffer()
        image2 = np.fromstring(s, np.uint8).reshape((height, width, 4))

        # winname = 'temp'
        # cv.namedWindow(winname, cv.WINDOW_GUI_NORMAL)
        # cv.imshow(winname, image)
        # cv.waitKey(0)
        # cv.destroyWindow(winname)
        return image1[:,:,0:3], image2[:,:,0:3]

    else:
        plt.show()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
