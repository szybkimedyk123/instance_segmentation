import numpy as np
import cv2
import pyrealsense2 as rs
import torch


def get_frame(pipeline, align, temporal, spatial, colorizer):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    depth_frame = temporal.process(depth_frame)
    depth_frame = spatial.process(depth_frame)

    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image, colorized_depth


def mask_step(model, transforms, image):
    # image preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image)
    image = image.to(torch.device('cuda'))

    # instance segmentation
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    scores = output[0]["scores"].detach().cpu()
    all_labels = output[0]["labels"].detach().cpu()
    all_boxes = output[0]["boxes"].detach().cpu()
    all_masks = output[0]["masks"].detach().cpu()

    # Filter out low confidence detections
    keep_indices = (scores > 0.5).nonzero().squeeze()

    if keep_indices.numel() == 0:                       # nothing detected
        return None, None, None

    labels = []
    boxes = []
    masks = []
    if keep_indices.numel() == 1:
        labels.append(all_labels[0])
        boxes.append(all_boxes[0])
        masks.append(all_masks[0])
    else:
        for index in np.asarray(keep_indices):
            labels.append(all_labels[index])
            boxes.append(all_boxes[index])
            masks.append(all_masks[index])

    return labels, boxes, masks


def calculate_center_of_mass(vertices):
    n = len(vertices)
    cx = np.sum(vertices[:, 0]) / n
    cy = np.sum(vertices[:, 1]) / n
    cz = np.sum(vertices[:, 2]) / n
    center_of_mass3d = (cx, cy, cz)
    return center_of_mass3d


def bb3d_to_2d(image, aabb, init_mask):
    height, width, channels = image.shape
    vertices = np.array(aabb.get_box_points())
    center_of_mass3d = calculate_center_of_mass(vertices)
    arr_mask = np.asarray(init_mask.squeeze())

    # apply a binary threshold to arr_mask
    mask = cv2.threshold(arr_mask, 0.5, 1, cv2.THRESH_BINARY)[1]

    # GET Mask x and y min/max values
    y_vals, x_vals = np.nonzero(mask)
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)

    max_y_mask = np.max(y_vals)
    min_y_mask = np.min(y_vals)
    mask_center_x = (x_min+x_max) / 2
    mask_center_y = (y_min + y_max) / 2
    mask_x_length = abs(x_max - x_min)
    mask_y_length = abs(y_max - y_min)

    x_min = np.min(vertices[:, 0]) - center_of_mass3d[0]
    x_max = np.max(vertices[:, 0]) - center_of_mass3d[0]
    y_min = np.min(vertices[:, 1]) - center_of_mass3d[1]
    y_max = np.max(vertices[:, 1]) - center_of_mass3d[1]
    z_min = np.min(vertices[:, 2]) - center_of_mass3d[2]
    z_max = np.max(vertices[:, 2]) - center_of_mass3d[2]

    points_3d = np.array([
        [x_min, y_min, z_min],  # 0
        [x_max, y_min, z_min],  # 1
        [x_max, y_max, z_min],  # 2
        [x_min, y_max, z_min],  # 3
        [x_min, y_min, z_max],  # 4
        [x_max, y_min, z_max],  # 5
        [x_max, y_max, z_max],  # 6
        [x_min, y_max, z_max]  # 7
    ])

    x_3Dlength = abs(x_max - x_min)
    y_3Dlength = abs(y_max - y_min)
    z_3Dlength = abs(z_max - z_min)
    ytoxproperty = y_3Dlength/x_3Dlength
    ztoxproperty = z_3Dlength/x_3Dlength

    focal_length = 330
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])

    # Creating bounding box vertices in 2D
    points_2d, _ = cv2.projectPoints(points_3d, (0, 0, 0), (0, 0, 0), K, None)
    # corrected_points_2d = [tuple(map(int, point[0])) for point in points_2d]

    points_2d = [tuple(map(int, point[0])) for point in points_2d]

    center_of_mass = np.mean(points_2d, axis=0)

    difference_x = mask_center_x - center_of_mass[0]
    difference_y = mask_center_y - center_of_mass[1]

    # Searching for min max for 2d points
    x_min2d = y_min2d = float('inf')
    x_max2d = y_max2d = float('-inf')
    for point in points_2d:
        x, y = point
        if x < x_min2d:
            x_min2d = x
        if x > x_max2d:
            x_max2d = x
        if y < y_min2d:
            y_min2d = y
        if y > y_max2d:
            y_max2d = y

    x_min2d = x_min2d - center_of_mass[0]
    x_max2d = x_max2d - center_of_mass[0]
    y_min2d = y_min2d - center_of_mass[1]
    y_max2d = y_max2d - center_of_mass[1]

    if abs(x_min2d - x_max2d) < mask_x_length or abs(x_min2d - x_max2d) > mask_x_length+20:
        for i in range(len(points_2d)):
            if points_2d[i][0] == x_min2d+center_of_mass[0]:
                points_2d[i] = (int(-mask_x_length /2 - 0.1*mask_x_length),int(points_2d[i][1]))

            if points_2d[i][0] == x_max2d+center_of_mass[0]:
                points_2d[i] = (int(mask_x_length/2 + 0.1*mask_x_length),int(points_2d[i][1]))
        x_min2d = - mask_x_length / 2 - 20
        x_max2d = mask_x_length / 2 + 20
        center_of_mass = np.mean(points_2d, axis=0)

        difference_x = mask_center_x - center_of_mass[0]
        x_min2d = x_min2d - center_of_mass[0]
        x_max2d = x_max2d - center_of_mass[0]

    if abs(y_min2d - y_max2d) > mask_y_length+20:
        for i in range(len(points_2d)):
            if points_2d[i][1] == y_min2d + center_of_mass[1]:
                points_2d[i] = (int(points_2d[i][0]), int(-mask_y_length / 2 - 0.1 * mask_y_length))
            if points_2d[i][1] == y_max2d + center_of_mass[1]:
                points_2d[i] = (int(points_2d[i][0]), int(mask_y_length / 2 + 0.1 * mask_y_length))

        y_min2d = -mask_y_length / 2 - 20
        y_max2d = mask_y_length / 2 + 20
        center_of_mass = np.mean(points_2d, axis=0)
        difference_y = mask_center_y - center_of_mass[1]
        y_min2d = y_min2d - center_of_mass[1]
        y_max2d = y_max2d - center_of_mass[1]

    corrected_points_2d=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    scale_x = round(0.1*mask_x_length)
    scale_z = round(0.15*abs(z_max - z_min))
    rotation = mask_center_y/height
    oppos_rot = (1-rotation)*(1-rotation)

    diff = 0
    diff2 = 0
    for i in range(len(points_2d)):
        if i < 4:
            corrected_points_2d[i] = (int(points_2d[i][0] + difference_x - scale_x),
                                      int(points_2d[i][1] + difference_y + scale_z * oppos_rot * 0.7))
            if i > 1:
                corrected_points_2d[i] = (int(points_2d[i][0] + difference_x - scale_x),
                                          int(points_2d[i][1] + difference_y + scale_z * oppos_rot * 0.7
                                                              + rotation * mask_y_length))
            if i < 2:
                if int(corrected_points_2d[i][1]) > max_y_mask or int(corrected_points_2d[i][1]) < max_y_mask:
                    if i == 0:
                        diff = int(corrected_points_2d[i][1]) - max_y_mask
                        if diff > 5:
                            diff = diff - 5
                    corrected_points_2d[i] = (int(points_2d[i][0] + difference_x - scale_x),
                                              int(points_2d[i][1] + difference_y + scale_z * oppos_rot * 0.7 - diff))

        else:
            corrected_points_2d[i] = (int(points_2d[i][0] + difference_x + scale_x),
                                      int(points_2d[i][1] + difference_y - scale_z * rotation * 2))
            if i > 5:
                corrected_points_2d[i] = (int(points_2d[i][0] + difference_x + scale_x),
                                          int(points_2d[i][1] + difference_y - scale_z * rotation * 2
                                                              - rotation * mask_y_length - diff))
            if i == 6 or i == 7:
                if int(corrected_points_2d[i][1]) < min_y_mask:
                    if i == 6:
                        diff2 = min_y_mask - int(corrected_points_2d[i][1])
                    corrected_points_2d[i] = (int(points_2d[i][0] + difference_x + scale_x),
                                              int(points_2d[i][1] + difference_y - scale_z * rotation * 2
                                                  - rotation * mask_y_length - diff + diff2))
                    corrected_points_2d[5] = (int(points_2d[5][0] + difference_x + scale_x),
                                              int(points_2d[5][1] + difference_y - scale_z * rotation * 2 + diff2))
                    corrected_points_2d[4] = (int(points_2d[4][0] + difference_x + scale_x),
                                              int(points_2d[4][1] + difference_y - scale_z * rotation * 2 + diff2))

    # CHECKING IF Y LENGTH IS NOT TOO HIGH
    y_2Dlength = abs(int(points_2d[0][1] + difference_y + scale_z * oppos_rot * 0.7 - diff)
                     - int(points_2d[2][1] + difference_y + scale_z * oppos_rot * 0.7 + rotation * mask_y_length))
    x_2Dlength = abs(int(points_2d[0][0] + difference_x - scale_x)
                     - int(points_2d[1][0] + difference_x - scale_x))
    ytox2dproperty = y_2Dlength / x_2Dlength
    length_diff = round(ytox2dproperty * x_2Dlength) - round(ytoxproperty * x_2Dlength)

    if length_diff > 0:
        corrected_points_2d[2] = (
            int(corrected_points_2d[2][0]), int(corrected_points_2d[2][1] + length_diff * 2*(1 + rotation*2)))
        corrected_points_2d[3] = (
            int(corrected_points_2d[3][0]), int(corrected_points_2d[3][1] + length_diff * 2*(1 + rotation*2)))
        corrected_points_2d[4] = (
            int(corrected_points_2d[4][0]), int(corrected_points_2d[4][1] + length_diff * 2 * (1 + rotation * 2)))
        corrected_points_2d[5] = (
            int(corrected_points_2d[5][0]), int(corrected_points_2d[5][1] + length_diff * 2 * (1 + rotation * 2)))

    cv2.line(image, corrected_points_2d[0], corrected_points_2d[1], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[1], corrected_points_2d[2], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[2], corrected_points_2d[3], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[3], corrected_points_2d[0], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[4], corrected_points_2d[5], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[5], corrected_points_2d[6], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[6], corrected_points_2d[7], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[7], corrected_points_2d[4], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[0], corrected_points_2d[6], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[1], corrected_points_2d[7], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[2], corrected_points_2d[4], (255, 0, 0), thickness=2)
    cv2.line(image, corrected_points_2d[3], corrected_points_2d[5], (255, 0, 0), thickness=2)

    return image
