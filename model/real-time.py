from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
import torch
import open3d as o3d

import pyrealsense2 as rs
import numpy as np
import cv2

import time
import os

from dataset import LabDataset
from maskrcnn import MaskRCNN


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


def get_frame(pipeline, align, temporal, spatial):
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

def calculate_center_of_mass(vertices):
    n = len(vertices)
    cx = np.sum(vertices[:, 0]) / n
    cy = np.sum(vertices[:, 1]) / n
    cz = np.sum(vertices[:, 2]) / n
    center_of_mass3d = (cx, cy, cz)
    return center_of_mass3d
def bb3d_to_2d(image,aabb,mask):
    # PART OF MAKING 2D VERTICES FROM 3D VERTICES

    # Get image shapes
    # height=720
    # width=1280
    height, width, channels = image.shape
    #print(height,width)
    vertices = np.array(aabb.get_box_points())

    center_of_mass3d=calculate_center_of_mass(vertices)
    #print("center of mass 3d",center_of_mass3d)
    print("mask",mask)
    arr_mask = np.asarray(mask.squeeze())
    print(type(arr_mask))
    # apply a binary threshold to arr_mask
    arr_mask = cv2.threshold(arr_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    print(arr_mask.shape)
    mask=arr_mask
    # GET Mask x and y min/max values
    y_vals, x_vals = np.nonzero(mask)
    x_min, x_max = np.min(x_vals), np.max(x_vals)
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    # Showing results
    max_y_mask=np.max(y_vals)
    min_y_mask=np.min(y_vals)
    # print("x_min mask:", x_min)
    # print("x_max mask:", x_max)
    # print("y_min mask:", y_min)
    # print("y_max mask:", y_max)
    mask_center_x=(x_min+x_max)/2
    mask_center_y = (y_min + y_max) / 2
    # print("mask x center",mask_center_x)
    # print("mask y center",mask_center_y)
    mask_x_length=abs(x_max - x_min)
    mask_y_length=abs(y_max - y_min)

    x_min = np.min(vertices[:, 0]) - center_of_mass3d[0]
    x_max = np.max(vertices[:, 0]) - center_of_mass3d[0]
    y_min = np.min(vertices[:, 1]) - center_of_mass3d[1]
    y_max = np.max(vertices[:, 1]) - center_of_mass3d[1]
    z_min = np.min(vertices[:, 2]) - center_of_mass3d[2]
    z_max = np.max(vertices[:, 2]) - center_of_mass3d[2]

    # print("x_min",x_min)
    # print("x_max", x_max)
    # print("y_min", y_min)
    # print("y_max", y_max)
    # print("z_min", z_min)
    # print("z_max",z_max)

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
    ytoxproperty=y_3Dlength/x_3Dlength
    ztoxproperty=z_3Dlength/x_3Dlength
    # print("y to x property ", ytoxproperty)
    # print("z to x property ", ztoxproperty)
    # transformation matrix
    #180*mask_x_length/210
    #round(100+(mask_center_y/100))
    focal_length = 330
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])
    #intrinsics.intrinsic_matrix= intrinsics.intrinsic_matrix/10
    #K= intrinsics.intrinsic_matrix
    # Creating bounding box vertices in 2D
    points_2d, _ = cv2.projectPoints(points_3d, (0, 0, 0), (0, 0, 0), K, None)
    #corrected_points_2d = [tuple(map(int, point[0])) for point in points_2d]

    points_2d = [tuple(map(int, point[0])) for point in points_2d]

    #print("points_2d",points_2d)
    center_of_mass = np.mean(points_2d, axis=0)
    #print("center of mass", center_of_mass)
    difference_x=mask_center_x-center_of_mass[0]
    difference_y=mask_center_y-center_of_mass[1]
    # print("difference x :",difference_x)
    # print("difference y :",difference_y)

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
    # Showing results
    x_min2d = x_min2d - center_of_mass[0]
    x_max2d = x_max2d - center_of_mass[0]
    y_min2d = y_min2d - center_of_mass[1]
    y_max2d = y_max2d - center_of_mass[1]
    # print("2dx_min", x_min2d)
    # print("2dx_max", x_max2d)
    # print("2dy_min", y_min2d)
    # print("2dy_max", y_max2d)
    if abs(x_min2d-x_max2d)<mask_x_length or abs(x_min2d-x_max2d) > mask_x_length+20:
        for i in range(len(points_2d)):
            #x,y=point
            if points_2d[i][0]==x_min2d+center_of_mass[0]:
                points_2d[i]=(int(-mask_x_length /2 - 0.1*mask_x_length),int(points_2d[i][1]))
                #point=x,y
            if points_2d[i][0]==x_max2d+center_of_mass[0]:
                points_2d[i]=(int(mask_x_length/2 + 0.1*mask_x_length),int(points_2d[i][1]))
        x_min2d = -mask_x_length / 2 - 20
        x_max2d = mask_x_length / 2 +  20
        center_of_mass = np.mean(points_2d, axis=0)
        # print("center of mass", center_of_mass)
        difference_x = mask_center_x - center_of_mass[0]
        #difference_y = mask_center_y - center_of_mass[1]
        x_min2d = x_min2d - center_of_mass[0]
        x_max2d = x_max2d - center_of_mass[0]
        #y_min2d = y_min2d - center_of_mass[1]
        #y_max2d = y_max2d - center_of_mass[1]
    #elif abs(x_min2d-x_max2d) > 1.5*mask_x_length:

    if abs(y_min2d - y_max2d) > mask_y_length+20:
        #print("spelniony warunek")
        for i in range(len(points_2d)):
            # x,y=point
            if points_2d[i][1] == y_min2d + center_of_mass[1]:
                points_2d[i] = ( int(points_2d[i][0]), int(-mask_y_length / 2 - 0.1 * mask_y_length))
                # point=x,y
            if points_2d[i][1] == y_max2d + center_of_mass[1]:
                points_2d[i] = (int(points_2d[i][0]), int(mask_y_length / 2 + 0.1 * mask_y_length))
        y_min2d = -mask_y_length / 2 - 20
        y_max2d = mask_y_length / 2 + 20
        center_of_mass = np.mean(points_2d, axis=0)
        # print("center of mass", center_of_mass)
        #difference_x = mask_center_x - center_of_mass[0]
        difference_y = mask_center_y - center_of_mass[1]
        #x_min2d = x_min2d - center_of_mass[0]
        #x_max2d = x_max2d - center_of_mass[0]
        y_min2d = y_min2d - center_of_mass[1]
        y_max2d = y_max2d - center_of_mass[1]
        # print("poprawione 2dx_min", x_min2d)
        # print("poprawione 2dx_max", x_max2d)
        # print("poprawione 2dy_min", y_min2d)
        # print("poprawione 2dy_max", y_max2d)
    #     y_min2d = -mask_y_length / 2
    #     y_max2d = mask_y_length / 2

    corrected_points_2d=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    # Scale x to rotate a little front and back surface of bounding box, so we can see better it's 3d
    #scale_x=round((abs(x_min2d)+abs(x_max2d))*0.06)
    scale_x = round(0.1*mask_x_length)
    #if mask_center_x < width/2:
    #    scale_x=-scale_x
    # Scale y also makes us able to rotate visualization of bounding box, value should be depending on depth,(so how
    # close are we to the object), and length of the object
    #scale_y=round((abs(z_max)+abs(z_min))*0.2)
    scale_z = round(0.15*abs(z_max - z_min))
    rotation=mask_center_y/height
    #rotation=rotation*rotation
    oppos_rot=(1-rotation)*(1-rotation)
    #rotation=0
    diff=0
    diff2=0
    #print("rotation",rotation)
    for i in range(len(points_2d)):
        if i<4:
            corrected_points_2d[i] = (int(points_2d[i][0] +difference_x-scale_x), int(points_2d[i][1]+difference_y+scale_z*oppos_rot*0.7)) #0.7*scale_y
            if i>1 :
                corrected_points_2d[i] = (int(points_2d[i][0] + difference_x - scale_x),
                                          int(points_2d[i][1] + difference_y + scale_z*oppos_rot*0.7 + rotation*mask_y_length))
            if i<2 :
                if int(corrected_points_2d[i][1]) > max_y_mask or int(corrected_points_2d[i][1])< max_y_mask : # set difference and adjust points if
                    # front BB low side points are lower than object the lowest point in mask
                    #print("spelniony warunek")
                    if i==0:
                        diff = int(corrected_points_2d[i][1]) - max_y_mask
                        if diff >5 :
                            diff = diff - 5
                    #print("corr points y",int(corrected_points_2d[i][1]) )
                    #print("y_min2d",max_y_mask)
                    #print ("diff ",diff)
                    corrected_points_2d[i] = (int(points_2d[i][0] + difference_x - scale_x ),
                                              int(points_2d[i][1] + difference_y + scale_z * oppos_rot * 0.7 - diff))

        else:
            corrected_points_2d[i] = (int(points_2d[i][0] + difference_x+scale_x), int(points_2d[i][1] + difference_y-scale_z*rotation*2)) #1.7*scale_y-20
            if i>5:
                corrected_points_2d[i] = (int(points_2d[i][0] + difference_x + scale_x),
                                          int(points_2d[i][1] + difference_y - scale_z*rotation*2 - rotation*mask_y_length - diff))
            if i==6 or i==7:
                if int(corrected_points_2d[i][1]) < min_y_mask :
                    #print("spelniony warunek")
                    #print("min_y mask",min_y_mask)
                    #print("corr points2d",int(corrected_points_2d[i][1]) )
                    if i==6:
                        diff2 = min_y_mask - int(corrected_points_2d[i][1])
                    #print("diff2",diff2)
                    corrected_points_2d[i] = (int(points_2d[i][0] + difference_x + scale_x),
                                             int(points_2d[i][1] + difference_y - scale_z*rotation*2- rotation*mask_y_length - diff +diff2 ))  # 0.7*scale_y
                    corrected_points_2d[5] = (int(points_2d[5][0] + difference_x + scale_x),
                                              int(points_2d[5][1] + difference_y - scale_z * rotation * 2 + diff2))
                    corrected_points_2d[4] = (int(points_2d[4][0] + difference_x + scale_x),
                                              int(points_2d[4][1] + difference_y - scale_z * rotation * 2 + diff2))
            #corrected_points_2d[i] = (int(points_2d[i][0] + difference_x + scale_x), int(points_2d[i][1] + difference_y - 1.5 * scale_y - 20))
            #or
            #corrected_points_2d[i] = (int(points_2d[i][0] + difference_x + scale_x), int(points_2d[i][1] + difference_y - 2 * scale_y ))
    #print("corrected points_2d", corrected_points_2d)

    # CHECKING IF Y LENGTH IS NOT TOO HIGH
    y_2Dlength = abs(int(points_2d[0][1] + difference_y + scale_z * oppos_rot * 0.7 - diff) - int(
        points_2d[2][1] + difference_y + scale_z * oppos_rot * 0.7 + rotation * mask_y_length))
    x_2Dlength = abs(
        int(points_2d[0][0] + difference_x - scale_x) - int(points_2d[1][0] + difference_x - scale_x))
    # print("y2dlength",y_2Dlength)
    # print("x2dlength",x_2Dlength)
    ytox2dproperty = y_2Dlength / x_2Dlength
    #print("ytoxprop2d", ytox2dproperty)
    length_diff = round(ytox2dproperty * x_2Dlength) - round(ytoxproperty * x_2Dlength)
    if length_diff > 0:
        #print("obnizamy lekko")
        corrected_points_2d[2] = (
            int(corrected_points_2d[2][0]), int(corrected_points_2d[2][1] + length_diff * 2*(1 + rotation*2)))
        corrected_points_2d[3] = (
            int(corrected_points_2d[3][0]), int(corrected_points_2d[3][1] + length_diff * 2*(1 + rotation*2)))
        corrected_points_2d[4] = (
            int(corrected_points_2d[4][0]), int(corrected_points_2d[4][1] + length_diff * 2 * (1 + rotation * 2)))
        corrected_points_2d[5] = (
            int(corrected_points_2d[5][0]), int(corrected_points_2d[5][1] + length_diff * 2 * (1 + rotation * 2)))

    # DRAWING ONE BB3D IN IMAGE BASED ON VERTICES 2D
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

    #SHOWING RESULT IMAGE

    return image

if __name__ == "__main__":
    # const init
    categories = {
        "miska2d":      0,
        "kubek2d":      1,
        "pudelko2d":    2,
        "czapka2d":     3,
        "but2d":        4,
        "kieliszek2d":  5,
        "lampka2d":     6
    }
    root_dir = '/home/lab/ZSD-IS/repo/zsd-is/model/mask_v2'
    checkpoint_path = '/home/lab/ZSD-IS/dataset/checkpoint_2023-05-31_13-28-04.pth'
    transforms = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # model initiation
    model = MaskRCNN(root_dir=root_dir)
    model.load_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()
    # get intrinsic matrix from bag TODO: get from stream
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open('/home/lab/ZSD-IS/repo/zsd-is/model/BB3D from depth/filtering/testbag.bag')
    intrinsics = bag_reader.metadata.intrinsics

    # RealSense initiation
    colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)
    pipeline = rs.pipeline()
    config = rs.config()
    resolution_width = 1280
    resolution_height = 720
    config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.rgb8, 30)
    pipeline.start(config)
    temporal = rs.temporal_filter(smooth_alpha=0.05, smooth_delta=20, persistence_control=8)
    spatial = rs.spatial_filter(smooth_alpha=0.5, smooth_delta=20, magnitude=2, hole_fill=0)
    # # create axes
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)
    # create image plot
    im1 = ax1.imshow(np.zeros((resolution_height, resolution_width)))
    im2 = ax2.imshow(np.zeros((resolution_height, resolution_width)))
    im3 = ax3.imshow(np.zeros((resolution_height, resolution_width)), cmap='gray', vmin=0, vmax=255)
    im4 = ax4.imshow(np.zeros((resolution_height, resolution_width)))

    plt.ion()
    try:
        while True:
            # image = cv2.imread('/home/lab/ZSD-IS/repo/zsd-is/dataset/96.png')  # None  # image from realsense
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image, depth_image, depth_colorized = get_frame(pipeline, align, temporal, spatial)
            if image is None:
                continue
            im1.set_data(image)
            im2.set_data(depth_colorized)

            rgb = o3d.geometry.Image(np.array(image))
            depth = o3d.geometry.Image(np.array(depth_image))

            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False,
                                                                          depth_scale=1.0, depth_trunc=5000.0)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(
                intrinsics))

            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            labels, boxes, masks = mask_step(model, transforms, image)  # MaskRCNN

            if labels is None and boxes is None and masks is None:
                continue
            bboxes3d = []
            for mask in masks:
                arr_mask = np.asarray(mask.squeeze())
                print(type(arr_mask))
                # apply a binary threshold to arr_mask
                arr_mask = cv2.threshold(arr_mask, 0.1, 1, cv2.THRESH_BINARY)[1]
                print(arr_mask.shape)
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                masked_depth = np.multiply(depth_image, arr_mask)
                # erode mask
                kernel = np.ones((5, 5), np.uint8)
                masked_depth = cv2.erode(masked_depth, kernel, iterations=1)
                depth = o3d.geometry.Image(np.array(masked_depth))

                im3.set_data(masked_depth)

                rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
                                                                              convert_rgb_to_intensity=False,
                                                                              depth_scale=1.0, depth_trunc=5000.0)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(
                    intrinsics))

                pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                aabb = pcd.get_axis_aligned_bounding_box()
                aabb.color = (1, 0, 0)
                image_with_box = bb3d_to_2d(image, aabb, mask)
                # o3d.visualization.draw_geometries([pcd, obb])

            im4.set_data(image)
            plt.pause(0.001)

    except KeyboardInterrupt:
        print('!!!!!!!!!!!!!loop interrupted!!!!!!!!!!!!!')
    # off
    pipeline.stop()
