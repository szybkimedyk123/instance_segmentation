import numpy as np
import cv2
import pyrealsense2 as rs
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from model.maskrcnn import MaskRCNN
import open3d as o3d
from utils import *
import time


if __name__ == '__main__':
    # const init
    categories = {
        "miska2d": 0,
        "kubek2d": 1,
        "pudelko2d": 2,
        "czapka2d": 3,
        "but2d": 4,
        "kieliszek2d": 5,
        "lampka2d": 6
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

    # RealSense init
    colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)
    pipeline = rs.pipeline()
    config = rs.config()
    resolution_width = 1280
    resolution_height = 720
    config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.rgb8, 30)
    temporal = rs.temporal_filter(smooth_alpha=0.05, smooth_delta=20, persistence_control=8)
    spatial = rs.spatial_filter(smooth_alpha=0.5, smooth_delta=20, magnitude=2, hole_fill=0)
    temp = pipeline.start(config)

    # intrinsics matrix init
    profile = temp.get_stream(rs.stream.depth)
    intrinsics_rs = profile.as_video_stream_profile().get_intrinsics()
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(intrinsics_rs.width, intrinsics_rs.height,
                              intrinsics_rs.fx, intrinsics_rs.fy,
                              intrinsics_rs.ppx, intrinsics_rs.ppy)
    kernel = np.ones((5, 5), np.uint8)
    fps = 0
    while True:
        frame_time = time.time()
        # **************** Get frame ****************
        image, depth_image, depth_colorized = get_frame(pipeline, align, temporal, spatial, colorizer)
        if image is None or depth_image is None or depth_colorized is None:
            continue

        # **************** Mask R-CNN ****************
        labels, boxes, masks = mask_step(model, transforms, image)  # MaskRCNN
        if labels is None and boxes is None and masks is None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # # **************** 2D bounding box and label ****************
        # image2print = image.copy()
        # for mask, box, label in zip(masks, boxes, labels):
        #     category_str = list(categories.keys())[int(label)]
        #     cv2.putText(image2print, category_str,
        #                 (int(box[0]), int(box[1] + 5)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (0, 0, 255), 1, cv2.LINE_AA)
        #     cv2.rectangle(image2print,
        #                   (int(box[0]), int(box[1])),
        #                   (int(box[2]), int(box[3])),
        #                   (0, 0, 255), 2)

        # **************** Open3D ****************
        rgb = o3d.geometry.Image(np.array(image))
        # depth = o3d.geometry.Image(np.array(depth_image))
        # rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
        #                                                               convert_rgb_to_intensity=False,
        #                                                               depth_scale=1.0,
        #                                                               depth_trunc=10000.0)
        # whole_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(intrinsics))
        # whole_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #R = whole_pcd.get_rotation_matrix_from_xyz((-np.pi / 6, 0, 0))
        # whole_pcd.rotate(R, center=(0, 0, 0))

        # **************** 3D bounding box for each mask ****************
        bboxes3d = []
        for mask in masks:
            arr_mask = np.asarray(mask.squeeze())
            arr_mask = cv2.threshold(arr_mask, 0.1, 1, cv2.THRESH_BINARY)[1]
            masked_depth = np.multiply(depth_image, arr_mask)
            masked_depth = cv2.erode(masked_depth, kernel, iterations=2)
            depth = o3d.geometry.Image(np.array(masked_depth))
            # rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
            #                                                               convert_rgb_to_intensity=False,
            #                                                               depth_scale=1.0, depth_trunc=5000.0)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, o3d.camera.PinholeCameraIntrinsic(
                intrinsics))
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd = pcd.voxel_down_sample(voxel_size=10)

            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)
            R = pcd.get_rotation_matrix_from_xyz((-np.pi / 6, 0, 0))
            pcd.rotate(R, center=(0, 0, 0))
            aabb = pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            bboxes3d.append(aabb)

        # **************** 3D to 2D projection ****************
        image_with_box = image.copy()
        for box3d, mask, label, box2d in zip(bboxes3d, masks, labels, boxes):
            try:
                image_with_box = bb3d_to_2d(image_with_box, box3d, mask)
            except Exception as e:
                print('warning: ', e)
            category_str = list(categories.keys())[int(label)]
            cv2.putText(image_with_box, category_str[:-2],
                        (int(box2d[0]), int(box2d[1] + 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)

        # **************** show image ****************
        image_with_box = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
        cv2.putText(image_with_box, 'FPS: {:.2f}'.format(fps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('image', image_with_box)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            rgb = o3d.geometry.Image(np.array(image))
            depth = o3d.geometry.Image(np.array(depth_image))
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
                                                                          convert_rgb_to_intensity=False,
                                                                          depth_scale=1.0,
                                                                          depth_trunc=10000.0)
            whole_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img,
                                                                       o3d.camera.PinholeCameraIntrinsic(intrinsics))
            whole_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            #whole_pcd = whole_pcd.voxel_down_sample(voxel_size=10)
            R = whole_pcd.get_rotation_matrix_from_xyz((-np.pi / 6, 0, 0))
            whole_pcd.rotate(R, center=(0, 0, 0))
            o3d.visualization.draw_geometries([whole_pcd] + bboxes3d)

        if key == ord('q'):
            break

        current_time = time.time()

        fps = 1 / (current_time - frame_time)

    print('\n****** DONE ******\n')
