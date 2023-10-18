import numpy as np
import cv2
import pyrealsense2 as rs
import torch
import torchvision.transforms as transforms
from maskrcnn import MaskRCNN
import open3d as o3d


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
    pipeline.start(config)
    temporal = rs.temporal_filter(smooth_alpha=0.05, smooth_delta=20, persistence_control=8)
    spatial = rs.spatial_filter(smooth_alpha=0.5, smooth_delta=20, magnitude=2, hole_fill=0)

    # Open3D init
    # get intrinsic matrix from bag TODO: get from stream
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open('/home/lab/ZSD-IS/dataset/intrinsic_matrix.bag')
    intrinsics = bag_reader.metadata.intrinsics

    while True:
        # Get frame
        image, depth_image, depth_colorized = get_frame(pipeline, align, temporal, spatial)
        if image is None or depth_image is None or depth_colorized is None:
            continue
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mask R-CNN
        labels, boxes, masks = mask_step(model, transforms, image)  # MaskRCNN
        if labels is None and boxes is None and masks is None:
            continue

        # for mask, box, label in zip(masks, boxes, labels):
        #     category = list(categories.keys())[int(label)]
        #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        #     cv2.putText(image, category, (int(box[0]), int(box[1] + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        # Open3D 3D bounding box
        rgb = o3d.geometry.Image(np.array(image))
        depth = o3d.geometry.Image(np.array(depth_image))
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
                                                                      convert_rgb_to_intensity=False,
                                                                      depth_scale=1.0,
                                                                      depth_trunc=10000.0)
        whole_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(intrinsics))
        whole_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        R = whole_pcd.get_rotation_matrix_from_xyz((-np.pi / 6, 0, 0))
        whole_pcd.rotate(R, center=(0, 0, 0))


        bboxes3d = []
        for mask in masks:
            arr_mask = np.asarray(mask.squeeze())
            # print(type(arr_mask))
            # apply a binary threshold to arr_mask
            arr_mask = cv2.threshold(arr_mask, 0.1, 1, cv2.THRESH_BINARY)[1]
            # print(arr_mask.shape)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            masked_depth = np.multiply(depth_image, arr_mask)
            # erode mask
            kernel = np.ones((5, 5), np.uint8)
            masked_depth = cv2.erode(masked_depth, kernel, iterations=2)
            depth = o3d.geometry.Image(np.array(masked_depth))

            # im3.set_data(masked_depth)

            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
                                                                          convert_rgb_to_intensity=False,
                                                                          depth_scale=1.0, depth_trunc=5000.0)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(
                intrinsics))

            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # find statistical outliers
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)
            # rotate point cloud
            pcd.rotate(R, center=(0, 0, 0))


            aabb = pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            # image_with_box = bb3d_to_2d(image, aabb, mask)
            bboxes3d.append(aabb)
            # o3d.visualization.draw_geometries([pcd, aabb])
        o3d.visualization.draw_geometries([whole_pcd] + bboxes3d)
        # show image
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
