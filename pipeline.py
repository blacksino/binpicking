import open3d as o3d
import numpy as np
import os
import json
import cv2
import xml.etree.ElementTree as ET
import glob
from matplotlib import pyplot as plt
import rich
from mmdeploy_runtime import Detector
import cv2
from glob import glob
import numpy as np
import os
import sys
import ctypes
import shutil
import time

current_path = '/home/wsco/code/jsg'
# os.environ["LD_LIBRARY_PATH"] += f":{current_path}/dependency/lib"


class Pipeline:
    """
    Read the json file and get the rgbd image and the point cloud,
    and then do the preprocessing,including the point cloud filtering,sampling,cropping and
    locating its upper surface center.
    """

    def __init__(self, calib_path=current_path + '/calib',
                 dylib_path=current_path + '/build/libRGBDCameraSDK.so',
                 model_path=current_path + '/model/swinmaskrcnn',
                 data_path=current_path + '/data'):
        self.calib_path = calib_path
        self.dylib_path = dylib_path
        self.model_path = model_path
        self.data_path = data_path

        self.intrinsics,self.extrinsics = self._parse_xml(xml_dir=calib_path)
        self.detector = Detector(model_path=model_path, device_name='cuda', device_id=0)
        self.data_path = data_path
        self.rgb2robot = self._convert_extrinsics()

    def init_camera(self):
        self._init_camera(self.dylib_path)

    def _convert_extrinsics(self):
        # convert tx,ty,tz,rx,ry,rz to transformation matrix
        tx, ty, tz, rx, ry, rz = self.extrinsics
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [tx, ty, tz]
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])
        transform_matrix = np.dot(translation_matrix, rotation_matrix)
        transform_matrix = np.linalg.inv(transform_matrix)
        return transform_matrix


    def _init_camera(self, dylib_path="./build/libRGBDCameraSDK.so"):
        try:
            self.sdk = ctypes.CDLL(dylib_path, mode=os.GRND_NONBLOCK)
            self.sdk.MV3D_RGBD_StartCapture()
        except OSError as e:
            print(f"Error loading library: {e}")
            sys.exit(1)

    def _capture_rgbd(self):
        try:
            self.sdk.MV3D_RGBD_GetFrame()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None, None
        rgb_img = cv2.imread("RGB.png")
        depth_img = cv2.imread("Depth.png", cv2.IMREAD_UNCHANGED)
        shutil.move("RGB.png", os.path.join(self.data_path, "RGB.png"))
        shutil.move("Depth.png", os.path.join(self.data_path, "Depth.png"))
        timestamp = time.time()
        local_time = time.localtime(timestamp)
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
        os.rename(os.path.join(self.data_path, "RGB.png"),
                  os.path.join(self.data_path, f"{formatted_time}_RGB.png"))
        os.rename(os.path.join(self.data_path, "Depth.png"),
                  os.path.join(self.data_path, f"{formatted_time}_Depth.png"))
        return rgb_img, depth_img



    def run_pipeline(self):
        rgb_img,depth_img = self._capture_rgbd()
        rgbd_img = np.concatenate([rgb_img, depth_img[..., np.newaxis]], axis=-1)
        roi_list = self.get_roi(rgbd_img)
        num_of_detection = len(roi_list)
        transform_matrix_list = []
        for i in range(num_of_detection):
            result = self.fit_plane_normal(roi_list[i]["roi"], roi_list[i]["box"])
            result = self.rgb2robot @ result
            transform_matrix_list.append(result)
        return transform_matrix_list


    @staticmethod
    def _parse_xml(xml_dir):
        intrinsics_path = os.path.join(xml_dir, "intrinsics.mfa")
        tree = ET.parse(intrinsics_path)
        root = tree.getroot()
        intrinsics = root.find(".//RGBD_MAX/RGB/Intrins")
        fx, fy, cx, cy = map(float, [intrinsics.get('fx'), intrinsics.get('fy'), intrinsics.get('cx'), intrinsics.get('cy')])

        extrinsics_path = os.path.join(xml_dir, "extrinsics.mfa")
        tree = ET.parse(extrinsics_path)
        root = tree.getroot()
        extrinsics = root.find(".//SYS_CALIB_INFO/Robot2rgb")
        tx, ty, tz, rx, ry, rz = map(float, [extrinsics.get('tx'), extrinsics.get('ty'), extrinsics.get('tz'),
                                             extrinsics.get('rx'), extrinsics.get('ry'), extrinsics.get('rz')])

        return [fx, fy, cx, cy], [tx, ty, tz, rx, ry, rz]

    @staticmethod
    def get_rgbd_img(img_path):
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    def _run_inference(self, rgb_img):
        assert rgb_img.shape[-1] == 3, "Detector input must be an rgb image!"
        rich.print("[bold red]Running inference...[/bold red]")
        bboxes, labels, masks = self.detector(rgb_img)
        rich.print("[bold green]Inference Done![/bold green]")
        return {"bboxes": bboxes, "labels": labels, "masks": masks}

    def get_roi(self, rgbd_img):
        rgb_img = rgbd_img[..., :3]
        inference_results = self._run_inference(rgb_img)
        num_of_detection = len(inference_results["bboxes"])
        roi_list = []
        if num_of_detection == 0:
            rich.print("[bold red]No object detected![/bold red]")
            return None

        bboxes, labels, masks = inference_results["bboxes"], inference_results["labels"], inference_results["masks"]
        for i in range(len(bboxes)):
            bbox, label, mask = bboxes[i], labels[i], masks[i]
            [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
            if score < 0.5:
                rich.print(f"Skip bbox {bbox} with score {score}")
                continue
            current_roi_img = rgbd_img[top: bottom, left: right]
            roi_list.append({"roi": current_roi_img, "box": bbox[:4], "label": label, "mask": mask})
        return roi_list

    def down_sample_using_voxel_grid(self, pcd, voxel_size=0.005):
        """
        Down sample the point cloud using voxel grid
        """
        d_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        return d_pcd

    def remove_outlier(self, pcd, nb_points=20, radius=0.05):

        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return pcd.select_by_index(ind)

    def update_intrinsics_after_crop(self, cx, cy, box):
        x, y, w, h = box
        new_cx = cx - x
        new_cy = cy - y
        return new_cx, new_cy

    def rgbd_to_pcd(self, rgbd_img, box):
        rgb = rgbd_img[..., :3].astype(np.uint8)
        depth = rgbd_img[..., -1].astype(np.float32)
        o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),
                                                                      o3d.geometry.Image(depth),
                                                                      depth_scale=10000,
                                                                      convert_rgb_to_intensity=False)
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        new_cx, new_cy = self.update_intrinsics_after_crop(self.intrinsics[2],
                                                           self.intrinsics[3],
                                                           box)
        intrinsics.set_intrinsics(rgbd_img.shape[1], rgbd_img.shape[0],
                                  fx=self.intrinsics[0],
                                  fy=self.intrinsics[1],
                                  cx=new_cx, cy=new_cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, intrinsics)
        return pcd

    def fit_plane_normal(self, roi_image, box,visualize=False):
        """
        Fit the plane normal of the point cloud
        """
        pcd = self.rgbd_to_pcd(roi_image, box)
        pcd = self.down_sample_using_voxel_grid(pcd)
        pcd = self.remove_outlier(pcd)

        # get depth histogram
        all_depth = np.asarray(pcd.points)[:, 2]
        hist, bin_edge = np.histogram(all_depth, bins=10)
        # get top 3 largest bin
        top_3 = np.argsort(hist)[-3:]
        top_3_edge_bound = [[bin_edge[each], bin_edge[each + 1]] for each in top_3]

        new_pcd = o3d.geometry.PointCloud()
        top_3_edge_bound = np.array(top_3_edge_bound)
        idx = np.argmax(top_3_edge_bound[:,0])

        mask = np.logical_and(all_depth >= top_3_edge_bound[idx][0], all_depth < top_3_edge_bound[idx][1])
        new_pcd += pcd.select_by_index(np.where(mask)[0])

        center = pcd.get_center()
        plane_model, inliers = new_pcd.segment_plane(distance_threshold=0.02,
                                                     ransac_n=3,
                                                     num_iterations=1000)
        if (len(inliers) / len(new_pcd.points)) < 0.5:
            inlier_ratio = len(inliers) / len(new_pcd.points)
            rich.print(f"[bold red]Inlier ratio is {inlier_ratio}, too small![/bold red]")
        [a, b, c, d] = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-6)
        if np.dot(plane_normal, np.array([0, 0, 1])) > 0:
            plane_normal = -plane_normal
        # get x axis direction using PCA
        points = np.asarray(pcd.points)
        points = points - center
        cov = np.cov(points.T)
        evalue, evector = np.linalg.eig(cov)
        x_axis = evector[:, np.argmax(evalue)]
        if np.dot(x_axis, np.array([0, 0, 1])) < 0:
            x_axis = -x_axis

        y_axis = np.cross(plane_normal, x_axis)


        rotation_matrix = np.array([x_axis, y_axis, plane_normal]).T
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = center

        # draw local axis
        if visualize:
            local_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            local_axis.rotate(rotation_matrix, center=(0, 0, 0))
            local_axis.translate(center)

            global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 1.0])
            o3d.visualization.draw_geometries([pcd, local_axis, global_axis])
        return transform_matrix

    @staticmethod
    def tool_coordinate_calib(calib_mat,robot_mat):
        return calib_mat @ robot_mat


    def get_upper_surface_center(self, pcd):
        """
        Get the upper surface center of the object
        """

        raise NotImplementedError


if __name__ == '__main__':
    jsg_ppl = Pipeline()
    # test
    # rgb_img = cv2.imread("./test_rgb/test_rgb.bmp")
    # depth_img = cv2.imread("./test_rgb/test_depth.tiff", cv2.IMREAD_UNCHANGED)
    # rgbd_img = np.concatenate([rgb_img, depth_img[..., np.newaxis]], axis=-1)
    # roi_list = jsg_ppl.get_roi(rgbd_img)
    # for i in range(15):
    #     jsg_ppl.fit_plane_normal(roi_list[i]["roi"], roi_list[i]["box"])
    jsg_ppl.run_pipeline()