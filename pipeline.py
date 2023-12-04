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

current_path = os.path.dirname(__file__)


class Pipeline:
    """
    Read the json file and get the rgbd image and the point cloud,
    and then do the preprocessing,including the point cloud filtering,sampling,cropping and
    locating its upper surface center.
    """

    def __init__(self, calib_path=current_path + '/calib',
                 dylib_path=current_path + '/build/libRGBDCameraSDK.so',
                 model_path=current_path + '/model/swinmaskrcnn'):
        self.intrinsics = self._parse_xml(xml_dir=calib_path)
        self._init_camera(dylib_path)
        self.detector = Detector(model_path=model_path, device_name='cuda', device_id=0)

    def _init_camera(self, dylib_path="./build/libRGBDCameraSDK.so"):
        self.sdk = ctypes.CDLL(dylib_path, mode=os.GRND_NONBLOCK)
        self.sdk.MV3D_RGBD_StartCapture()

    @staticmethod
    def _parse_xml(self, xml_dir):
        intrinsics_path = os.path.join(xml_dir, "intrinsics.xml")
        tree = ET.parse(intrinsics_path)
        root = tree.getroot()
        # get "RGBD_MAX" node
        RGBD_MAX = list(filter(lambda x: x.tag == "RGBD_MAX", root.getchildren()))[0]
        # get "RGB" node
        RGB = list(filter(lambda x: x.tag == "RGB", RGBD_MAX.getchildren()))[0]
        intrinsics = list(filter(lambda x: x.tag == "Intrins", RGB.getchildren()))[0]
        fx, fy, cx, cy = intrinsics.get('fx'), intrinsics.get('fy'), intrinsics.get('cx'), intrinsics.get('cy')
        fx, fy, cx, cy = map(float, [fx, fy, cx, cy])
        return [fx, fy, cx, cy]

    def get_recent_rgbd(self, rgbd_dir="./RGBD_data", fmt="tiff"):
        # get most recent rgbd image
        img_list = glob.glob(os.path.join(rgbd_dir, f"*.{fmt}"))
        img_list.sort(key=os.path.getmtime)
        return self.get_rgbd_img(img_list[-1])

    @staticmethod
    def get_rgbd_img(self, img_path):
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

        for i, result in enumerate(inference_results.items()):
            bbox, label, mask = result["bboxes"], result["labels"], result["masks"]
            [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
            if score < 0.5:
                rich.print(f"Skip bbox {bbox} with score {score}")
                continue
            current_roi_img = rgbd_img[top: bottom, left: right]
            roi_list.append({"roi": current_roi_img, "box": bbox, "label": label, "mask": mask})
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
        # draw xyz axis and pcd
        # xyz_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, xyz_axis])

        return pcd

    def fit_plane_normal(self, roi_image, box):
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
        # keep points in the top 3 largest bin
        new_pcd = o3d.geometry.PointCloud()
        for each in top_3_edge_bound:
            mask = np.logical_and(all_depth >= each[0], all_depth < each[1])
            new_pcd += pcd.select_by_index(np.where(mask)[0])

        center = new_pcd.get_center()
        plane_model, inliers = new_pcd.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)
        if (len(inliers) / len(new_pcd.points)) < 0.5:
            print(f"Inliers ratio is {len(inliers) / len(new_pcd.points)}, too small!")
        [a, b, c, d] = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-6)
        if np.dot(plane_normal, np.array([0, 0, 1])) > 0:
            plane_normal = -plane_normal
        # get x axis direction using PCA
        points = np.asarray(new_pcd.points)
        points = points - center
        cov = np.cov(points.T)
        evalue, evector = np.linalg.eig(cov)
        x_axis = evector[:, np.argmax(evalue)]
        y_axis = np.cross(plane_normal, x_axis)
        rotation_matrix = np.array([x_axis, y_axis, plane_normal]).T
        # draw local axis
        local_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        local_axis.rotate(rotation_matrix, center=(0, 0, 0))
        local_axis.translate(center)

        global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 2.0])
        o3d.visualization.draw_geometries([pcd, local_axis, global_axis])
        return plane_normal, center

    def get_upper_surface_center(self, pcd):
        """
        Get the upper surface center of the object
        """

        raise NotImplementedError


if __name__ == '__main__':
    jsg_ppl = Pipeline()
