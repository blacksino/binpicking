import open3d as o3d
import numpy as np
import os
import json
import cv2
import xml.etree.ElementTree as ET
import glob
from matplotlib import pyplot as plt
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

    def __init__(self, calib_path=None,dylib_path=None,model_path=None):
        # assert  os.path.exists(json_path), "The json file does not exist!"
        # self.info = json.load(open(json_path))
        # self._parse_json()
        self.intrinsics = self._parse_xml(calib_path)
        self._init_camera(dylib_path)
        self.detector = Detector(model_path=model_path, device_name='cuda', device_id=0)

    def _init_camera(self, dylib_path):
        self.sdk = ctypes.CDLL("/home/wsco/code/jsg/build/libRGBDCameraSDK.so", mode=os.GRND_NONBLOCK)
        self.sdf.MV3D_RGBD_StartCapture()


    def _parse_xml(self, xml_path=current_path+'/calib'):

        tree = ET.parse(xml_path)
        root = tree.getroot()
        # get "RGBD_MAX" node
        RGBD_MAX = list(filter(lambda x: x.tag == "RGBD_MAX", root.getchildren()))[0]
        # get "RGB" node
        RGB = list(filter(lambda x: x.tag == "RGB", RGBD_MAX.getchildren()))[0]
        intrinsics = list(filter(lambda x: x.tag == "Intrins", RGB.getchildren()))[0]
        fx, fy, cx, cy = intrinsics.get('fx'), intrinsics.get('fy'), intrinsics.get('cx'), intrinsics.get('cy')
        fx, fy, cx, cy = map(float, [fx, fy, cx, cy])
        return [fx, fy, cx, cy]
        # get "fx" node,"fy" node,"cx" node,"cy" node
    def _parse_json(self):
        """
        Parse the json file
        """
        self.img_info = self.info["images"]
        self.anno_dict = self.info["annotations"]
        # group the annotations by image id
        self.anno_info = {anno["image_id"]: [] for anno in self.anno_dict}
        for anno in self.anno_dict:
            self.anno_info[anno["image_id"]].append(anno)

    def get_rgbd_img(self):
        img_path = self.img_info[index]["file_name"]
        if "depth_file_name" in self.img_info[index]:
            depth_path = self.img_info[index]["depth_file_name"]
        else:
            depth_base_name = os.path.basename(img_path)
            depth_base_name = depth_base_name.replace("彩色图", "深度图")
            depth_base_name = depth_base_name.replace(".jpg", ".tiff")
            # search depth file name in the given directory
            depth_path = glob.glob(os.path.join(depth_dir, depth_base_name))
            if len(depth_path) == 0:
                print("The depth file does not exist!")
                return None
            else:
                depth_path = depth_path[0]
        rgb_img = cv2.imread(img_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # check if the sizes of the rgb and depth images match
        if rgb_img.shape[:2] != depth_img.shape:
            print("The sizes of the rgb and depth images do not match!")
            return None
        # concatenate the rgb and depth image
        rgbd_img = np.concatenate([rgb_img, depth_img[..., np.newaxis]], axis=-1)
        return rgbd_img

    def get_roi(self, rgbd_img, box, mask):

        # crop image using box
        crop_img = rgbd_img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        roi_results = list()

        # seg = each_anno["segmentation"]
        # convert seg contours points to mask
        # mask = np.zeros_like(rgbd_img[..., -1])
        # pts = np.array([seg]).reshape(-1,2).astype(np.int32)
        # pts = pts.reshape(-1, 1, 2)
        # mask = cv2.fillPoly(mask, [pts] , color=255)
        # crop mask using box
        cropped_mask = mask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        # get the roi of the object
        roi = crop_img * (cropped_mask[..., np.newaxis].astype(bool))
        roi_results.append({"roi": roi, "box": box})
        return roi_results

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
                                                                      depth_scale=10000, convert_rgb_to_intensity=False)
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        new_cx, new_cy = self.update_intrinsics_after_crop(cx, cy, box)
        intrinsics.set_intrinsics(rgbd_img.shape[1], rgbd_img.shape[0],
                                  fx=fx, fy=fy, cx=new_cx, cy=new_cy)
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
        if len(inliers) / len(new_pcd.points) < 0.5:
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
    json_path = "/Users/blacksino/PycharmProjects/code/js_rgbd/train.json"
    detector = Detector(model_path='/root/workspace/mmdeploy/swinmaskrcnn', device_name='cuda', device_id=0)

    # image_path = "/mm/test.jpg"
    image_path = sys.argv[1]
    depth_path = sys.argv[2]

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    depth = np.concatenate([img, depth[:, :, None]], axis=-1)

    bboxes, labels, masks = detector(img)

    output_mask = np.zeros(img.shape[:-1])
    # 使用阈值过滤推理结果，并绘制到原图中
    indices = [i for i in range(len(bboxes))]
    a = 0
    for index, bbox, label_id, mask in zip(indices, bboxes, labels, masks):
        output_mask = np.zeros(img.shape[:-1])
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        if score < 0.5:
            continue
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))
        print(mask.shape)

        ori_resize_mask = cv2.resize(mask, (right - left, bottom - top))

        resize_mask = cv2.resize(np.uint8(mask > 0), (right - left, bottom - top), interpolation=cv2.INTER_NEAREST) * (
                    index + 1)
        resize_mask *= (output_mask[top: bottom, left: right] == 0)
        output_mask[top: bottom, left: right] = resize_mask

        pre = Pipeline([fx, fy, cx, cy])
        roi_results = pre.get_roi(depth, [left, top, right - left, bottom - top], output_mask)
        plane_normal, center = pre.fit_plane_normal(roi_results[0]["roi"], roi_results[0]["box"])
        print(plane_normal, center)
        a += 1
        if a == 2:
            break
    cv2.imwrite("/mm/output.jpg", np.uint8(output_mask > 0) * 255)

    translation = [-1328.073802142068, 324.131363107185, 2200.6887077484457]
    euler = [-2.9317062982468785, -0.019257383521215202, 1.6256755676337336]
    from scipy.spatial.transform import Rotation as R

    r = R.from_euler('xyz', euler, degrees=False)
    rot_mat = r.as_matrix()
    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = rot_mat
    extrinsics[:3, 3] = np.array(translation)
    extrinsics[3, 3] = 1
    extrinsics = np.linalg.inv(extrinsics)

    print(extrinsics)

    normal = np.ones(4)
    normal[:3] = plane_normal
    new_center = np.ones(4)
    new_center[:3] = center * 1000

    print(new_center)
    print(normal)
    normal = extrinsics[:3, :3] @ normal[:3]
    new_center = extrinsics @ new_center
    print(normal, new_center)

    # r = R.from_rotvec(normal)
    # euler = r.as_euler('zyx',degrees=True)
    # print(euler)

    additional_euler = [0, np.deg2rad(15), bo, ]
    additional_translation = [0, 0, 393]
    r = R.from_euler("xyz", additional_euler, degrees=False)
    rot = r.as_matrix()
    print(rot)

    additional_ex = np.zeros((4, 4))
    additional_ex[:3, :3] = rot
    additional_ex[:3, 3] = additional_translation
    additional_ex[3, 3] = 1

    normal = rot @ normal
    print(normal)

    r = R.from_rotvec(normal)
    euler = r.as_euler('zyx', degrees=True)
    print("euler", euler)
    print("add", additional_ex)
    # additional_ex = np.linalg.inv(additional_ex)
    print("center", additional_ex @ new_center)



