"""
Capture a single RGB and depth frame and save them to output.pcd in
the libpcl PCD format. View the resulting cloud with:

    pcl_viewer output.pcd

"""
import time

from freenect2 import Device, FrameType
import numpy as np
import cv2
from PIL import Image
import threading
import torch
import open3d as o3d

from mayavi import mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def objcetDetection(rgb, depth):

    if(type(rgb) != str):

        #with torch.no_grad():
        image = rgb.to_array()
        image = np.flip(image, axis=1)

        #Predict Objects in image
        results = model(image)
        df = results.pandas().xyxy[0]

        #find rows with bottle in it
        rows = df.loc[df['name'] == 'cup']

        #calculate middle point of rectangle
        x_middle = (rows.iloc[0]['xmin']+rows.iloc[0]['xmax'])/2
        y_middle = (rows.iloc[0]['ymin']+rows.iloc[0]['ymax']) / 2



        #Merge RGB and Depth Image
        undistorted, registered, with_big_depth= device.registration.apply(
            rgb, depth, with_big_depth=True)
        with_big_depth_array = with_big_depth.to_array()
        #with_big_depth_array = np.flip (with_big_depth_array, axis=0)

        with_big_depth_array = np.flip(with_big_depth_array, axis=1)
        # Combine the depth and RGB data together into a single point cloud.

        plt.clf()
        plt.imshow(with_big_depth_array, cmap='viridis', interpolation='nearest')
        print(with_big_depth_array[int(y_middle), int(x_middle)])
        # Add a colorbar for reference

        plt.colorbar(label='Depth (mm)')
        circle = patches.Circle((int(x_middle), int(y_middle)), radius=10, edgecolor='r', facecolor='none',
                                linewidth=2)
        plt.gca().add_patch(circle)

        # Show the plot
        plt.title('Kinect Depth Heatmap')
        #plt.savefig('kinect_depth_heatmap.png')
        #plt.show(block = False)
        plt.close()

        with_big_depth_xyz_array = device.registration.get_big_points_xyz_array(with_big_depth)
        with_big_depth_xyz_array = np.flip(with_big_depth_xyz_array, axis=1)
        print(with_big_depth_xyz_array[int(y_middle), int(x_middle)])
        #Save data in pont cloud "ply" file if needed
        # depth_xyz = device.registration.get_big_points_xyz_array(with_big_depth)
        # depth_xyz = a.reshape(-1, 3)
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(depth_xyz)
        # Save the point cloud to a file (e.g., in PLY format)
        #o3d.io.write_point_cloud("point_cloud.ply", pcd)


        #with open('output.pcd', 'wb') as fobj:
        #    device.registration.write_big_pcd(fobj, with_big_depth, rgb)

        #Save image with dot for  middle
        image = rgb.to_array()
        image_ = np.flip(image, axis = 1)
        center = (int(x_middle), int(y_middle))
        image_ = cv2.circle( cv2.UMat(image_), center=center, radius=2, color=(0, 0, 255), thickness=5)
        cv2.imwrite("rgb_image.jpg", image_)





# Open the default device and capture a color and depth frame.
device = Device()
frames = {}

#Load object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
with device.running():
    thread = threading.Thread(target=objcetDetection, args=("rgb",))
    for type_, frame in device:
        frames[type_] = frame
        if FrameType.Color in frames and FrameType.Depth in frames:
            rgb, depth = frames[FrameType.Color], frames[FrameType.Depth]

            #Start new thread for object detection, without new thread the camera therad will stop
            if (not thread.is_alive()):
                thread = threading.Thread(target=objcetDetection, args=(rgb,depth))
                thread.deamon = True
                thread.start()





