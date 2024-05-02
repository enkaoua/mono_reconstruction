import numpy as np
import open3d as o3d
import cv2
import glob


def process_pc(cloud, pose,hand_eye, max_distance, scale=1):
    '''
    Processes point cloud by:
    1. filtering pc beyond max distance
    2. converting pc to OpenCV coordinate system by flipping Y and Z axes
    3. transforming pc from camera to aruco coordinates using given pose

    Args:
    - cloud: input point cloud
    - pose: transformation from camera to aruco frame
    - max_distance: max distance to filter points that are too far away
    
    '''

    aruco_cloud = o3d.geometry.PointCloud()

    current_pnts = np.array(cloud.points) 
    current_colors = np.array(cloud.colors)

    # filtering anything further than a metre (or given distance)
    #distance_filter = current_pnts[:,2]<float(max_distance)
    #current_pnts = current_pnts[distance_filter, :]
    #current_colors = current_colors[distance_filter, :]


    current_pnts = current_pnts * scale 
    # filtering anything further than a metre (or given distance)
    #distance_filter = current_pnts[:,0]<0
    #current_pnts = current_pnts[distance_filter, :]
    #current_colors = current_colors[distance_filter, :]

    # converting to opeCV coords from openGL by flipping Y and Z- otherwise pc appears flipped on the image
    current_pnts_hom = cv2.convertPointsToHomogeneous(current_pnts).squeeze().squeeze() # converting to homogenous
    flip_yx = np.eye(4)
    #flip_yx[0,0]=-1
    #flip_yx[1,1]=-1
    #flip_yx[2,2]=-1
    current_pnts_hom_openCV = flip_yx @ current_pnts_hom.T
    
    # scale 
    scale_transform = np.eye(4)*scale
    current_pnts_hom_openCV = scale_transform @ current_pnts_hom_openCV

    # transforming current point cloud to aruco coord system
    current_pnts_first_hom_aruco = np.linalg.inv(pose) @ np.linalg.inv(hand_eye) @ current_pnts_hom_openCV
    
    """ # scale 
    scale_transform = np.eye(4)*scale
    current_pnts_first_hom_aruco = scale_transform @ current_pnts_first_hom_aruco
    """
    current_pnts_aruco = cv2.convertPointsFromHomogeneous(current_pnts_first_hom_aruco.T).squeeze().squeeze() # convert back to cartesian system

    # adding new points and colors to open3d pc object
    aruco_cloud.points = o3d.utility.Vector3dVector(current_pnts_aruco)
    aruco_cloud.colors = o3d.utility.Vector3dVector(current_colors)

    return aruco_cloud
  


def main(original_dir='/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/august_recordings/aug_1_endo',
         original_poses_dir='/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/reconstruction/pc_poses/rec_1',
         results_dir='/Users/aure/Documents/CARES/code/mono_reconstruction/results/aug_1'): 
    
    # 0) initialise pointcloud where we will be adding all pointclouds
    global_pcd = o3d.geometry.PointCloud()

    # 1) load predicted poses
    predicted_poses = np.load(f'{results_dir}/pred_poses.npy')
    poses = glob.glob(f'{original_poses_dir}/*.npy')

    # load he
    hand_eye = np.loadtxt('/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/august_recordings/zoomed_calibration/hand_eye.txt')
    # 2) loop through PCs and poses
    for i in range(60, 61): 

        # 2.1) load open3d pointcloud
        pcd = o3d.io.read_point_cloud(f'{results_dir}/{i}.ply') 
        # 2.2) load pose from camera to aruco
        predicted_pose = predicted_poses[i]
        pose = np.load(poses[i])
        # 1.2) transform pose of first pointcloud to pose of current pointcloud
        #current_pnts_hom = cv2.convertPointsToHomogeneous(np.array(pcd.points  ) *1000000).squeeze().squeeze() # converting to homogenous
        #current_pnts_first_hom_aruco = np.linalg.inv(hand_eye) @ np.linalg.inv(pose) @ current_pnts_hom.T
        #current_pnts_aruco = cv2.convertPointsFromHomogeneous(current_pnts_first_hom_aruco.T).squeeze().squeeze() # convert back to cartesian system

        # adding new points and colors to open3d pc object
        #pcd.points = o3d.utility.Vector3dVector(current_pnts_aruco )
        # TODO
        processed_pc = process_pc(pcd, pose, hand_eye, 1.0, scale=10000)

        # 1.3) set pose of current pointcloud to T
        #pcd.transform(np.linalg.inv(T.squeeze()))
        # 1.4) add pcd to global pointcloud
        global_pcd += processed_pc

        # 2.4) save pointcloud
    # 4) visualise pointcloud
    pc_pth = '/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/registration/registered_models/rec_1/registered_surface.ply'
    #pc_pth = f'/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/reconstruction/pc_poses/rec_10/out0.ply'
    #pc_pth = '/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/reconstruction/cropped/rec_1/cropped.ply'
    out_pcd = o3d.io.read_point_cloud(pc_pth)

    # multiply global pc points to scale by 1000
    global_pcd.points = o3d.utility.Vector3dVector(np.array(global_pcd.points))
    # paint each pointcloud single color
    #global_pcd.paint_uniform_color([1, 0.706, 0])
    #out_pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([global_pcd, out_pcd]) # 
    

    return 


if __name__=='__main__': 
    main(results_dir='/Users/aure/Documents/CARES/code/mono_reconstruction/results/aug_1',
         ) 