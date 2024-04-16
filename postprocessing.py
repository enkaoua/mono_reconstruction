import numpy as np
import open3d as o3d

def main(results_dir='/Users/aure/Documents/CARES/code/mono_reconstruction/results/aug_10_testing_over_1'): 
    
    # 0) initialise pointcloud where we will be adding all pointclouds
    global_pcd = o3d.geometry.PointCloud()

    # 1) load poses
    poses = np.load(f'{results_dir}/pred_poses.npy')
    # 2) loop through PCs and poses
    for i in range(0, 1): 

        # 2.1) load open3d pointcloud
        pcd = o3d.io.read_point_cloud(f'{results_dir}/{i}.ply')
        # 2.2) load pose
        pose = poses[i]

        if i==0:
            # 1.1) set pose of first pointcloud
            T = np.copy(pose)
           
        else:
            # 1.2) transform pose of first pointcloud to pose of current pointcloud
            T = pose  @ T
            # 1.3) set pose of current pointcloud to T
        pcd.transform(np.linalg.inv(T).squeeze())
        # 1.4) add pcd to global pointcloud
        global_pcd += pcd

        # 2.4) save pointcloud
    # 4) visualise pointcloud
    out_pcd = o3d.io.read_point_cloud(f'/Users/aure/Documents/CARES/code/mono_reconstruction/data/rec_aug/reconstruction/pc_poses/rec_10/out0.ply')
    
    # multiply global pc points to scale by 1000
    global_pcd.points = o3d.utility.Vector3dVector(np.array(global_pcd.points) * 100000)
    # paint each pointcloud single color
    global_pcd.paint_uniform_color([1, 0.706, 0])
    out_pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([global_pcd, out_pcd])
    

    return 


if __name__=='__main__': 
    main(results_dir='/Users/aure/Documents/CARES/code/mono_reconstruction/results/aug_10_testing_over_1',
         ) 