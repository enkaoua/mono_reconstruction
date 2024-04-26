import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import open3d as o3d
def generate_sphere(radius=1, num_points=1000):
    """
    generates sphere of given radius

    :param radius: (int) radius of sphere
    :param num_points: (int) how many points in the point cloud - determines how dense it is
    :return: (ndarray of shape N,3) point cloud representing sphere
    """

    phi = np.random.uniform(low=0, high=np.pi, size=(num_points,))
    theta = np.random.uniform(low=0, high=2*np.pi, size=(num_points,))

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    point_cloud = np.vstack([x, y, z]).T

    return point_cloud

def main(): 
    points = np.loadtxt('/Users/aure/Documents/CARES/code/mono_reconstruction/data/target_points.txt')
    r = 1

    global_pc = o3d.geometry.PointCloud()
    for p in points:
        # create 3D spheres with given radius at points
        pc = generate_sphere(radius=r, num_points=1000)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)

        # transform circle's center to point p
        T = np.eye(4)
        T[:3, 3] = p
        pcd.transform(T)

        

        # add circle to global pointcloud
        global_pc += pcd

    # calculate normals of pc and add to mesh
    global_pc.estimate_normals()

    # Orient the normals
    global_pc.orient_normals_towards_camera_location(global_pc.get_center())

    # save pointcloud to file
    o3d.io.write_point_cloud("/Users/aure/Documents/CARES/code/mono_reconstruction/results/pc.ply", global_pc)
    # Create the mesh using the Ball-Pivoting Algorithm (BPA)
    #radii = [0.0005, 0.001, 0.002, 0.004]
    #bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(global_pc, o3d.utility.DoubleVector(radii))

    # Save the mesh
    #o3d.io.write_triangle_mesh("/Users/aure/Documents/CARES/code/mono_reconstruction/results/mesh.ply", bpa_mesh)

    #o3d.visualization.draw_geometries([ bpa_mesh])
   

    
    return 


if __name__=='__main__': 
    main() 