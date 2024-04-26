#%%

import open3d.core as o3c
import open3d as o3d
import numpy as np

datapath = "/home/service/datasets/Stanford3dDataset_v1.2_Aligned_Version"

ppath = f"{datapath}/Area_1/conferenceRoom_1/conferenceRoom_1.txt"

data = np.loadtxt(ppath)

cloud = o3c.Tensor(data[:, :3], dtype=o3c.float32, device=o3c.Device("CUDA:0"))

#%%
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
