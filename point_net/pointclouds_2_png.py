import open3d as o3d
import numpy as np
import os
import re
if os.path.exists('points'):
    os.system('rm -r points')
os.mkdir('points')
curr_dir=os.getcwd()
curr_dir=os.path.join(curr_dir,'s3dis_test_pointclouds')
points=os.listdir(curr_dir)
points=[point for point in points if point.endswith('.ply')]

pred_points=[point for point in points if 'predicted' in point]
ground_points=[point for point in points if 'ground' in point]

# Sort pred_points list by extracting the first integer from each filename
pred_points.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

# Sort ground_points list by extracting the first integer from each filename
ground_points.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

# plot and save the plots as png, and then close the plots
for i in range(len(pred_points)):
    pred_name=pred_points[i]
    ground_name=ground_points[i]

    epoch=re.findall(r'\d+',pred_name)[0]

    pcd_pred = o3d.io.read_point_cloud(pred_name)
    pcd_ground = o3d.io.read_point_cloud(ground_name)
    R = pcd_pred.get_rotation_matrix_from_xyz((0.7 * np.pi, 0, 0.8 * np.pi))
    pcd_pred.rotate(R, center=(0, 0, 0))
    pcd_ground.rotate(R, center=(0, 0, 0))


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_pred)
    vis.add_geometry(pcd_pred)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('points/pred_'+str(epoch)+'.png')
    vis.destroy_window()

    vis=o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_ground)
    vis.add_geometry(pcd_ground)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('points/ground_'+str(epoch)+'.png')
    vis.destroy_window()


