# Source code for 3D bat reconstruction pipeline
  Here is the repo contains all the source code for our work "A Model-Based Deep-Learning Approach to Reconstructing the Highly Articulated Flight Kinematics of Bats". There are three major conponents, soft rasterization, LBS model, and Blender post-processing. 
## Soft rasterization
Our pipeline is based on Soft rasterization and need to be setup using: https://github.com/ShichenLiu/SoftRas and under "model" folder, there are a few scripts used for kinematic reconstruction and visualization.

     scale_model.py is used for reconstructing kinematic using silhouette image and manually designed template
     
## LBS model
  Our pipeline used the LBS model from https://github.com/marcbadger/avian-mesh (3D Bird Reconstruction A Dataset, Model, and Shape Recovery from a Single View)
  
      avian-mesh/models/LBS.py was imported in the scale_model.py above
## Blender template design / post-processing
  Blender_script: Contains three script for template design and export, kinematic import, and reconstruction visualization with camera array position. And the code is in bpy and need to be copy to Blender script console to use.

      Blender_script/camera_array_with_camera_matrix.txt  #for visualizing reconstruction with camera array location
      Blender_script/load_kinematics.txt                  # for loading kinematics to a refine template
      Blender_script/template_create.txt                  # for design the template and skeleton manually and export the template as .pkl file including information about template mesh (face, vertices, etc), 
                                                          # skeleton (initial position of head and tail), LBS matrix (linear blender skinning matirx)
   
