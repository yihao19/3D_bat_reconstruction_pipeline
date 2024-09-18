# Source code for 3D bat reconstruction pipeline
  Here is the repo contains all the source code for our work "A Model-Based Deep-Learning Approach to Reconstructing the Highly Articulated Flight Kinematics of Bats". There are three major conponents, soft rasterization, LBS model, and Blender post-processing. 
## Soft rasterization
Our pipeline is based on Soft rasterization and need to be setup using: https://github.com/ShichenLiu/SoftRas and under "model" folder, there are a few scripts used for kinematic reconstruction and visualization.

     1."scale_model.py" is used for reconstructing kinematic using silhouette image and manually designed template
## LBS model
  Our pipeline used the LBS model from https://github.com/marcbadger/avian-mesh (3D Bird Reconstruction A Dataset, Model, and Shape Recovery from a Single View)
## Blender template design / post-processing
  1. Blender_script: Contains three script for template design and export, kinematic import, and reconstruction visualization with camera array position
   
