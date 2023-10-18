# Instance Segmentation


## Project assumptions

- real-time work using RGBD RealSense D435 camera; 
- static objects detection; 
- objects size limited from 0.1[m]x0.01[m]x0.01[m] to 1[m]x1[m]x1[m]; 


## Expected final accuracy

- segmentation accuracy > 80% 
- IoU > 60%. 


## Tech stack

- **language:** Python 3.7
- **libraries:**
    - PyTorch - neural network building
    - OpenCV - vision processing
    - Intel RealSense SDK 2.0 -  connection with RealSense
    - PCL, Open3D - voxelisation
    - NumPy, scikit-learn - 3D bounding boxes

## Developers
- Mateusz Koloszko
- Gabriela Koncewicz
- Jakub Pilarski
- Weronika Kapusta
- Tomasz Szymanek
- Paweł Iskrzycki