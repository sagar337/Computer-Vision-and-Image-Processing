# KeypointDetection
Detect key points in the input image using SIFT
Write programs to detect keypoints in an image according to the following steps, which are also the rst three
steps of Scale-Invariant Feature Transform (SIFT).
1. Generate four octaves. Each octave is composed of ve images blurred using Gaussian kernels. For each
octave, the bandwidth parameters  (ve dierent scales) of the Gaussian kernels are shown in Tab. 1.
2. Compute Dierence of Gaussian (DoG) for all four octaves.
3. Detect keypoints which are located at the maxima or minima of the DoG images. You only need to provide
pixel-level locations of the keypoints; you do not need to provide sub-pixel-level locations.
In your report, please (1) include images of the second and third octave and specify their resolution (width 
height, unit pixel); (2) include DoG images obtained using the second and third octave; (3) clearly show all the

octave 
1st    1/sqrt(2)  1   sqrt(2)   2   2*sqrt(2)
2nd    sqrt(2)    2   2*sqrt(2) 4   4*sqrt(3)
3rd    2*sqrt(2)  4   4*sqrt(2) 8   8*sqrt(2)
4th   4*sqrt(2)   8    8*sqrt(2)  16  16*sqrt(2)

Table 1: The bandwidth parameters  (ve dierent scales) of the Gaussian kernels used in the rst step of
keypoint detection.
detected keypoints using white dots on the original image (4) provide coordinates of the ve left-most detected
keypoints (the origin is set to be the top-left corner).

