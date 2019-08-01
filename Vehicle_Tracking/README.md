# Sensor Fusion using Extended Kalman Filters
![alt text][video_gif]

### The steps of this project are the following:

* Estimate the position and velocity using Vehicle Kinematic equations
* Initialize various Co-variance matrices such as the Process Co-variance matrix, the state co-variance matrix etc.
* Compute the Estimation error and Measurement error
* Compute the Kalman Gain using errors from previous step
* For Lidar, compute the Kalman Gain using an Ordinary Kalman Filter
* For Radar, compute the Kalman Gain using Jacobians and the Extended Kalman Filter
* Use the Kalman Gain to update the estimate and estimation errors
* Repeat the above process for all measurements from the Sensor

### Flowchart of the pipeline:
![alt text][flowchart]

### Predict and Update equations:
![alt text][equations]

[//]: # (Image References)
[video_gif]: ./output/video.gif "Vehicle Tracking"
[flowchart]: ./output/flowchart.gif "Flowchart"
[equations]: ./output/equations.gif "Equations"
