# Behavioral Cloning Project

The project was divided into these phases :  
* Data Collection
* Data Pre-processing
* Training
* Testing

Data Collection
---
* In order to cover as much scenarios as possible, the data was collected as follows :

    Method Name | Method Description  | Number of laps per track | Which track ?
    ------------- |------------- | -------------| ------------- | 
    Center Drive |Drive at the center of the lane  | 2 |  Both |
    Curve Drive |Drive smoothly around the curves  | 1| Both |
    Recovery Drive |Recover the car from the side of track onto the middle  | 1 | Both |
    
* The above methods were used collect data from these cameras : 
    * Front Camera
    * Left Camera
    * Right Camera
    
    
 * Dataset sample :  
    
    ![alt text] [f_sample] ![alt text][l_sample] ![alt text][r_sample]
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
[//]: # (Image References)
[f_sample]: ./misc/front_sample.jpg "Front view"
[l_sample]: ./misc/left_sample.jpg "Left view"
[r_sample]: ./misc/right_sample.jpg "Right view"
