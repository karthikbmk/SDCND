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
    Front camera  
    ![alt text][f_sample]  
    Left Camera  
    ![alt text][l_sample]  
    Right Camera    
    ![alt text][r_sample]  
     
 Data Pre-processing 
 ---
 * In order to furthur augment the size of the dataset, I flipped the images (left to right), and reversed the steering angle, and add them to the dataset
 * Also, the images were cropped a bit, as there were useless information in the incoming images such as trees etc.
 
 Training
 ---
 * Here is the architecture of the model :
 
Layer (type)      | Output Shape | Param #   
------------------|-------------------------|--------------------|
lambda_1 (Lambda)        |    (None, 160, 320, 3)       | 0         
cropping2d_1 (Cropping2D)   | (None, 70, 320, 3)        | 0        
conv2d_1 (Conv2D)           | (None, 66, 316, 32)      | 2432      
max_pooling2d_1 (MaxPooling2 (None, 33, 158, 32)      | 0         
conv2d_2 (Conv2D)        |    (None, 31, 156, 64)     |  18496     
max_pooling2d_2 (MaxPooling2d) | (None, 15, 78, 64)      |  0         
conv2d_3 (Conv2D)        |    (None, 13, 76, 128)     |  73856     
conv2d_4 (Conv2D)        |    (None, 11, 74, 256)     |  295168    
dropout_1 (Dropout)       |   (None, 11, 74, 256)     |  0         
conv2d_5 (Conv2D)        |     (None, 9, 72, 512)     |   1180160   
max_pooling2d_3 (MaxPooling2D) | (None, 4, 36, 512)     |   0         
flatten_1 (Flatten)      |    (None, 73728)           |  0         
dense_1 (Dense)         |     (None, 64)              |  4718656   
dropout_2 (Dropout)    |      (None, 64)              |  0         
dense_2 (Dense)       |       (None, 32)              |  2080      
dropout_3 (Dropout)  |         (None, 32)              |  0         
dense_3 (Dense)    |          (None, 16)              |  528       
dense_4 (Dense)   |           (None, 1)               |  17        

Total params: 6,291,393  
Trainable params: 4,721,281  
Non-trainable params: 1,570,112  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
[//]: # (Image References)
[f_sample]: ./misc/front_sample.jpg "Front view"
[l_sample]: ./misc/left_sample.jpg "Left view"
[r_sample]: ./misc/right_sample.jpg "Right view"
