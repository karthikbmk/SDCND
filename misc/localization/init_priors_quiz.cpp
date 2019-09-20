#include <iostream>
#include <vector>

using std::vector;
using std::cout;
using std::endl;

// initialize priors assuming vehicle at landmark +/- 1.0 meters position stdev
vector<float> initialize_priors(int map_size, vector<float> landmark_positions,
                                float position_stdev);

int main() {
  // set standard deviation of position
  float position_stdev = 1.0f;

  // set map horizon distance in meters 
  int map_size = 25;

  // initialize landmarks
  vector<float> landmark_positions {5, 10, 20};

  // initialize priors
  vector<float> priors = initialize_priors(map_size, landmark_positions,
                                           position_stdev);

  // print values to stdout 
  for (int p = 0; p < priors.size(); ++p) {
    std::cout << priors[p] << std::endl;
  }

  return 0;
}

// TODO: Complete the initialize_priors function
vector<float> initialize_priors(int map_size, vector<float> landmark_positions,
                                float position_stdev) {

  // initialize priors assuming vehicle at landmark +/- 1.0 meters position stdev

  // set all priors to 0.0
  vector<float> priors(map_size, 0.0);
    
  // TODO: YOUR CODE HERE
  float init_blf = 1.0 / (landmark_positions.size() * (position_stdev*2 + 1));
  
  for (float i = 0 ; i< priors.size(); i ++)
  {
      for (float j = 0 ; j <landmark_positions.size() ; j++) 
      {
          if (i <= landmark_positions[j] + position_stdev && i >= landmark_positions[j] - position_stdev)
            {
                priors[i] = init_blf;
            }
      }
  }
  
  
  return priors;
}
