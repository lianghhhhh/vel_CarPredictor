## path_planner
-> plan a path to avoid obstacles for car to run on
- input: current velocity, delta state *10, obstacles *5
- output: planned path points (x, y, angle) *3

## src: car_predictor
-> using car's target velocity and current velocity to predict delta state

## vel_predictor
-> using car's current velocity and delta state to predict target velocity
