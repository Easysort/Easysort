# Projects and subtasks

This document outlines the projects needed to prove robotic sorting can be done reliably and at a low cost.
Truely scaling operations comes after.

Each project has a list of subtasks. Some subtasks might be missing, and will be added when we find them.

This list is updated every Sunday by @Apros7 (Lucas Vilsen)

### Reliable tracking and path planning
We need to reliably be able to track the position of an object in 3d space, it's position over time on the conveyor belt, and plan the best path for the robot to pick up the object. We use an Intel Realsense D435 Stereo Camera. The inference pipeline outputs a bounding box and a pickup point. Based on the center point, we can get a 3d position of the object at a specific time. This point should then be tracked, a plan should be made, and the robot should perform the path to pick up the object.

- [x] Get pickup point from AI inference pipeline
- [x] Get depth data from pickup point
- [ ] Calculate 3d position of object accurately (<1cm error)
- [ ] Track object over time
- [ ] Be able to follow object over time with the robot
- [ ] Calculate path to pickup object with the robot ([Issue #39](https://github.com/Easysort/Easysort/issues/39))

### Automated data collection and selection for labelling
Data collection is happening at all times when the robot is running. The data should be automatically uploaded. The images should be run through our model, based on active learning methods (other uncertainty methods?) pick the datapoints needed for training, and automate the labelling process.

- [x] Upload data automatically 
- [ ] Run images through model
- [ ] Perform active learning to select datapoints for labelling
- [ ] Send datapoints to be labelled
- [ ] Workflow for downloaded labelled data
- [ ] Released first 1000 labelled images for public use

### Standardization of training and inference
The training and inference pipeline should be standardised. This will allow us to test new models and ideas faster.

- [ ] Standardize training pipeline
- [ ] Standardize inference pipeline
- [ ] Use Neptune for logging
- [ ] Be able to evaluate any model from Neptune with a single workflow

### Stability and speed of robot
We need to be able to run the robot at full speed with no stability issues. This requires a proper frame design, figuring out how to mount the robot, and test of speed and limitation.

- [ ] Design frame
- [ ] Manifacture frame
- [ ] Test stability
- [ ] Design robot mount
- [ ] Construct robot mount
- [ ] Test speed

### Reliable operation
A realiable process for how to process a bag of waste from drop off to sorting to output fractions needs to be established. What exactly is required for this to happen, how to we setup a small scale operation to test our approach, and what should the floor plan look like?

- [ ] Design floor plan
- [ ] Process requirements
- [ ] Design process flow

### Product market fit
We need to find the right customers. Our long vision is set, but we need to find the right customers in the short term to start scaling operations and make the business sustainable.
