# people_detection_2024
detect numbers of people from video stream

## Purpose
Create an inference solution that can observe a real-time camera stream and will raise an alert.
If more than n (for example 4 ) people are visible continuously for more than 2 minutes. 
Mark a region to observe, if necessary.

The code should have a script that can take an image or video and demo the alerts. 
Use free videos, for example videos at https://www.pexels.com/search/videos/crowd/

## How to do

### Create video streaming handle 
An example to handle video stream is Open CV library

### Create classifier model 
In order to detect object, like "person", classifier is needed, for example with Yolo model.
For the yolo model, COCO classes/labels name may be needed.
Yolo model works in this case frame by frame.

### Calculate people  
Calculate objects, with label "person", in a frame.
If the number of "person" more than n (for example 4), mark the frame id.

### Calculate time 
Calculate/Get the frame rate of the video. 
Time is calculated based on frame rate, start frame, and time threshold.
Minimal number of frame is time threshold multiplied by frame rate.
Get start frame, as the first frame marked from "people calculation" above.
If minimal number number of frame is exceeded, then create an alert.

