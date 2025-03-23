
# Detailed plan for Victory

The aim of this plan is to be a meta-plan as concrete as possible. This will help us keep an eye on the path the project is taking.

[Big project mistakes are usually](https://www.benkuhn.net/pjm/): not freaking out soon enough, slowly moving off-plan, not working on top-priority items, prioritizing wrongly and incomplete information about system behavior. This should help us avoid those.

Our primary concern is going fast and staying on the track we consciously choose to follow.

This plan is updated every Sunday by @Apros7 (Lucas Vilsen)

## Plan (in ranked order of priority)
1. Rewrite Arduino controls to be able to pick up an object and sort it in under 2 seconds.
2. Have a complete pipeline for recognizing and calculating object 3d positions running faster than 30it/min on MacBook.
3. Get Jetson running the pipeline faster than 20it/min and communicating with Arduino.
4. Write test plan for DTU Test 7. April
5. Be able to visualize robot movements using PyQt6 to better debug movement/image recognition issues.
6. Autoupload videos to SupaBase for labelling.
7. Autoupload information (Weight, CO2, classification) to SupaBase at least every 6 hours and show it on the EasyView Dashboard for DTU as a test user.


## Who is working on what?
- Lucas: 1, 2, 4, 6
- Erik: 3, 5


## Links:

- [Open questions](open-questions.md)
