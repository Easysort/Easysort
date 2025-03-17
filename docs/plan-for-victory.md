
# Detailed plan for Victory

The aim for this plan is to be a meta-plan as concrete as possible. This will help us keep an eye on the path the project is taking.

Big project mistakes are usually: not freaking out soon enough, slowly moving off-plan, not working on top-priority items, prioritizing wrongly and incomplete information about system behaviour. This should help us avoid those.

We care about going fast. As long as velocity is high, a specific timeline is useless to us. 

This plan is updated every sunday by @Apros7 (Lucas Vilsen)

## Plan (in ranked order of priority)
1. Be able to turn suction on/off from arduino
2. Rewrite arduino controls to be able to pick up and object and sort in under 2 seconds.
3. Fix robot stabilization issues with regards to Z axis to under 1cm each direction.
4. Have a complete pipeline for recognizing and calculating object 3d positions running faster than 30it/min on macbook.
5. Get Jetson working and running the pipeline faster than 20it/min and communcating with Ardunio.
6. Be able to visualize robot movements using PyQt6 to better debug movement/image recognition issues.
7. Autoupload information (Weight, CO2, classification) to supabase at least every 6 hours and show it on the EasyView Dashboard for DTU as a test user.


## Who is working on what?
- Lucas: 1, 3, 4
- Erik: 2, 5, 6


## Links:

- [Open questions](open-questions.md)