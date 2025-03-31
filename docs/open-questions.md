
# Open questions

Question we don't know the answer to yet.

Will be updated every sunday by @Apros7 (Lucas Vilsen)

1. How do we scale data collection enough to be >80% objects categorized and >95% accuracy?

   Possible solutions：

   - **Manual Labeling**

     - **Advantages**: High dataset quality, ensuring the accuracy of labels.

     - **Disadvantages**: Time-consuming and labor-intensive.

   - **Using Generative Models to Expand the Dataset**

     - **Advantages**:
       - Generative models (such as GANs or Diffusion Models) can quickly generate new samples, expanding the dataset efficiently.
       - Based on my previous experience using GANs to augment a small medical dataset, I found that generative models have great potential in improving both classification performance and dataset diversity.

     - **Disadvantages**:

       - GAN training can be unstable, often leading to mode collapse, which results in inconsistent or even unusable generated data.

       - Diffusion Models are more stable than GANs but require more computational resources and time to generate high-quality samples.

   - Data Quality Control

     - **Strategy**: Remove erroneous, duplicated, or unclear data to enhance the overall quality of the dataset.

2. How do we calculated the trajectory of the robot so it moves most efficiently and >30 objects per minute? What format should be used (e.g gcode?)?





