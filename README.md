# RelPose

Re-implementation and extension of [relpose](https://arxiv.org/pdf/2208.05963.pdf). 

Relpose: Given two images of different views of an object, predicts the angle difference between the views.

In this project, we wish to extend relpose to text prompts. Given two images of different views of an object, optimize a CLIP textual representation that represents each view using [an image is worth one word](https://textual-inversion.github.io/). By doing so we end up with textual representation optimized for different views of an object. Then, we wish to train a model that predicts the angle difference between these textual representations (completely supervised). 

Therefore, we will end up with a model that takes two textual representations and predicts the angle difference between them. We want to use this model to guide each view of a text-to-3d Generative model with proper textual representation. For instance, Let's say we want to generate "a dog". First, we get the textual representation **R** of the prompt. Then, if we want to optimize the side view of the prompt (e.g., 80 degrees from the front view), we use **R** and a random representation **R2** to the network, and the network predicts an angle difference **A** between **R** and **R2**. We freeze all of the parameters in the network as well as **R** and optimize **R2** with an L2 Loss on the predicted angle **A** and the desired angle **80**. 

This way, for any desired angle, we can optimize a textual prompt. When optimizing different views of a nerf in text-to-3D models, we will be able to provide much better textual prompts and avoid all views being trained with the same prompt.

