# Semantic Segmentation

Creating a Semantic Segmentation Model trained using a UNet Model with Skip Connections which can be trained on any model with any number of classes.

---

### 1. What it contains

**a.** <u>**utils.py**</u> - it contains the definition for all the classes and helper function (TBD)

**b.** <u>**determinemasks.py**</u> - it contains functionality for determining the number of masks and the associated unique colors, if unknown. It helps preparing the dataset for the Multi-Class Segmentation

---
### 2. An Example

![Alt Text](analysis/comparison_7.png)

The image on the right shows what my implementation learned after **50** epochs training with a **0.80-0.10-0.20** split of the dataset.

---

Here is the [MIT License](LICENSE)