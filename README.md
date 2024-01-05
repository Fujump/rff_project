# Project for STA303(2023 fall)
## Abstract
Random Fourier Features (RFF) is a method to approximate the kernel function $k(\cdot, \cdot)$ by
   a random feature map $\phi(\cdot)$. It is considered as a breakthrough in kernel methods, since it
   makes kernel methods scalable to large datasets. It is widely used in kernel methods, such as
   Support Vector Machine (SVM), Gaussian Process Regression (GPR), etc. In this paper, we conduct
   experiments to investigate the effect of dimensionality and sampling distribution on the
   performance of RFF. We also compare the performance of RFF with other kernel methods, such as
   Support Vector Machine (SVM), Gaussian Process Regression (GPR), etc. We find that RFF is
   sensitive to the dimensionality and sampling distribution. We also find that RFF is comparable to
   other kernel methods in terms of accuracy, but it is much faster than other kernel methods in
   terms of training and inference time. Interestingly, we also find that RFF is comparable to other kernel methods in terms of interpretability.
## Contributions
Wang Ma contributed the main ideas and led others, Qiang Hu was responsible for the
random feature experiment and analysis, Jingjuan Huang was responsible for the GPR experiment and analysis,
and Junjie Qiu assisted Hu Qiang in the experiment and led the writing
## Package Requirements
```bash
pip install -r requirements.txt
```
## File Descriptions
The `utils/` includes the basic codes for rff;
<br>
The `commons/` includes codes to experiment RFFGPR, RFFRegression, svc and customed NN;
<br>
The `notebooks/` includes codes to visualize the results;
<br>
The `scripts/` includes some codes to conduct experiments on linux.