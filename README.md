# Mask disclosure discrete robustness

The deep learning model has achieved great success in the field of the image. By converting pixels in images into effective representations, downstream tasks can be completed according to representations. However, the deep learning model is not robust in the process of converting pixels in an image into effective representations. Against attacks, the recognition of the model can be rendered invalid by carrying out undetectable pixel perturbation on the image. Rather than just countering perturbations, we found a huge gap between the deep learning model and human vision, which relies on discrete pixels on an image to learn effective representations. By learning discrete pixel features, the model can convert discrete pixel features into effective representations. The model trained by mask images will lose the generalization of its original image but can improve the ability to use mask information (if the model only learns continuous pixels, it will be vulnerable to discrete mask attacks; If only discrete mask is learned to process images, the performance of the original test set will be lost). The ability of the model trained by mask to capture sparse pixel information can reach an incredible level, and the detection performance can reach more than 50% with only a few pixels. The model trained by the original training set and the mask training set can not only have high generalization performance but also have robustness against mask attacks.

You can find the mask method in file mask.py. 

## Mask Attack

* $\alpha$ : mask  high significance continuous area 
* $\beta$:  randomly mask continuous area
* $\gamma$: randomly mask discrete area

| mask ratio | $\alpha$ | $ \beta $ | $ \gamma $ |
| ---------- | :------- | --------- | :--------- |
| 0 %        | 93.89    | 93.89     | 93.89      |
| 2 %        | 89.98    | 90.20     | 65.14      |
| 5 %        | 86.90    | 87.73     | 45.96      |
| 10 %       | 81.48    | 82.00     | 27.64      |
| 15 %       | 74.12    | 75.49     | 16.00      |
| 20 %       | 67.11    | 68.36     | 14.07      |