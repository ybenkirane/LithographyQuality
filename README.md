# LithographyQuality
Analyzing the quality of a nanofabricated lithography pattern. 


To compute the quality of the lithography pattern, you can use a combination of techniques like image registration and structural similarity index measure (SSIM). Image registration will help you align the generated pattern with the original pattern, and SSIM will provide you with a quality score.

--> https://en.wikipedia.org/wiki/Structural_similarity#:~:text=SSIM%20is%20used%20for%20measuring,distortion%2Dfree%20image%20as%20reference.


Goals: 

1) Iterate overlays for multiple sub-sections of a larger lithography array. 

2) Plot the quality score as a multivariable function of input lithography parameters

3) Find optimal score within a error domain of input parameters
        Incorporate Principal Component Analysis (PCA) as there may be too many variables to adequatly visualize... 
        https://builtin.com/data-science/step-step-explanation-principal-component-analysis
        --> Embedding : https://www.toptal.com/machine-learning/embeddings-in-machine-learning#:~:text=An%20embedding%20is%20a%20low,for%20a%20particular%20data%20structure.

4) Do the same with final post-LiftOff patterns and compare quality scores. 
    
