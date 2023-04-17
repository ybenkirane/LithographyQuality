# LithographyQuality
Analyzing the quality of a nanofabricated lithography pattern. 


To compute the quality of the lithography pattern, you can use a combination of techniques like image registration and structural similarity index measure (SSIM). Image registration will help you align the generated pattern with the original pattern, and SSIM will provide you with a quality score.

--> https://en.wikipedia.org/wiki/Structural_similarity#:~:text=SSIM%20is%20used%20for%20measuring,distortion%2Dfree%20image%20as%20reference.


Goals: 
    Iterate overlays for multiple sub-sections of a larger lithography array. 
    Plot the quality score as a multivariable function of input lithography parameters
    Find optimal score within a error domain of input parameters
    Do the same with final post-LiftOff patterns and compare quality scores. 
