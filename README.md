To run this on a set of files do the following commmand
`python global_prediction.py --data_directory /path/to/data`

When running on files I noticed an average `0.011` `mAP` improvement from the initial predictions and an average `0.07` improvement in `Miou`


For this solution I implemented the Global Optimization step from this paper https://arxiv.org/pdf/2004.02678.pdf as well as using the parameter optimization step found in the supplementary materials. My algorithm consists of two steps
 * Using DP to calculate the number of scenes to combine the supershots in as well as keeping a position matrix to backtrack through to get the scenes
 * Running SGD on the shots themselves to find the optimal similarity score given the scene configuration obtained from above

 I then repeat both those steps until I got a solution. Since the paper didn't specify how to handle the multimodal aspect of the data I made the following assumptions
 * To calculate (cosine) similarity between two supershots I should calculate it for each modality separately and combine them. Since I didn't have the compute to learn parameters I decided to follow the ratio the authors used in the github repo
 * For parameters each shot shoud have 4 parameter matrices each corresponding to a separate modality

