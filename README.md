# Current validation practice undermines surgical AI development
This is the code associated with the paper Reinke et al. "Current validation practice undermines surgical AI development" (2025). It contains the code for the following three experiments:
1. Dependent test samples inflate confidence
2. Averages hide critical failures
3. Aggregation choices can flip the winner

Please note that the individual challenge results (i.e., metric scores per algorithm) are not publicly available, as participant-level consent for data sharing was not obtained.

## Dependent test samples inflate confidence
This experiment investigated how ignoring data dependencies in temporally structured surgical video data can lead to severely underestimated model uncertainty. To ensure relevance across both low-level and high-level prediction tasks, we focused on two widely used benchmark tasks, instrument segmentation (RobustMIS) [[Roß et al., 2021]](https://www.sciencedirect.com/science/article/pii/S136184152030284X) and surgical action recognition (CholecT45) [[Nwoye et al. 2023]](https://www.sciencedirect.com/science/article/abs/pii/S1361841523000646). Both datasets exhibit a clear hierarchical structure, with multiple correlated frames per patient case

The code is stored in the files `exp1_dependent-test-samples-inflate-confidence_segmentation.py` for the instrument segmentation task and `exp1_dependent-test-samples-inflate-confidence_triplet-recognition.py` for the action triplet recognition task. Code is written in Python. 

## Averages hide critical failures
This experiment investigated whether global (non-stratified) aggregation of metric scores can conceal algorithm weaknesses under challenging image conditions. To enable stratified analysis across clinically relevant image characteristics, we focused on the RobustMIS dataset, for which we had access to structured metadata on visual artifacts or image properties [[Roß et al., 2023]](https://www.sciencedirect.com/science/article/pii/S1361841523000269). While the original study [[Roß et al., 2023]](https://www.sciencedirect.com/science/article/pii/S1361841523000269) employed these annotations to analyze model robustness across visual conditions, our analysis focused on how global aggregation can obscure property-dependent performance differences that are critical for assessing validation reliability. Specifically, we analyzed the multi-instance segmentation task of the RobustMIS challenge [[Roß et al., 2021]](https://www.sciencedirect.com/science/article/pii/S136184152030284X), using the MI_DSC scores from the seven participating algorithms

The code is stored in the file `exp2_averages-hide-critical-failures.R`. Code is written in R. 

## Aggregation choices can flip the winner
This experiment investigated how different aggregation strategies affect algorithm rankings, given that aggregation schemes are rarely reported in practice. We focused on the data from the RobustMIS challenge [[Roß et al., 2021]](https://www.sciencedirect.com/science/article/pii/S136184152030284X), as we had access to frame-level performance scores from all participating algorithms, enabling systematic comparison across different aggregation strategies. Specifically, we used the DSC scores of the ten participants of the binary segmentation task to simulate alternative ranking outcomes.

The code is stored in the file `exp3_aggregation-choices-can-flip-the-winner.R`. Code is written in R. 
