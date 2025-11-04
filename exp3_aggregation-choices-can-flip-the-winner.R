library(dplyr)
library(plyr)
library(tidyr)
library(stringr) 
library(ggplot2)
library(matrixStats)

############################
### Set paths
############################
# TODO set path for working directory
path <- ""
setwd(path)

# TODO set path for data directory
data_path <- ""

############################
### Define functions
############################

#' Calculates a frame-wise ranking
#' 
#' @param data A data frame including challenge results
#' @param fun Aggregation operator
#' @returns A data frame including the aggregated scores per algorithm and ranks
frame_wise_ranking <- function(data, fun = mean) {
  mean_per_algo <- aggregate(data$DSC, by=list(Category=data$team_name), FUN=fun)
  mean_per_algo$rank <- rank(-mean_per_algo$x, ties.method="min")
  colnames(mean_per_algo) <- c("Algorithm", "MeanScore", "Rank")
  return(mean_per_algo)
}

#' Calculates a video-wise ranking
#' 
#' @param data A data frame including challenge results
#' @param teams A vector with all team names
#' @param fun Aggregation operator
#' @returns A data frame including the aggregated scores per algorithm and ranks
video_wise_ranking <- function(data, teams, fun = mean) {
  per_video_rankings <- data.frame(matrix(nrow=length(teams), ncol=0))
  per_video_rankings <- cbind(per_video_rankings, sort(teams))
  
  for(v in unique(data$surgery)) {
    data_v <- subset(data, surgery == v)
    mean_per_algo_video <- aggregate(data_v$DSC, by=list(Category=data_v$team_name), FUN=fun)
    mean_per_algo_video$rank <- rank(-mean_per_algo_video$x, ties.method="min")
    
    per_video_rankings <- cbind(per_video_rankings, mean_per_algo_video$rank)
  }
  
  mean_rank_per_video <- data.frame(Category = per_video_rankings[, 1],
                                    MeanRank = rowMeans(per_video_rankings[, 2:11]))
  
  mean_rank_per_video$Rank <- rank(mean_rank_per_video$MeanRank, ties.method="min")
  
  return(mean_rank_per_video)
}

#' Calculates a phase-wise ranking
#' 
#' @param data A data frame including challenge results
#' @param teams A vector with all team names
#' @param fun Aggregation operator
#' @returns A data frame including the aggregated scores per algorithm and ranks
phase_wise_ranking <- function(data, teams, fun = mean) {
  per_phase_rankings <- data.frame(matrix(nrow=length(teams), ncol=0))
  per_phase_rankings <- cbind(per_phase_rankings, sort(teams))
  
  for(p in unique(data$phase)) {
    data_p <- subset(data, phase == p)
    mean_per_algo_phase <- aggregate(data_p$DSC, by=list(Category=data_p$team_name), FUN=fun)
    mean_per_algo_phase$rank <- rank(-mean_per_algo_phase$x, ties.method="min")
    
    per_phase_rankings <- cbind(per_phase_rankings, mean_per_algo_phase$rank)
  }
  
  mean_rank_per_phase <- data.frame(Category = per_phase_rankings[, 1],
                                    MeanRank = rowMeans(per_phase_rankings[, 2:ncol(per_phase_rankings)]))
  
  mean_rank_per_phase$Rank <- rank(mean_rank_per_phase$MeanRank, ties.method="min")
  
  return(mean_rank_per_phase)
}

#' Calculates a phase-wise video-wise ranking
#' 
#' @param data A data frame including challenge results
#' @param teams A vector with all team names
#' @param fun Aggregation operator
#' @returns A data frame including the aggregated scores per algorithm and ranks
per_phase_per_video_rankings <- function(data, teams, fun = mean) {
  per_phase_video_rankings <- data.frame(matrix(nrow=length(teams), ncol=0))
  per_phase_video_rankings <- cbind(per_phase_video_rankings, sort(teams))
  
  for(p in unique(data$phase)) {
    data_p <- subset(data, phase == p)
    
    for(v in unique(data_p$surgery)) {
      data_p_v <- subset(data_p, surgery == v)
      
      mean_per_algo_phase_video <- aggregate(data_p_v$DSC, by=list(Category=data_p_v$team_name), FUN=fun)
      mean_per_algo_phase_video$rank <- rank(-mean_per_algo_phase_video$x, ties.method="min")
      
      per_phase_video_rankings <- cbind(per_phase_video_rankings, mean_per_algo_phase_video$rank)
      colnames(per_phase_video_rankings)[ncol(per_phase_video_rankings)]  <- paste("rank", "phase", p, v, sep=".")
    }
  }
  
  return(per_phase_video_rankings)
}

#' Calculates a ranking per phase
#' 
#' @param data A data frame including challenge results
#' @param teams A vector with all team names
#' @param fun Aggregation operator
#' @returns A data frame including the aggregated scores per algorithm and ranks
phase_wise_video_wise_ranking <- function(data, teams, phase_video_rankings) {
  phase_videowise_rankings <- data.frame(matrix(nrow=length(teams), ncol=0))
  phase_videowise_rankings <- cbind(phase_videowise_rankings, sort(teams))
  
  for(p in unique(data$phase)) {
    phase_cols <- colnames(phase_video_rankings)[grep(paste("phase", p, sep="."), colnames(phase_video_rankings))]
    phase_rankings <- data.frame(phase_video_rankings[, phase_cols])
    
    mean_rankings_phase_videowise <- rowMeans(phase_rankings)
    
    phase_videowise_rankings <- cbind(phase_videowise_rankings, rank(mean_rankings_phase_videowise, ties.method="min"))
  }
  
  mean_rank_over_phases <- data.frame(Category = phase_videowise_rankings[, 1],
                                      MeanRank = rowMeans(phase_videowise_rankings[, 2:ncol(phase_videowise_rankings)]))
  
  mean_rank_over_phases$Rank <- rank(mean_rank_over_phases$MeanRank, ties.method="min")
  
  return(mean_rank_over_phases)
}

#' Calculates a video-wise phase-wise ranking
#' 
#' @param data A data frame including challenge results
#' @param teams A vector with all team names
#' @param fun Aggregation operator
#' @returns A data frame including the aggregated scores per algorithm and ranks
video_wise_phase_wise_ranking <- function(data, teams, phase_video_rankings) {
  video_phasewise_rankings <- data.frame(matrix(nrow=length(teams), ncol=0))
  video_phasewise_rankings <- cbind(video_phasewise_rankings, sort(teams))
  
  for(v in unique(data$surgery)) {
    vid_cols <- colnames(phase_video_rankings)[grep(paste("Sigma", v, sep="_"), colnames(phase_video_rankings))]
    vid_rankings <- data.frame(phase_video_rankings[, vid_cols])
    
    mean_rankings_video_phasewise <- rowMeans(vid_rankings)
    
    video_phasewise_rankings <- cbind(video_phasewise_rankings, rank(mean_rankings_video_phasewise, ties.method="min"))
  }
  
  mean_rank_over_vids <- data.frame(Category = video_phasewise_rankings[, 1],
                                    MeanRank = rowMeans(video_phasewise_rankings[, 2:ncol(video_phasewise_rankings)]))
  
  mean_rank_over_vids$Rank <- rank(mean_rank_over_vids$MeanRank, ties.method="min")
}

#' Calculates a weighted phase-wise ranking
#' 
#' @param data A data frame including challenge results
#' @param teams A vector with all team names
#' @param fun Aggregation operator
#' @returns A data frame including the aggregated scores per algorithm and ranks
weighted_phase_wise_ranking <- function(data, teams, fun = mean) {
  per_phase_rankings <- data.frame(matrix(nrow=length(teams), ncol=0))
  per_phase_rankings <- cbind(per_phase_rankings, sort(teams))
  colnames(per_phase_rankings) <- "Algorithm"
  
  for(p in unique(data$phase)) {
    data_p <- subset(data, phase == p)
    mean_per_algo_phase <- aggregate(data_p$DSC, by=list(Category=data_p$team_name), FUN=fun)
    mean_per_algo_phase$rank <- rank(-mean_per_algo_phase$x, ties.method="min")
    
    per_phase_rankings <- cbind(per_phase_rankings, mean_per_algo_phase$rank)
    colnames(per_phase_rankings)[ncol(per_phase_rankings)]  <- paste("rank", "phase", p, sep=".")
  }
  
  per_phase_rankings <- per_phase_rankings %>%select(sort(names(.)))
  
  # Weights based on phase importance 
  weights <- c(1, 3, 2, 1, 0, 2, 1, 1, 1, 3, 1)
  
  weighted_mean_ranks <- rowWeightedMeans(as.matrix(per_phase_rankings[,-1]), weights)
  
  weighted_mean_rank_per_phase <- data.frame(Category = per_phase_rankings[, 1],
                                             MeanRank = weighted_mean_ranks)
  
  weighted_mean_rank_per_phase$Rank <- rank(weighted_mean_rank_per_phase$MeanRank, ties.method="min")
  
  return(weighted_mean_rank_per_phase)
}

############################
### Read data
############################

# Binary segmentation for stage 3 of the RobustMIS challenge
data_bin <- read.csv(paste(data_path, "robustmis_results_concat_binary_segmentation.csv", sep=""))
data_bin_s3 <- subset(data_bin, stage == "Stage_3")

data_bin_s3[is.na(data_bin_s3)] <- 0
data_bin_s3 <- data_bin_s3 %>% mutate(surgery = paste(surgery_type,surgery_number, sep="_"))
data_bin_s3$phase <- as.numeric(data_bin_s3$phase)
data_bin_s3 <- subset(data_bin_s3, team_name != "human")

teams <- unique(data_bin_s3$team_name)

############################
### Calculate rankings
############################

# Set aggregation operator as 5th percentile (as in RobustMIS challenge)
aggr_operator = function(x) quantile(x, probs=0.05)

frame_wise_ranking_bin <- frame_wise_ranking(data_bin_s3, fun=aggr_operator)
video_wise_ranking_bin <- video_wise_ranking(data_bin_s3, teams, fun=aggr_operator)
phase_wise_ranking_bin <- phase_wise_ranking(data_bin_s3, teams, fun=aggr_operator)
per_phase_per_video_rankings_bin <- per_phase_per_video_rankings(data_bin_s3, teams, fun=aggr_operator)
phase_wise_video_wise_ranking_bin <- phase_wise_video_wise_ranking(data_bin_s3, teams, per_phase_per_video_rankings_bin)
video_wise_phase_wise_ranking_bin <- phase_wise_video_wise_ranking(data_bin_s3, teams, per_phase_per_video_rankings_bin)
weighted_phase_wise_ranking_bin <- weighted_phase_wise_ranking(data_bin_s3, teams, fun=aggr_operator)

# Combine all rankings into one data frame
all_rankings_bin <- data.frame(frame_wise_ranking_bin$Algorithm)
all_rankings_bin$FrameWise <- frame_wise_ranking_bin$Rank
all_rankings_bin$VideoWise <- video_wise_ranking_bin$Rank
all_rankings_bin$PhaseWise <- phase_wise_ranking_bin$Rank
all_rankings_bin$PhaseVideo <- phase_wise_video_wise_ranking_bin$Rank
all_rankings_bin$VideoPhase <- video_wise_phase_wise_ranking_bin$Rank
all_rankings_bin$WeightedPhase <- weighted_phase_wise_ranking_bin$Rank

colnames(all_rankings_bin)[1] <- "Algorithm"

# Sort entries by frame-wise ranking
all_rankings_bin <- all_rankings_bin[order(all_rankings_bin$FrameWise, decreasing = FALSE),]

# Anonymize results
all_rankings_bin$Algorithm <- paste0("A", 1:nrow(all_rankings_bin))

# Reformat ranking results into a publication-style table:
# shows, for each rank, which algorithms achieved that 
# rank across all aggregation schemes

rank_list <- list()

for (col in colnames(all_rankings_bin)[2:7]) {
  # Group by rank and concatenate all algorithms sharing the same rank
  tmp <- aggregate(Algorithm ~ get(col), data = all_rankings_bin, FUN = function(x) paste(x, collapse = " "))
  colnames(tmp) <- c("Rank", col)
  rank_list[[col]] <- tmp
}

# Merge all ranking columns by Rank
rank_table_df <- Reduce(function(x, y) merge(x, y, by = "Rank", all = TRUE), rank_list)

rank_table_df <- rank_table_df[order(rank_table_df$Rank), ]

print(rank_table_df)

############################
### Calculate Kendall's tau
############################

# Calculate Kendall's tau compared to frame-wise ranking
kendall_bin <- c()

for(i in 3:7) {
  kendall_bin <- cbind(kendall_bin,
                       cor(all_rankings_bin$FrameWise, 
                           all_rankings_bin[,i], 
                           method="kendall"))
}

print(kendall_bin)

# Kendall's tau statistics
min(kendall_bin)
mean(kendall_bin)
median(kendall_bin)
max(kendall_bin)

############################
### Calculate rank changes
############################

ranking_changes <- as.vector(as.matrix(all_rankings_bin[, 2] - all_rankings_bin[,3:7]))
min(abs(ranking_changes))
mean(abs(ranking_changes))
median(abs(ranking_changes))
max(abs(ranking_changes))

length(ranking_changes[ranking_changes > 0])/length(ranking_changes)
length(ranking_changes[ranking_changes < 0])/length(ranking_changes) 
length(ranking_changes[ranking_changes == 0])/length(ranking_changes) 

############################
### Plot individual metric scores
############################
team_map <- setNames(
  paste0("A", 1:10),
  c("haoyun", "CASIA_SRL", "www",
    "fisensee", "Uniandes", "SQUASH",
    "Caresyntax", "Djh", "NCT", "VIE")
)

data_bin_s3$team_name <- team_map[as.character(data_bin_s3$team_name)]


data_bin_s3$team_name <- factor(data_bin_s3$team_name, 
                                levels = paste0("A", 1:10))

colors <- c("#009e73", "#d55e00", "#0072b2", "#f0e442", "#cc79a7", "#e69f00",
            "#56b4e9", "#000000", "#662d91", "#c69c6d")


ggplot(data_bin_s3, aes(x=team_name, y=DSC, color=team_name)) + 
  geom_boxplot(size=1.1) + 
  geom_jitter(position=position_jitter(0.2), alpha = 0.1) +
  scale_fill_manual(values=colors) +
  scale_color_manual(values=colors) +
  theme_bw()
