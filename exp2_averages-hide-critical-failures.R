library(dplyr)
library(plyr)
library(stringr) 
library(ggplot2)
library(tidyr)
library(matrixStats)

############################
### Set paths
############################
# TODO set path for working directory
path <- ""

# TODO set path for data directory
data_path <- ""
setwd(path)

############################
### Define functions
############################

#' Calculates the median MI_DSC over all frames per algorithm and 
#' creates a ranking based on median scores
#' 
#' @param dat A data frame including challenge results
#' @returns A data frame including the median aggregate score and rank per algorithm
create_ranking <- function(dat) {
  ranking <- aggregate(dat$instance_dice_coefficient, list(dat$team_name), 
                       FUN = median, na.rm = TRUE)
  ranking$rank <- rank(-ranking$x)
  ranking = ranking[order(ranking$rank),]
  colnames(ranking) <- c("Algorithm", "Aggregate", "Rank")
  
  return(ranking)
}

#' Generates a data frame with ranks across stratified artifacts for one algorithm 
#' 
#' @param algo A string with the team identifier
#' @returns A vector with the team identifier and ranks across stratified artifacts
ranks_per_algo <- function(algo) {
  rank_per_algo <- c(algo, 
                     ranking_full[ranking_full$Algorithm %in% algo, ]$Rank, 
                     ranking_blood[ranking_blood$Algorithm %in% algo, ]$Rank, 
                     ranking_motion[ranking_motion$Algorithm %in% algo, ]$Rank, 
                     ranking_reflections[ranking_reflections$Algorithm %in% algo, ]$Rank, 
                     ranking_smoke[ranking_smoke$Algorithm %in% algo, ]$Rank, 
                     ranking_cov_instru[ranking_cov_instru$Algorithm %in% algo, ]$Rank, 
                     ranking_overexposed[ranking_overexposed$Algorithm %in% algo, ]$Rank, 
                     ranking_underexposed[ranking_underexposed$Algorithm %in% algo, ]$Rank, 
                     ranking_material[ranking_material$Algorithm %in% algo, ]$Rank,
                     ranking_low_artifacts[ranking_low_artifacts$Algorithm %in% algo, ]$Rank)
  
  return(rank_per_algo)
}

#' Generates a data frame with median scores over frames across stratified 
#' artifacts for one algorithm 
#' 
#' @param algo A string with the team identifier
#' @returns A vector with the team identifier and median scores
#' across stratified artifacts
aggr_per_algo <- function(algo) {
  agg_per_algo <- c(algo,
                    round(ranking_full[ranking_full$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_blood[ranking_blood$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_motion[ranking_motion$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_reflections[ranking_reflections$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_smoke[ranking_smoke$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_cov_instru[ranking_cov_instru$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_overexposed[ranking_overexposed$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_underexposed[ranking_underexposed$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_material[ranking_material$Algorithm %in% algo, ]$Aggregate,2),
                    round(ranking_low_artifacts[ranking_low_artifacts$Algorithm %in% algo, ]$Aggregate,2))
  
  return(agg_per_algo)
}

#' Bootstrap the difference in median DSC to no stratification (not hierarchical)
#' 
#' @param dat A data frame including challenge results
#' @param artifact_datasets A list of data frames including challenge results per artifact
#' @param ranking_full Ranking without stratification
#' @param n_boot Number of bootstraps
#' @param min_frames min number of frames
#' @returns Bootstrapped results
bootstrap_delta <- function(dat,
                              artifact_datasets,
                              ranking_full,
                              n_boot = 1000,
                              min_frames = 5) {
  
  bootstrap_results <- data.frame()
  
  for (artifact_name in names(artifact_datasets)) {
    dat_artifact <- artifact_datasets[[artifact_name]]
    
    for (team in unique(dat$team_name)) {
      
      dsc_artifactifact <- dat_artifact %>% filter(team_name == team) %>% pull(instance_dice_coefficient)
      dsc_full <- dat %>% filter(team_name == team) %>% pull(instance_dice_coefficient)
      
      if (length(dsc_artifactifact) < min_frames | length(dsc_full) < min_frames) next
      
      # Sample median difference
      boot_diffs <- replicate(n_boot, {
        med_art <- median(sample(dsc_artifactifact, length(dsc_artifactifact), replace = TRUE), na.rm = TRUE)
        med_full <- median(sample(dsc_full, length(dsc_full), replace = TRUE), na.rm = TRUE)
        med_art - med_full
      })
      
      # Calculate confidence intervals
      ci_low <- quantile(boot_diffs, 0.025, na.rm = TRUE)
      ci_high <- quantile(boot_diffs, 0.975, na.rm = TRUE)
      
      bootstrap_results <- rbind(
        bootstrap_results,
        data.frame(
          Algorithm = team,
          Artifact = artifact_name,
          CI_Low = ci_low,
          CI_High = ci_high
        )
      )
    }
  }
  
  return(bootstrap_results)
}

#' Bootstrap the difference in median DSC to no stratification (hierarchical)
#' 
#' @param dat A data frame including challenge results
#' @param artifact_datasets A list of data frames including challenge results per artifact
#' @param ranking_full Ranking without stratification
#' @param n_boot Number of bootstraps
#' @returns Bootstrapped results
bootstrap_delta_hierarchical <- function(dat, artifact_datasets, ranking_full,
                                         n_boot = 1000) {
  bootstrap_results <- data.frame()
  
  for (artifact_name in names(artifact_datasets)) {
    print(paste("====== Bootstrap artifact:", artifact_name, sep=" "))
    dat_artifact <- artifact_datasets[[artifact_name]]
    
    for (team in unique(dat$team_name)) {
      print(paste("-- Team: ", team, sep=" "))
      dat_artifact_team <- dat_artifact %>% filter(team_name == team)
      dat_full_team <- dat %>% filter(team_name == team)
      
      vids_artifact <- unique(dat_artifact_team$surgery_number)
      vids_full <- unique(dat_full_team$surgery_number)
      
      common_vids <- intersect(vids_artifact, vids_full)
      
      # Sample median difference hierarchically (over video level)
      boot_diffs <- replicate(n_boot, {
        sampled_vids <- sample(common_vids, length(common_vids), replace = TRUE)
        
        sample_art <- dat_artifact_team %>% filter(surgery_number %in% sampled_vids)
        sample_full <- dat_full_team %>% filter(surgery_number %in% sampled_vids)
        
        boot_artifact <- c()
        boot_full <- c()
        
        for (vid in sampled_vids) {
          dsc_artifact <- dat_artifact_team %>% filter(surgery_number == vid) %>% pull(instance_dice_coefficient)
          dsc_full <- dat_full_team %>% filter(surgery_number == vid) %>% pull(instance_dice_coefficient)
          
          boot_artifact <- c(boot_artifact, sample(dsc_artifact, length(dsc_artifact), replace = TRUE))
          boot_full <- c(boot_full, sample(dsc_full, length(dsc_full), replace = TRUE))
        }
        
        med_art <- median(boot_artifact, na.rm = TRUE)
        med_full <- median(boot_full, na.rm = TRUE)
        
        med_art - med_full
      })
      
      ci_low <- quantile(boot_diffs, 0.025, na.rm = TRUE)
      ci_high <- quantile(boot_diffs, 0.975, na.rm = TRUE)
      
      bootstrap_results <- rbind(
        bootstrap_results,
        data.frame(
          Algorithm = team,
          Artifact = artifact_name,
          CI_Low = ci_low,
          CI_High = ci_high
        )
      )
    }
  }
  
  return(bootstrap_results)
}

############################
### Read data
############################

# Challenge data
data_matrix=read.csv(paste(data_path, 'robustmis_results_concat_no_detection.csv', sep=""), header = T)
data_matrixMIS <- subset(data_matrix, task=="Testcase.MULTIPLE_INSTANCE_SEGMENTATION" & stage=="Stage_3")
data_matrixMIS <- data_matrixMIS %>% mutate(frame = paste(frame_nr,surgery_type,surgery_number, sep="_")) 
data_matrixMIS <- subset(data_matrixMIS, team_name != "Djh")
data_matrixMIS <- subset(data_matrixMIS, team_name != "Prometheus")

# Meta data (artifacts)
meta_dat <- read.csv(paste(data_path, "meta_data_aggregated.csv", sep=""))
meta_dat_stage3 <- data.frame(subset(meta_dat, stage=="Stage_3"))

# Combine into single data frame
dat_comb <- merge(data_matrixMIS, meta_dat_stage3, by="frame")

#########################################
### Calculate rankings and metric scores
#########################################

# Full data set
ranking_full <- create_ranking(dat_comb)
n_cases_full <- length(unique(dat_comb$frame))

# Ranking blood
dat_blood <- subset(dat_comb, toggle_background_blood == 1 | toggle_instrument_blood == 1)
n_cases_blood <- length(unique(dat_blood$frame))
ranking_blood <- create_ranking(dat_blood)

# Ranking motion
dat_motion <- subset(dat_comb, toggle_background_motion == 1 | toggle_instrument_motion == 1)
n_cases_motion <- length(unique(dat_motion$frame))
ranking_motion <- create_ranking(dat_motion)

# Ranking covered by material
dat_material <- subset(dat_comb, toggle_instrument_covered_material == 1)
n_cases_material <- length(unique(dat_material$frame))
ranking_material <- create_ranking(dat_material)

# Ranking reflections
dat_reflections <- subset(dat_comb, toggle_background_reflections == 1 | toggle_instrument_covered_reflections == 1)
n_cases_reflections <- length(unique(dat_reflections$frame))
ranking_reflections <- create_ranking(dat_reflections)

# Ranking smoke
dat_smoke <- subset(dat_comb, toggle_background_smoke == 1 | toggle_instrument_smoke == 1)
n_cases_smoke <- length(unique(dat_smoke$frame))
ranking_smoke <- create_ranking(dat_smoke)

# Ranking instrument covered instrument
dat_cov_instru <- subset(dat_comb, toggle_instrument_covered_instrument == 1)
n_cases_cov_instru <- length(unique(dat_cov_instru$frame))
ranking_cov_instru <- create_ranking(dat_cov_instru)

# Ranking overexposed
dat_overexposed <- subset(dat_comb, toggle_instrument_overexposed == 1)
n_cases_overexposed <- length(unique(dat_overexposed$frame))
ranking_overexposed <- create_ranking(dat_overexposed)

# Ranking underexposed
dat_underexposed <- subset(dat_comb, toggle_instrument_underexposed == 1)
n_cases_underexposed <- length(unique(dat_underexposed$frame))
ranking_underexposed <- create_ranking(dat_underexposed)

# Ranking for frames with one or less of the above artifacts
dat_low_artifacts <- subset(dat_comb, 
                     (toggle_background_blood + toggle_instrument_blood +
                        toggle_background_motion + toggle_instrument_motion +
                        toggle_instrument_covered_material +
                        toggle_background_reflections + toggle_instrument_covered_reflections +
                        toggle_background_smoke + toggle_instrument_smoke +
                        toggle_instrument_covered_instrument +
                        toggle_instrument_overexposed + toggle_instrument_underexposed) <= 1)

n_cases_low_artifacts <- length(unique(dat_low_artifacts$frame))
ranking_low_artifacts <- create_ranking(dat_low_artifacts)

artifact_datasets <- list(
  "Blood" = dat_blood, 
  "Motion" = dat_motion, 
  "Reflections" = dat_reflections, 
  "Smoke" = dat_smoke, 
  "Material" = dat_material, 
  "Overexposed" = dat_overexposed, 
  "Underexposed" = dat_underexposed, 
  "Instrument Covered" = dat_cov_instru, 
  "Low Artifacts" = dat_low_artifacts 
  )

## Combine all rankings in one data frame
rankings_combined <- data.frame(matrix(ncol = 11, nrow = 0))
for (alg in ranking_full$Algorithm) {
  rankings_combined <- rbind(rankings_combined, ranks_per_algo(alg))
}
colnames(rankings_combined) <- c("Team", "Rank Full", "Rank Blood", "Rank Motion",
                             "Rank Reflections", "Rank Smoke",
                             "Rank Instrument covered", "Rank overexposed",
                             "Rank underexposed", "Rank material",
                             "Rank low artifacts")

## Combine all aggregated scores in one data frame
aggregates_combined <- data.frame(matrix(ncol = 11, nrow = 0))
for (alg in ranking_full$Algorithm) {
  aggregates_combined <- rbind(aggregates_combined, aggr_per_algo(alg))
}
colnames(aggregates_combined) <- c("Team", "Rank Full", "Rank Blood", "Rank Motion",
                              "Rank Reflections", "Rank Smoke",
                              "Rank Instrument covered", "Rank overexposed",
                              "Rank underexposed", "Rank material",
                              "Rank low artifacts")
n_cols <- ncol(aggregates_combined)

for (i in 2:n_cols) {
  aggregates_combined[, i] <- as.numeric(aggregates_combined[, i])
}

########################################
### Calculate data distribution sizes
########################################
n_artifact <- data.frame(Artifact = c("Full", "Blood", "Motion","Reflections", 
                                      "Smoke", "Instrument covered", 
                                      "Overexposed", "Underexposed", "Material",
                                      "Low artifacts"),
                         n = c(n_cases_full, n_cases_blood, n_cases_motion, 
                               n_cases_reflections, n_cases_smoke,
                               n_cases_cov_instru, n_cases_overexposed,
                               n_cases_underexposed, n_cases_material,
                               n_cases_low_artifacts))

n_artifact$Percentage <- round(n_artifact$n / n_cases_full * 100, 2)

#################################################
### Calculate median performances across teams
#################################################
aggregates_combined <- rbind(aggregates_combined, c("Median performance", round(colMedians(as.matrix(aggregates_combined[, 2:n_cols])), 2)))
median_performance <- data.frame(aggregates_combined[8, 2:n_cols])

median_performance <- rbind(median_performance, as.numeric(median_performance) - as.numeric(median_performance$Rank.Full))

colnames(median_performance) <- c("Rank Full", "Rank Blood", "Rank Motion",
                                "Rank Reflections", "Rank Smoke",
                                "Rank Instrument covered", 
                                "Rank overexposed", "Rank underexposed",
                                "Rank material", "Rank low artifacts")

#################################################
### Calculate difference to full
### data for individual algorithms 
#################################################
differences <- data.frame(matrix(ncol = 8, nrow = 0))
differences <- rbind(differences, median_performance[2,2:(n_cols-1)])

for(i in 1:7) {
  algo_difference <- aggregates_combined[i, 2:n_cols]
  algo_difference <- rbind(algo_difference, as.numeric(algo_difference) - as.numeric(algo_difference$`Rank Full`))
  differences <- rbind(differences, algo_difference[2,2:(n_cols-1)])
}

differences$Algorithm <- c("Median", "A1", "A2", "A3",
                           "A4", "A5", "A6", "A7")

differences_restructured <- 
  differences %>% 
  pivot_longer(
    cols     = starts_with("Rank"),
    names_to = "Artifact"
  )

differences_restructured$value <- as.numeric(differences_restructured$value)

differences_restructured$Artifact<-gsub("Rank ","",as.character(differences_restructured$Artifact))
differences_restructured$Artifact <- str_to_title(differences_restructured$Artifact)

differences_restructured$Algorithm <- factor(differences_restructured$Algorithm, c("Median", "A1", 
                                                                                   "A2", "A3", 
                                                                                   "A4", "A5", 
                                                                                   "A6", "A7"))

#################################################
### Calculate uncertainty of deltas 
#################################################
bootstrap_results <- bootstrap_delta_hierarchical(dat_comb, artifact_datasets, 
                                                  ranking_full, n_boot = 1000)

algo_map <- data.frame(
  Algorithm_real = ranking_full$Algorithm,
  Algorithm = paste0("A", seq_along(ranking_full$Algorithm))
)

bootstrap_results$Algorithm <- algo_map$Algorithm[
  match(bootstrap_results$Algorithm, algo_map$Algorithm_real)
]

differences_with_ci <- merge(
  differences_restructured,
  bootstrap_results,
  by.x = c("Algorithm", "Artifact"),
  by.y = c("Algorithm", "Artifact"),
  all.x = TRUE
)

differences_with_ci$Artifact[differences_with_ci$Artifact == "Instrument Covered"] <- "Intersecting\nInstruments"
differences_with_ci$Artifact[differences_with_ci$Artifact == "Low Artifacts"] <- "Low-artifact\nscenes"

differences_with_ci$Artifact <- factor(differences_with_ci$Artifact,
                                            c("Low-artifact\nscenes", "Blood",
                                              "Reflections", "Smoke",
                                              "Motion", "Material",
                                              "Overexposed", "Underexposed",
                                              "Intersecting\nInstruments"))

#################
### Plot results
#################

ggplot(differences_with_ci, aes(x = Artifact, y = value, fill = Algorithm)) +
  geom_bar(position = position_dodge(width = 0.9), stat = "identity") +
  geom_errorbar(
    aes(ymin = CI_Low, ymax = CI_High),
    position = position_dodge(width = 0.9),
    width = 0.3,
    linewidth = 0.6
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "right"
  ) +
  ylab("Difference in median DSC to no stratification") +
  xlab("Artifact")
