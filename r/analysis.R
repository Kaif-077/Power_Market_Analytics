# R version of one analysis: test-period error of the wind forecasts and a day-block bootstrap against ENTSO-E.
# Base R only. Run from the project root:   Rscript r/analysis.R
# Verified: run on R 4.1.1 by the author; the numbers below matched the Python pipeline exactly. The bootstrap
# interval differs slightly from Python's because random numbers differ (R: -116.5 to -17.8; Python: about -117 to -19).
# Expected output:
#   entsoe_benchmark          MAE 1387.3  RMSE 2194.4  bias  55.4
#   hybrid_entsoe_correction  MAE 1323.3  RMSE 2025.2  bias  27.9
#   gbm_weather_plus_obs      MAE 2182.1  RMSE 3069.5  bias -322.1
#   hybrid minus ENTSO-E (daily MAE difference): about -64 MW, 95% interval about -117 to -19

p <- read.csv("results/predictions_test.csv", check.names = FALSE)
names(p)[1] <- "time_utc"
models <- c("entsoe_benchmark", "hybrid_entsoe_correction", "gbm_weather_plus_obs")

stats <- function(m) {
  ok <- !is.na(p[[m]]) & !is.na(p$actual)
  e <- p[[m]][ok] - p$actual[ok]
  c(n = sum(ok), MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), bias = mean(e))
}
tab <- t(sapply(models, stats))
print(round(tab, 1))
write.csv(round(tab, 1), "results/r_metrics_test.csv")

# Day-block bootstrap: resample whole days, difference in daily MAE (hybrid minus ENTSO-E).
ok <- complete.cases(p[, c(models[1:2], "actual")])
q <- p[ok, ]
q$day <- substr(q$time_utc, 1, 10)
daily <- aggregate(cbind(hybrid = abs(q$hybrid_entsoe_correction - q$actual),
                         entsoe = abs(q$entsoe_benchmark - q$actual)) ~ q$day, FUN = mean)
d <- daily$hybrid - daily$entsoe
set.seed(1)
boots <- replicate(4000, mean(sample(d, length(d), replace = TRUE)))
cat(sprintf("\nDays: %d | mean daily MAE difference: %.1f MW | 95%% interval: %.1f to %.1f | share of days better: %.2f\n",
            length(d), mean(d), quantile(boots, 0.025), quantile(boots, 0.975), mean(d < 0)))

dir.create("results/figures", showWarnings = FALSE)
png("results/figures/r_mae_by_model.png", width = 700, height = 420)
barplot(tab[, "MAE"], names.arg = c("ENTSO-E", "Hybrid", "Weather model"), ylab = "MAE (MW)",
        main = "Wind forecast error, test period (R)", col = c("grey60", "steelblue", "grey80"))
dev.off()
