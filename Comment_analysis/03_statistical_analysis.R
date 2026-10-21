invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "English_United States.utf8")))
.libPaths(c("C:/Users/47050/AppData/Local/R/win-library/4.5", .libPaths()))
# ============================================================
# Statistical Analysis Script
# Hypotheses: H1 (negative comment rate), H2 (discourse shift)
# Output:  analysis_output/statistical/statistical_report.txt  — full text report
#          analysis_output/statistical/                 — all plots
# ============================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(tibble)   # for deframe()

INPUT_PATH <- "D:/00瑞典/CSS/Master’s Thesis/数据/10.12data/comments_sentiment.csv"   # adjust to local path
OUTPUT_DIR <- Sys.getenv("WEIBO_STAT_OUTPUT", unset = "analysis_output")
FIG_DIR    <- file.path(OUTPUT_DIR, "statistical")
dir.create(FIG_DIR,    showWarnings = FALSE, recursive = TRUE)

REPORT_PATH <- file.path(FIG_DIR, "statistical_report.txt")
# Open report file (overwrite on each run)
rcon <- file(REPORT_PATH, open = "w", encoding = "UTF-8")

# Helper: write to both console and report file
rpt <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  writeLines(msg, con = rcon)
}
rpt_df <- function(df) {
  txt <- capture.output(print(df))
  cat(paste(txt, collapse = "\n"), "\n\n")
  writeLines(txt, con = rcon)
  writeLines("", con = rcon)
}

# ── Data loading ────────────────────────────────────────────
df <- read.csv(INPUT_PATH, stringsAsFactors = FALSE,
               colClasses = c(post_mid = "character"))

# Write report header with timestamp
rpt("Statistical Analysis Report")
rpt("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M"))
rpt(strrep("=", 60))
rpt(sprintf("Total comments loaded: %d", nrow(df)))

# Retain all five topics.
df_main <- df 
rpt(sprintf("Comments for cross-group analysis (all five topics): %d",
            nrow(df_main)))
rpt("")

# The five topics available for cross-group comparison
topics_all <- sort(unique(df_main$keyword_topic))
TOPIC_EN <- c(
  "交通"   = "Traffic",
  "消费纠纷" = "Consumer disputes",
  "邻里纠纷" = "Neighbourhood conflicts",
  "职业学历" = "Professional/academic",
  "治安事件" = "Crime and legal disputes"
)

# Recode topic and group labels to English; set factor levels for regression
# B Neutral is the reference category in all logistic models
df_main <- df_main %>%
  mutate(
    keyword_topic_en = recode(keyword_topic, !!!TOPIC_EN),
    group_f  = factor(keyword_group,
                      levels = c("B_neutral", "A_gendered", "C_gendered"),
                      labels = c("B Neutral", "A Female-gendered",
                                 "C Male-gendered")),
    topic_f  = factor(keyword_topic_en),
    is_female       = as.integer(commenter_gender == "f"),
    # Publisher account type: Blue-V = institutional, Yellow-V = certified individual
    # Recoded as dummy with Blue-V as reference (most common in Group B)
    auth_yellowV  = as.integer(user_authentication == "黄V"),
    auth_ordinary = as.integer(user_authentication == "普通用户")
  )

df <- df %>%
  mutate(
    keyword_topic_en = recode(keyword_topic,
      "交通" = "Traffic", "消费纠纷" = "Consumer disputes",
      "邻里纠纷" = "Neighbourhood conflicts",
      "职业学历" = "Professional/academic",
      "治安事件" = "Crime and legal disputes"
    )
  )

# ── Helper functions ─────────────────────────────────────────
# chi_v(): wraps chisq.test and computes Cramer's V effect size
# droplevels() ensures empty factor levels are removed after subsetting,
# preventing a 3x2 table when only 2 groups are being compared
chi_v <- function(tab) {
  tab <- tab[rowSums(tab) > 0, , drop = FALSE]   # drop empty rows
  chi <- suppressWarnings(chisq.test(tab))
  v   <- sqrt(chi$statistic / (sum(tab) * (min(dim(tab)) - 1)))
  list(chi = chi, v = as.numeric(v))
}
# fp(): format p-value; collapses small values to "< 0.001"
fp <- function(p) ifelse(p < 0.001, "< 0.001", sprintf("= %.4f", p))

# Post-clustered sandwich covariance: HC1 and finite-cluster correction.
# Scores are summed within post, allowing arbitrary within-post dependence.
cluster_vcov <- function(model) {
  X <- model.matrix(model)
  stopifnot(model$converged, nrow(X) == nrow(df_main),
            all(is.finite(coef(model))), !anyNA(df_main$post_mid))
  cluster <- sub("^'+", "", trimws(df_main$post_mid))
  stopifnot(all(nzchar(cluster)))
  n <- nrow(X); k <- ncol(X); g <- length(unique(cluster))
  stopifnot(g > 1, n > k, model$rank == k)
  bread <- solve(crossprod(X, X * as.numeric(model$weights)))
  scores <- X * as.numeric(model$y - fitted(model))
  grouped <- rowsum(scores, cluster, reorder = FALSE)
  V <- (g / (g - 1)) * ((n - 1) / (n - k)) *
       bread %*% crossprod(grouped) %*% bread
  dimnames(V) <- list(colnames(X), colnames(X))
  if (requireNamespace("sandwich", quietly = TRUE)) {
    reference <- sandwich::vcovCL(model, cluster = cluster, type = "HC1", cadjust = TRUE)
    stopifnot(isTRUE(all.equal(V, reference, tolerance = 1e-7)))
  }
  V
}
cluster_coefficients <- function(model) {
  beta <- coef(model); se <- sqrt(diag(cluster_vcov(model)))
  result <- cbind(beta, se, beta / se, 2 * pnorm(abs(beta / se), lower.tail = FALSE))
  colnames(result) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
  result
}
rpt("Regression inference: post-clustered HC1 sandwich SE with finite-cluster correction.")
rpt("Wald tests use asymptotic normal/chi-square reference distributions.")
rpt("Descriptive chi-square tests below retain their conventional independence assumption.")
rpt(sprintf("Distinct post clusters: %d", length(unique(df_main$post_mid))))
rpt("")

# Direct A vs C contrast in the SAME adjusted main-effects model.
# Var(beta_A - beta_C) includes their covariance: do not compare p-values
# against B or subtract standard errors. This is a two-sided Wald test.
adjusted_ac <- function(model, outcome) {
  a <- "group_fA Female-gendered"
  c <- "group_fC Male-gendered"
  beta <- coef(model)
  V <- cluster_vcov(model)
  stopifnot(model$converged, all(c(a, c) %in% names(beta)))
  estimate <- unname(beta[a] - beta[c])
  variance <- unname(V[a, a] + V[c, c] - 2 * V[a, c])
  stopifnot(is.finite(estimate), is.finite(variance), variance > 0)
  se <- sqrt(variance)
  z <- estimate / se
  data.frame(
    Outcome = outcome, Comparison = "A Female-gendered vs C Male-gendered",
    N = nobs(model), Estimate = estimate, Std.Error = se, z = z,
    p_raw = 2 * pnorm(abs(z), lower.tail = FALSE),
    OR = exp(estimate), CI_lo = exp(estimate - qnorm(0.975) * se),
    CI_hi = exp(estimate + qnorm(0.975) * se)
  )
}

report_ac <- function(result) {
  rpt("--- Adjusted direct A vs C comparison ---")
  rpt("    Same main-effects model; controls: topic, commenter gender, Yellow-V.")
  rpt("    Post-clustered HC1 SE; finite-cluster correction; two-sided raw p.")
  rpt("    OR is an odds ratio, not a probability ratio; CI is pointwise 95%.")
  rpt(sprintf("    N = %d; log(OR) = %.6f; SE = %.6f; z = %.4f",
              result$N, result$Estimate, result$Std.Error, result$z))
  rpt(sprintf("    A vs C OR = %.4f; 95%% CI [%.4f, %.4f]; p = %.8g",
              result$OR, result$CI_lo, result$CI_hi, result$p_raw))
  rpt("")
}

pairs_raw <- list(
  c("A_gendered", "B_neutral"),
  c("A_gendered", "C_gendered"),
  c("C_gendered", "B_neutral")
)
pairs_lbl <- c("A Female-gendered vs B Neutral",
               "A Female-gendered vs C Male-gendered",
               "C Male-gendered vs B Neutral")


# ════════════════════════════════════════════════════════════
# H1: Do gendered headlines produce higher negative comment rates?
# ════════════════════════════════════════════════════════════
rpt(strrep("=", 60))
rpt("H1: Negative Expression Rate by Headline Group")
rpt(strrep("=", 60))

# ── Descriptive statistics ───────────────────────────────────
rpt("")
rpt("--- Descriptive Statistics ---")
desc_h1 <- df_main %>%
  group_by(group_f) %>%
  summarise(
    N            = n(),
    N_negative   = sum(is_negative),
    Neg_rate     = round(mean(is_negative), 4),
    .groups      = "drop"
  )
rpt_df(desc_h1)

# ── Overall chi-squared test (3 groups) ─────────────────────
rpt("--- Overall Chi-squared Test (3 Groups) ---")
h1_tab <- table(df_main$group_f, df_main$is_negative)
h1_res <- chi_v(h1_tab)
rpt(sprintf("chi-squared = %.3f,  df = %d,  p %s",
            h1_res$chi$statistic,
            h1_res$chi$parameter,
            fp(h1_res$chi$p.value)))
rpt(sprintf("Cramer's V  = %.4f", h1_res$v))
rpt("")

# ── Pairwise comparisons (Bonferroni correction) ─────────────
# k=3 pairs; multiply each raw p by 3; cap at 1
rpt("--- Pairwise Chi-squared Tests (Bonferroni-corrected, k=3) ---")
pw_h1 <- mapply(function(p, lbl) {
  sub <- df_main %>% filter(keyword_group %in% p)
  r   <- chi_v(table(sub$group_f, sub$is_negative))
  data.frame(
    Comparison   = lbl,
    chi2         = round(r$chi$statistic, 3),
    p_raw        = round(r$chi$p.value, 6),
    p_bonferroni = round(min(r$chi$p.value * 3, 1), 6),
    cramers_v    = round(r$v, 4)
  )
}, pairs_raw, pairs_lbl, SIMPLIFY = FALSE)
rpt_df(do.call(rbind, pw_h1))

# ── Topic-stratified A vs B ──────────────────────────────────
# Check whether the H1 effect is consistent across all five topics
# (rules out the possibility that one topic is driving the overall result)
rpt("--- Topic-stratified: A Female-gendered vs B Neutral ---")
topic_h1 <- lapply(topics_all, function(tp) {
  sA <- df_main %>% filter(keyword_topic == tp, keyword_group == "A_gendered")
  sB <- df_main %>% filter(keyword_topic == tp, keyword_group == "B_neutral")
  if (nrow(sA) < 10 | nrow(sB) < 10) return(NULL)
  tab <- table(
    c(rep("A", nrow(sA)), rep("B", nrow(sB))),
    c(sA$is_negative,      sB$is_negative)
  )
  chi <- suppressWarnings(chisq.test(tab))
  data.frame(
    Topic  = TOPIC_EN[tp],
    A_rate = round(mean(sA$is_negative), 3),
    B_rate = round(mean(sB$is_negative), 3),
    Diff   = round(mean(sA$is_negative) - mean(sB$is_negative), 3),
    chi2   = round(chi$statistic, 3),
    p      = round(chi$p.value, 4),
    sig    = ifelse(chi$p.value < 0.05, "*", "")
  )
})
rpt_df(do.call(rbind, Filter(Negate(is.null), topic_h1)))

# ── Logistic regression ──────────────────────────────────────
# Controls for topic fixed effects and commenter gender (self-reported, approximate)
# OR > 1 means higher odds of negative comments relative to B Neutral
rpt("--- Logistic Regression: is_negative ~ group + topic + commenter_gender ---")
rpt("    Reference group: B Neutral.  Commenter gender is self-reported")
rpt("    and treated as an approximate control.")
rpt("")
# Add publisher account type dummies to control for potential confounding
# Reference category: Blue-V (institutional accounts)
# auth_ordinary excluded: only 32 posts from unverified accounts,
# insufficient for stable coefficient estimation in both models
m1  <- glm(is_negative ~ group_f + topic_f + is_female + auth_yellowV,
           data = df_main, family = binomial())
cf1 <- cluster_coefficients(m1)
kdf1 <- as.data.frame(cf1)
# Convert log-odds to odds ratios; compute 95% CI via Wald approximation
kdf1$OR  <- round(exp(kdf1$Estimate), 4)
kdf1$CI_lo <- round(exp(kdf1$Estimate - 1.96 * kdf1$`Std. Error`), 4)
kdf1$CI_hi <- round(exp(kdf1$Estimate + 1.96 * kdf1$`Std. Error`), 4)
colnames(kdf1)[1:4] <- c("Estimate", "Std.Error", "z", "p")
kdf1 <- round(kdf1, 4)
rpt_df(kdf1)
rpt(sprintf("Model AIC = %.1f", AIC(m1)))
rpt("")

# H1: adjusted negative-comment odds in A relative to C.
ac_h1 <- adjusted_ac(m1, "is_negative")
report_ac(ac_h1)
write.csv(ac_h1, file.path(FIG_DIR, "adjusted_AC_H1.csv"), row.names = FALSE)

# ── Interaction term: group × topic ─────────────────────────
# Tests whether the framing effect size differs across event topics
# Uses a post-clustered joint Wald test for interaction terms
rpt("--- Interaction Model: is_negative ~ group * topic + commenter_gender ---")
rpt("    Tests whether the framing effect differs across event topics.")
rpt("")
m1_int <- glm(is_negative ~ group_f * topic_f + is_female + auth_yellowV,
              data = df_main, family = binomial())
rpt(sprintf("Main effects model AIC  = %.1f", AIC(m1)))
rpt(sprintf("Interaction model AIC   = %.1f", AIC(m1_int)))

# Cluster-robust joint Wald test of all group-by-topic interactions.
interaction_names <- grep(":", names(coef(m1_int)), value = TRUE)
bi <- coef(m1_int)[interaction_names]
Vi <- cluster_vcov(m1_int)[interaction_names, interaction_names, drop = FALSE]
wald <- as.numeric(crossprod(bi, solve(Vi, bi)))
wald_df <- length(bi)
wald_p <- pchisq(wald, df = wald_df, lower.tail = FALSE)
rpt(sprintf("Post-clustered joint Wald chi-sq = %.3f, df = %d, p %s",
            wald, wald_df, fp(wald_p)))
rpt("")
rpt("Interaction coefficients (group_f × topic_f):")
int_rows <- grep("group_f.*topic_f|topic_f.*group_f",
                 rownames(cluster_coefficients(m1_int)))
if (length(int_rows) > 0) {
  int_cf <- as.data.frame(cluster_coefficients(m1_int)[int_rows, , drop = FALSE])
  int_cf$OR <- round(exp(int_cf$Estimate), 4)
  colnames(int_cf)[1:4] <- c("Estimate", "Std.Error", "z", "p")
  rpt_df(round(int_cf, 4))
} else {
  rpt("  (No interaction terms found — check factor contrasts)")
  rpt("")
}


# ════════════════════════════════════════════════════════════
# H2: Do gendered headlines shift discourse toward gender language?
# ════════════════════════════════════════════════════════════
rpt(strrep("=", 60))
rpt("H2: Gender Language; Supplementary Factual Language Comparisons")
rpt(strrep("=", 60))
rpt("")

# ── Overall descriptive ──────────────────────────────────────
rpt("--- Overall Descriptive (all 5 topics) ---")
desc_h2 <- df %>%
  group_by(keyword_group) %>%
  summarise(
    N              = n(),
    Factual_rate   = round(mean(has_factual_lang), 4),
    Gender_rate    = round(mean(has_gender_lang),  4),
    Ratio_G_to_F   = round(mean(has_gender_lang) /
                             max(mean(has_factual_lang), 0.0001), 3),
    .groups = "drop"
  )
rpt_df(desc_h2)

# ── Overall chi-squared (3 groups) ──────────────────────────
rpt("--- Overall Chi-squared Tests (3 Groups) ---")
for (var in c("has_factual_lang", "has_gender_lang")) {
  r <- chi_v(table(df$keyword_group, df[[var]]))
  rpt(sprintf("  %-20s  chi2 = %.3f  p %s  V = %.4f",
              var, r$chi$statistic, fp(r$chi$p.value), r$v))
}
rpt("")

# ── Pairwise comparisons ─────────────────────────────────────
rpt("--- Pairwise Tests (Bonferroni-corrected) ---")
for (var in c("has_factual_lang", "has_gender_lang")) {
  rpt(sprintf("  Variable: %s", var))
  pw_h2 <- mapply(function(p, lbl) {
    sub <- df %>% filter(keyword_group %in% p)
    r   <- chi_v(table(sub$keyword_group, sub[[var]]))
    data.frame(
      Comparison   = lbl,
      chi2         = round(r$chi$statistic, 3),
      p_bonferroni = round(min(r$chi$p.value * 3, 1), 6),
      cramers_v    = round(r$v, 4)
    )
  }, pairs_raw, pairs_lbl, SIMPLIFY = FALSE)
  rpt_df(do.call(rbind, pw_h2))
}

# ── Topic-stratified A vs B ──────────────────────────────────
rpt("--- Topic-stratified: A vs B Chi-squared (factual & gender language) ---")
topic_h2 <- lapply(topics_all, function(tp) {
  sA <- df_main %>% filter(keyword_topic == tp, keyword_group == "A_gendered")
  sB <- df_main %>% filter(keyword_topic == tp, keyword_group == "B_neutral")
  if (nrow(sA) < 10 | nrow(sB) < 10) return(NULL)
  mk <- function(v1, v2) {
    tab <- table(c(rep("A", length(v1)), rep("B", length(v2))), c(v1, v2))
    suppressWarnings(chisq.test(tab))$p.value
  }
  data.frame(
    Topic      = TOPIC_EN[tp],
    A_factual  = round(mean(sA$has_factual_lang), 4),
    B_factual  = round(mean(sB$has_factual_lang), 4),
    p_factual  = round(mk(sA$has_factual_lang, sB$has_factual_lang), 4),
    sig_f      = ifelse(mk(sA$has_factual_lang, sB$has_factual_lang) < 0.05, "*", ""),
    A_gender   = round(mean(sA$has_gender_lang), 4),
    B_gender   = round(mean(sB$has_gender_lang), 4),
    p_gender   = round(mk(sA$has_gender_lang, sB$has_gender_lang), 4),
    sig_g      = ifelse(mk(sA$has_gender_lang, sB$has_gender_lang) < 0.05, "*", "")
  )
})
rpt_df(do.call(rbind, Filter(Negate(is.null), topic_h2)))

# ── Logistic regression for H2 ───────────────────────────────
# Same model structure as H1; outcome is has_gender_lang (gender discourse marker)
rpt("--- Logistic Regression: has_gender_lang ~ group + topic + commenter_gender ---")
rpt("    Reference group: B Neutral.")
rpt("")
# Same covariate structure as H1 for symmetry
m2 <- glm(has_gender_lang ~ group_f + topic_f + is_female + auth_yellowV,
          data = df_main, family = binomial())
cf2 <- cluster_coefficients(m2)
kdf2 <- as.data.frame(cf2)
kdf2$OR    <- round(exp(kdf2$Estimate), 4)
kdf2$CI_lo <- round(exp(kdf2$Estimate - 1.96 * kdf2$`Std. Error`), 4)
kdf2$CI_hi <- round(exp(kdf2$Estimate + 1.96 * kdf2$`Std. Error`), 4)
colnames(kdf2)[1:4] <- c("Estimate", "Std.Error", "z", "p")
kdf2 <- round(kdf2, 4)
rpt_df(kdf2)
rpt(sprintf("Model AIC = %.1f", AIC(m2)))
rpt("")

ac_h2 <- adjusted_ac(m2, "has_gender_lang")
report_ac(ac_h2)
write.csv(ac_h2, file.path(FIG_DIR, "adjusted_AC_H2.csv"), row.names = FALSE)

# Supplementary factual-language regression: same controls and cluster SE.
rpt("--- Supplementary Logistic Regression: Factual Language ---")
rpt("    Same controls and post-clustered inference as H1 and gender language.")
m3 <- glm(has_factual_lang ~ group_f + topic_f + is_female + auth_yellowV,
          data = df_main, family = binomial())
cf3 <- cluster_coefficients(m3)
factual_coefficients <- data.frame(Term = rownames(cf3),
  Estimate = cf3[, "Estimate"], Std.Error = cf3[, "Std. Error"],
  OR = exp(cf3[, "Estimate"]),
  CI_lo = exp(cf3[, "Estimate"] - qnorm(.975) * cf3[, "Std. Error"]),
  CI_hi = exp(cf3[, "Estimate"] + qnorm(.975) * cf3[, "Std. Error"]),
  p = cf3[, "Pr(>|z|)"], row.names = NULL)
rpt_df(factual_coefficients)
write.csv(factual_coefficients, file.path(FIG_DIR, "factual_regression_clustered.csv"),
          row.names = FALSE)
ac_factual <- adjusted_ac(m3, "has_factual_lang")
report_ac(ac_factual)
write.csv(ac_factual, file.path(FIG_DIR, "adjusted_AC_factual.csv"), row.names = FALSE)

# ── Summary table: H1 and H2 side by side ───────────────────
# Combines key group ORs from both models into one compact table
rpt("--- Summary: Key Logistic Regression Results ---")
rpt("    Outcome 1: is_negative  |  Outcome 2: has_gender_lang")
rpt("    (odds ratios; reference = B Neutral)")
rpt("")
grp_rows <- grep("group_f", rownames(cf1))
summ <- data.frame(
  Term            = rownames(cf1)[grp_rows],
  OR_isNeg        = round(exp(cf1[grp_rows, "Estimate"]), 3),
  p_isNeg         = round(cf1[grp_rows, "Pr(>|z|)"], 4),
  OR_genderLang   = round(exp(cf2[grp_rows, "Estimate"]), 3),
  p_genderLang    = round(cf2[grp_rows, "Pr(>|z|)"], 4)
)
rpt_df(summ)


group_results <- function(model, outcome) {
  terms <- c("group_fA Female-gendered", "group_fC Male-gendered")
  beta <- coef(model)[terms]
  se <- sqrt(diag(cluster_vcov(model)))[terms]
  old_se <- sqrt(diag(vcov(model)))[terms]
  data.frame(Outcome = outcome, Term = terms, OR = exp(beta),
    CI_lo = exp(beta - qnorm(.975) * se), CI_hi = exp(beta + qnorm(.975) * se),
    p = 2 * pnorm(abs(beta / se), lower.tail = FALSE),
    conventional_SE = old_se, cluster_SE = se, row.names = NULL)
}
exact_results <- bind_rows(group_results(m1, "Negative expression (H1)"),
                          group_results(m2, "Gender language (H2)"),
                          group_results(m3, "Factual language"))
write.csv(exact_results, file.path(FIG_DIR, "clustered_group_estimates.csv"), row.names = FALSE)

# ════════════════════════════════════════════════════════════
# Final summary: key numbers printed to report for quick reference
# ════════════════════════════════════════════════════════════
rpt(strrep("=", 60))
rpt("Results Summary")
rpt(strrep("=", 60))
rpt("")

neg_by_grp <- df_main %>%
  group_by(keyword_group) %>%
  summarise(r = mean(is_negative), .groups = "drop") %>%
  deframe()

rpt("[H1] Negative comment rate")
rpt(sprintf("  A Female-gendered = %.1f%%", neg_by_grp["A_gendered"] * 100))
rpt(sprintf("  B Neutral         = %.1f%%", neg_by_grp["B_neutral"]  * 100))
rpt(sprintf("  C Male-gendered   = %.1f%%", neg_by_grp["C_gendered"] * 100))
rpt(sprintf("  Overall chi2 = %.2f,  p %s,  V = %.4f",
            h1_res$chi$statistic, fp(h1_res$chi$p.value), h1_res$v))
or_a <- round(exp(coef(m1)["group_fA Female-gendered"]), 3)
or_c <- round(exp(coef(m1)["group_fC Male-gendered"]),   3)
rpt(sprintf("  Logistic OR (A vs B) = %.3f,  OR (C vs B) = %.3f", or_a, or_c))
rpt("")

rpt("[H2] Gender language rate")
for (grp in c("A_gendered", "B_neutral", "C_gendered")) {
  lbl <- c("A_gendered" = "A Female-gendered",
           "B_neutral"  = "B Neutral",
           "C_gendered" = "C Male-gendered")[grp]
  f_r <- round(mean(df_main$has_factual_lang[df_main$keyword_group == grp]) * 100, 2)
  g_r <- round(mean(df_main$has_gender_lang[df_main$keyword_group == grp])  * 100, 2)
  rat <- round(mean(df_main$has_gender_lang[df_main$keyword_group == grp]) /
               max(mean(df_main$has_factual_lang[df_main$keyword_group == grp]), 0.0001), 2)
  rpt(sprintf("  %s:  factual = %.2f%%,  gender = %.2f%%,  G/F ratio = %.2f",
              lbl, f_r, g_r, rat))
}
rpt("")
rpt("Analysis complete.")
rpt(sprintf("Report saved: %s", REPORT_PATH))

close(rcon)
cat(sprintf("\n✓ Report written to: %s\n", REPORT_PATH))


# ════════════════════════════════════════════════════════════
# Figures (English labels throughout)
# ════════════════════════════════════════════════════════════
theme_set(theme_minimal(base_size = 12))
colors3 <- c("A Female-gendered" = "#D64C6F",
             "B Neutral"         = "#5B8DB8",
             "C Male-gendered"   = "#4CAF7D")

# ── Figures ─────────────────────────────────────────────────
# Fig 1: Bar chart of negative comment rates; A B C on x-axis (left to right)
p1 <- df_main %>%
  group_by(group_f) %>%
  summarise(neg_rate = mean(is_negative), .groups = "drop") %>%
  ggplot(aes(x = group_f, y = neg_rate, fill = group_f)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = percent(neg_rate, accuracy = 0.1)),
            vjust = -0.4, size = 4) +
  scale_y_continuous(labels = percent_format(), limits = c(0, 0.75)) +
  scale_fill_manual(values = colors3) +
  scale_x_discrete(limits = c("A Female-gendered", "B Neutral", "C Male-gendered")) +
  labs(title = "Negative Expression Rate by Headline Group",
       x = NULL, y = "Negative expression rate (%)") +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 15, hjust = 0.5))
ggsave(file.path(FIG_DIR, "fig1_negative_rate.png"), p1,
       width = 7.6, height = 5.2, dpi = 300, bg = "white")

# Fig 2: Heatmap showing whether the H1 effect is consistent across topics
p2 <- df_main %>%
  group_by(keyword_topic_en, group_f) %>%
  summarise(neg_rate = mean(is_negative), .groups = "drop") %>%
  ggplot(aes(x = group_f,
             y = reorder(keyword_topic_en, neg_rate),
             fill = neg_rate)) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = percent(neg_rate, accuracy = 0.1)),
            color = "white", fontface = "bold", size = 4) +
  scale_fill_gradient(low = "#FDD5C0", high = "#C0392B",
                      labels = percent_format()) +
  scale_x_discrete(limits = c("A Female-gendered", "B Neutral", "C Male-gendered")) +
  labs(title = "Negative Expression Rate by Topic and Headline Group",
       x = NULL, y = NULL, fill = "Neg. rate") +
  theme(plot.title = element_text(face = "bold"))
ggsave(file.path(FIG_DIR, "fig2_heatmap.png"), p2,
       width = 9, height = 5, dpi = 150)

# Fig 3a: Grouped bar chart comparing factual vs gender language rates (H2)
p3a <- df %>%
  group_by(keyword_group) %>%
  summarise(
    Factual = mean(has_factual_lang),
    Gender  = mean(has_gender_lang),
    .groups = "drop"
  ) %>%
  mutate(keyword_group = recode(keyword_group,
    "A_gendered" = "A Female-gendered",
    "B_neutral"  = "B Neutral",
    "C_gendered" = "C Male-gendered"
  )) %>%
  pivot_longer(c(Factual, Gender),
               names_to = "Discourse type", values_to = "Rate") %>%
  ggplot(aes(x = keyword_group, y = Rate, fill = `Discourse type`)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = percent(Rate, accuracy = 0.01)),
            position = position_dodge(0.6), vjust = -0.4, size = 3.5) +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("Factual" = "#5B8DB8", "Gender" = "#D64C6F")) +
  scale_x_discrete(limits = c("A Female-gendered", "B Neutral", "C Male-gendered")) +
  labs(title = "Factual vs Gender Language by Framing Group",
       x = NULL, y = "Rate", fill = NULL) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")
ggsave(file.path(FIG_DIR, "fig3a_discourse_overall.png"), p3a,
       width = 8, height = 5, dpi = 150)

# Fig 3b: Same as 3a but faceted by event topic
p3b <- df_main %>%
  group_by(keyword_topic_en, group_f) %>%
  summarise(
    Factual = mean(has_factual_lang),
    Gender  = mean(has_gender_lang),
    .groups = "drop"
  ) %>%
  pivot_longer(c(Factual, Gender),
               names_to = "Discourse type", values_to = "Rate") %>%
  ggplot(aes(x = group_f, y = Rate, fill = `Discourse type`)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.65) +
  facet_wrap(~ keyword_topic_en, nrow = 2, scales = "free_y") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("Factual" = "#5B8DB8", "Gender" = "#D64C6F")) +
  scale_x_discrete(limits = c("A Female-gendered", "B Neutral", "C Male-gendered")) +
  labs(title = "Factual vs Gender Language by Topic",
       x = NULL, y = "Rate", fill = NULL) +
  theme(plot.title  = element_text(face = "bold"),
        legend.position = "bottom",
        axis.text.x = element_text(angle = 30, hjust = 1))
ggsave(file.path(FIG_DIR, "fig3b_discourse_by_topic.png"), p3b,
       width = 10, height = 7, dpi = 150)

# Fig 4: two independent panels; colour + shape, labels, clustered intervals.
forest_data <- exact_results %>% mutate(
  Group = ifelse(grepl("Female", Term), "Group A (Female-gendered)", "Group C (Male-gendered)"),
  y = ifelse(grepl("Female", Term), 2, 1),
  Label = sprintf("%.2f [%.2f, %.2f]", OR, CI_lo, CI_hi))
note <- paste(
  "Note. Group B (Gender-neutral) is the reference; the dashed line marks AOR = 1.",
  "Points show adjusted odds ratios; bars show 95% confidence intervals.",
  "Controls: event topic, commenter gender, and publisher account type.",
  "Standard errors are clustered by post.", sep = "\n")
make_forest <- function(outcome, title) {
  d <- forest_data %>% filter(Outcome == outcome)
  span <- diff(range(c(1, d$CI_lo, d$CI_hi)))
  limits <- range(c(1, d$CI_lo, d$CI_hi)) + c(-.30, .35) * span
  limits[1] <- max(.01, limits[1])
  ggplot(d, aes(x = OR, y = y, colour = Group, shape = Group)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "gray40", linewidth = .6) +
    geom_segment(aes(x = CI_lo, xend = CI_hi, yend = y), linewidth = .8) +
    geom_segment(aes(x = CI_lo, xend = CI_lo, y = y - .06, yend = y + .06), linewidth = .7) +
    geom_segment(aes(x = CI_hi, xend = CI_hi, y = y - .06, yend = y + .06), linewidth = .7) +
    geom_point(size = 3.5) +
    geom_text(aes(y = y + .23, label = Label), colour = "black", size = 4, show.legend = FALSE) +
    scale_y_continuous(breaks = c(2, 1),
      labels = c("Group A\n(Female-gendered)", "Group C\n(Male-gendered)"), limits = c(.65, 2.55)) +
    scale_x_continuous(limits = limits, breaks = scales::breaks_pretty(n = 4)) +
    scale_colour_manual(values = c("Group A (Female-gendered)" = "#D64C6F",
                                  "Group C (Male-gendered)" = "#26836B")) +
    scale_shape_manual(values = c("Group A (Female-gendered)" = 16,
                                 "Group C (Male-gendered)" = 17)) +
    labs(title = title, x = "Adjusted odds ratio", y = NULL, colour = NULL, shape = NULL) +
    theme_classic(base_size = 12) +
    theme(plot.title = element_text(face = "bold", size = 13),
          axis.text.y = element_text(size = 11, colour = "black", hjust = .5, margin = margin(r = 12)),
          legend.position = "bottom", legend.text = element_text(size = 9),
          legend.box = "horizontal", plot.margin = margin(12, 16, 10, 12)) +
    guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
}
p4a <- make_forest("Negative expression (H1)", "A. Negative expression (H1)")
p4b <- make_forest("Gender language (H2)", "B. Gender language (H2)")
p4c <- make_forest("Factual language", "C. Factual language")
for (panel in c("A", "B", "C")) {
  p <- switch(panel, A = p4a, B = p4b, C = p4c)
  standalone_title <- switch(panel,
    A = "Adjusted Odds Ratios for Negative Expression",
    B = "Adjusted Odds Ratios for Gender Language",
    C = "Adjusted Odds Ratios for Factual Language")
  p <- p + labs(title = standalone_title)

  standalone_note <- paste0("Note. Group B (Gender-neutral) is the reference (AOR = 1).\n",
    "Points = AOR; bars = 95% CI; standard errors clustered by post.\n",
    "Controls: event topic, commenter gender, and publisher account type.")
  ggsave(file.path(FIG_DIR, paste0("fig4", panel, "_odds_ratios.png")),
    p + labs(caption = standalone_note) +
      theme(plot.caption = element_text(hjust = 0, size = 10, lineheight = 1.25)),
    width = 7.4, height = 4.8, dpi = 300, bg = "white")
}
# Combined files and separate H2-only figure use vertical panels.
draw_stack <- function(plots, title) {
  grid::grid.newpage()
  grid::grid.text(title, x = .5, y = .977,
                  gp = grid::gpar(fontsize = 13, fontface = "bold"))
  n <- length(plots)
  top <- .943; bottom <- .17; panel_height <- (top - bottom) / n
  for (i in seq_along(plots)) {
    centre <- top - (i - .5) * panel_height
    print(plots[[i]], vp = grid::viewport(x = .5, y = centre,
      width = .98, height = panel_height - .012))
    if (i < n) grid::grid.lines(x = c(.04, .97),
      y = rep(top - i * panel_height, 2), gp = grid::gpar(col = "gray85"))
  }
  grid::grid.text(note,
    x = .045, y = .105, just = c("left", "center"),
    gp = grid::gpar(fontsize = 11, lineheight = 1.35))
}
save_stack <- function(plots, title, filename, height) {
  png(file.path(FIG_DIR, paste0(filename, ".png")), width = 9.4, height = height,
      units = "in", res = 240, bg = "white")
  draw_stack(plots, title); dev.off()
  pdf(file.path(FIG_DIR, paste0(filename, ".pdf")), width = 9.4, height = height)
  draw_stack(plots, title); dev.off()
}
save_stack(list(p4a, p4b, p4c), "Adjusted odds ratios by headline framing group",
           "fig4_odds_ratios", 14.2)
save_stack(list(p4b + labs(title = "A. Gender language (H2)"),
                p4c + labs(title = "B. Factual language")),
           "Adjusted odds ratios for gender and factual language",
           "fig4_H2_gender_factual", 10.6)
cat("Updated figures and post-clustered regression outputs saved in: ", FIG_DIR, "\n")
