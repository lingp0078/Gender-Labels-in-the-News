# ============================================================
# 描述性统计脚本
# 数据：微博帖子（Weibo_post_data_cleaning.csv）
#       + 评论（comments_sentiment.csv）
# 输出：三张HTML表，保留原gtsummary/gt样式；只读取最终标签，不重新分类
# ============================================================

library(tidyverse)
library(gtsummary)
library(gt)

# Input/output folders: use the script folder by default, or set
# WEIBO_DATA_DIR and WEIBO_OUTPUT_DIR to override without editing the script.
script_file <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
if (is.null(script_file)) {
  script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(script_arg)) script_file <- sub("^--file=", "", script_arg[1])
}
default_dir <- if (is.null(script_file)) getwd() else
  dirname(normalizePath(script_file, winslash = "/", mustWork = TRUE))
DATA_DIR <- Sys.getenv("WEIBO_DATA_DIR", unset = default_dir)
OUTPUT_DIR <- Sys.getenv("WEIBO_OUTPUT_DIR",
                        unset = file.path(DATA_DIR, "analysis_output", "descriptive"))
POSTS_PATH <- file.path(DATA_DIR, "Weibo_post_data_cleaning.csv")
COMMENTS_PATH <- file.path(DATA_DIR, "comments_sentiment.csv")

GROUP_CODES <- c("A_gendered", "B_neutral", "C_gendered")
GROUP_LABELS <- c("A Female-gendered", "B Neutral", "C Male-gendered")
TOPIC_LABELS <- c("交通" = "Traffic", "消费纠纷" = "Consumer disputes",
  "邻里纠纷" = "Neighbourhood conflicts", "治安事件" = "Crime and legal disputes",
  "职业学历" = "Professional/academic")
SOURCE_NOTE <- "*Source: Weibo posts from 2025.1.1-2026.3.28*"

require_columns <- function(data, required, name) {
  absent <- setdiff(required, names(data))
  if (length(absent)) stop(name, " missing columns: ", paste(absent, collapse = ", "))
}
read_input <- function(path) {
  if (!file.exists(path)) stop("Input file not found: ", path)
  data <- read_csv(path, col_types = cols(.default = col_character()),
                   show_col_types = FALSE)
  if (!nrow(data)) stop("Empty input file: ", path)
  if (nrow(problems(data))) stop("Malformed CSV records in: ", path)
  data
}
binary_field <- function(x, name) {
  value <- toupper(trimws(x))
  if (anyNA(value) || !all(value %in% c("0", "1", "FALSE", "TRUE"))) {
    stop(name, " must contain complete 0/1 or FALSE/TRUE values")
  }
  value %in% c("1", "TRUE")
}
prepare_groups <- function(data, name) {
  if (anyNA(data$keyword_group) || !all(data$keyword_group %in% GROUP_CODES)) {
    stop(name, ": missing or unrecognised framing group")
  }
  if (!all(GROUP_CODES %in% data$keyword_group)) stop(name, ": an A/B/C group is absent")
  if (anyNA(data$keyword_topic) || !all(data$keyword_topic %in% names(TOPIC_LABELS))) {
    stop(name, ": missing or unrecognised topic")
  }
  data %>% mutate(
    keyword_group = factor(keyword_group, levels = GROUP_CODES, labels = GROUP_LABELS),
    keyword_topic = factor(keyword_topic, levels = names(TOPIC_LABELS)))
}

posts <- read_input(POSTS_PATH)
comments <- read_input(COMMENTS_PATH)
require_columns(posts, c("post_mid_formatted", "keyword_group", "keyword_topic",
                        "user_authentication"), "Posts")
require_columns(comments, c("post_mid", "keyword_group", "keyword_topic", "is_negative",
  "has_gender_lang", "has_factual_lang", "commenter_gender"), "Comments")
post_ids <- sub("^'+", "", trimws(posts$post_mid_formatted))
comment_ids <- sub("^'+", "", trimws(comments$post_mid))
if (anyNA(post_ids) || any(!nzchar(post_ids)) || anyDuplicated(post_ids)) {
  stop("Post IDs must be nonmissing and unique")
}
if (anyNA(comment_ids) || any(!nzchar(comment_ids))) stop("Missing parent post ID")
if (any(!comment_ids %in% post_ids)) warning("Some comments have no parent in the post CSV")
if (anyNA(posts$user_authentication) ||
    !all(posts$user_authentication %in% c("蓝V", "黄V"))) {
  stop("Publisher account type must be complete Blue-V/Yellow-V; inspect input")
}
posts <- prepare_groups(posts, "Posts") %>%
  mutate(user_authentication = factor(user_authentication,
    levels = c("蓝V", "黄V"), labels = c("Blue-V", "Yellow-V")))
comments <- prepare_groups(comments, "Comments") %>% mutate(
  across(c(is_negative, has_gender_lang, has_factual_lang),
         ~ binary_field(.x, cur_column())),
  commenter_gender = factor(commenter_gender, levels = c("m", "f"),
                            labels = c("Male", "Female")))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
cat(sprintf("Posts: %d; comments: %d; comment-linked posts: %d\n",
            nrow(posts), nrow(comments), length(unique(comment_ids))))

# Helper: save gt table as HTML (gt 1.3.0 compatible)
save_gt_html <- function(gt_obj, path) {
  html_str <- as.character(gt::as_raw_html(gt_obj))
  writeLines(enc2utf8(html_str), path, useBytes = TRUE)
  invisible(path)
}




# ════════════════════════════════════════════════════════════
# 表1：帖子数据描述统计（仿 QMD tbl_summary 风格）
# 整体 + 按 keyword_group 分组
# ════════════════════════════════════════════════════════════

# --- 表1a：样本量、Topic分布 (n (x.x%)) 与分组占比 ---
tbl1_top <- posts %>%
    mutate(keyword_topic = recode(keyword_topic, !!!TOPIC_LABELS)) %>%
  select(keyword_group, keyword_topic) %>%
  mutate(
    N = 1, 
    Pct = 100 / nrow(posts)
  ) %>%
  # 严格控制行顺序：N -> Topics -> Percentage
  select(keyword_group, N, keyword_topic, Pct) %>%
  tbl_summary(
    by = keyword_group,
    type = list(
      N ~ "continuous", 
      Pct ~ "continuous",
      keyword_topic ~ "categorical"
    ),
    statistic = list(
      N ~ "{sum}",
      keyword_topic ~ "{n} ({p}%)", 
      Pct ~ "{sum}%"
    ),
    digits = list(
      N ~ 0, 
      keyword_topic ~ c(0, 1),
      Pct ~ 1
    ),
    label = list(
      keyword_topic ~ "Keyword Topics",
      Pct ~ "Column Percentage"
    )
  ) %>%
  add_overall(last = TRUE) %>%
  modify_header(all_stat_cols() ~ "**{level}**") %>%
  modify_header(label ~ "**Variable**") %>%
  modify_footnote(everything() ~ NA)

# --- 表1b：发布账号类型（n及组内百分比） ---
tbl1_bot <- posts %>%
  select(keyword_group, user_authentication) %>%
  tbl_summary(
    by = keyword_group,
    type = list(user_authentication ~ "categorical"),
    statistic = list(all_categorical() ~ "{n} ({p}%)"),
    digits = list(all_categorical() ~ c(0, 1)),
    label = list(user_authentication ~ "Publisher account type")
  ) %>%
  add_overall(col_lab = "**Overall**", last = TRUE) %>%
  modify_header(all_stat_cols() ~ "**{level}**") %>%
  modify_header(label ~ "**Variable**") %>%
  modify_footnote(all_stat_cols() ~ NA)

# ════════════════════════════════════════════════════════════
# 合并与导出
# ════════════════════════════════════════════════════════════

tbl1 <- tbl_stack(list(tbl1_top, tbl1_bot)) %>%
  as_gt() %>%
  tab_header(
    title = md("**Descriptive Statistics of Weibo Posts**")
  ) %>%
  tab_spanner(
    label   = "Framing Group",
    columns = starts_with("stat_")
  ) %>%
  tab_options(
    table.font.size = px(14),
    column_labels.font.weight = "bold",
    data_row.padding = px(3) # 稍微收紧行间距，使表格更紧凑
  ) %>%
  tab_footnote(
    footnote = md("N represents the number of posts."),
    locations = cells_body(rows = label == "N", columns = "label")
  ) %>%
  tab_footnote(
    footnote = md("Blue-V denotes institutional account authentication."),
    locations = cells_body(rows = label == "Blue-V", columns = "label")
  ) %>%
  tab_footnote(
    footnote = md("Yellow-V denotes individual account authentication."),
    locations = cells_body(rows = label == "Yellow-V", columns = "label")
  ) %>%
  tab_source_note(md(SOURCE_NOTE))

# 保存与预览
save_gt_html(tbl1, file.path(OUTPUT_DIR, "table1_descriptive_final.html"))

# ════════════════════════════════════════════════════════════
# 表2：评论数据描述统计（整体 + 按 keyword_group 分组）
# ════════════════════════════════════════════════════════════

# 表2a：样本量占比
tbl2_top <- comments %>%
  select(keyword_group) %>%
  mutate(N = 1, Pct = 100 / nrow(comments)) %>%
  tbl_summary(
    by        = keyword_group,
    type      = list(N ~ "continuous", Pct ~ "continuous"),
    statistic = list(N ~ "{sum}", Pct ~ "{sum}%"),
    digits    = list(N ~ 0, Pct ~ 1),
    label     = list(Pct ~ "Percentage")
  ) %>%
  add_overall(last = TRUE) %>%
  modify_header(all_stat_cols() ~ "{level}") %>%
  modify_header(label ~ "Variable") %>%
  modify_footnote(everything() ~ NA)

# 表2b：评论核心变量（删除评论者已认证和评论者粉丝数）
tbl2_bot <- comments %>%
  select(keyword_group,
         is_negative,
         has_gender_lang, has_factual_lang,
         commenter_gender) %>%
  mutate(across(c(is_negative, has_gender_lang, has_factual_lang),
                ~ factor(.x, levels = c(TRUE, FALSE), labels = c("TRUE", "FALSE")))) %>%
  tbl_summary(
    by = keyword_group,
    statistic = list(
      all_categorical() ~ "{p}%"
    ),
    digits = list(
      all_categorical() ~ 1
    ),
    label = list(
      is_negative        ~ "Negative comment",
      has_gender_lang    ~ "Gender language",
      has_factual_lang   ~ "Factual language",
      commenter_gender   ~ "Commenter gender"
    ),
    type = list(
      is_negative      ~ "categorical",
      has_gender_lang  ~ "categorical",
      has_factual_lang ~ "categorical"
    )
  ) %>%
  add_overall(col_lab = "**Overall**", last = TRUE) %>%
  modify_header(all_stat_cols() ~ "{level}") %>%
  modify_header(label ~ "Variable") %>%
  modify_footnote(all_stat_cols() ~ NA)

tbl2 <- tbl_stack(list(tbl2_top, tbl2_bot)) %>%
  as_gt() %>%
  tab_header(title = "Descriptive Statistics of Weibo Comments") %>%
  tab_spanner(
    label   = "Framing Group",
    columns = starts_with("stat_")
  ) %>%
  tab_footnote(
    footnote  = md("Categorical variables are reported as percentages."),
    locations = cells_title()
  ) %>%
  tab_footnote(
    footnote  = md("Negative comment: contains the commenter's own negative evaluation or emotion, including criticism, anger, fear or sadness. Comments combining praise and negative expression are included, regardless of the target or severity."),
    locations = cells_body(rows = label == "Negative comment", columns = "label")
  ) %>%
  tab_footnote(
    footnote  = md("Gender language: gender emphasis, comparisons, generalisations, stereotypes or gender-based insults."),
    locations = cells_body(rows = label == "Gender language", columns = "label")
  ) %>%
  tab_footnote(
    footnote = md("Factual language: reference to concrete event details, actions, consequences, evidence or legal issues."),
    locations = cells_body(rows = label == "Factual language", columns = "label")
  ) %>%
  tab_source_note(md(SOURCE_NOTE))

save_gt_html(tbl2, file.path(OUTPUT_DIR, "table2_comments_descriptive.html"))
cat("Table 2 saved\n")


# ════════════════════════════════════════════════════════════
# 表3：分话题 × 分组 评论数量交叉表
# ════════════════════════════════════════════════════════════

tbl3 <- comments %>%
  mutate(keyword_topic = recode(keyword_topic, !!!TOPIC_LABELS)) %>%
  count(keyword_topic, keyword_group, .drop = FALSE) %>%
  pivot_wider(names_from = keyword_group, values_from = n, values_fill = 0) %>%
  mutate(Total = rowSums(across(where(is.numeric)))) %>%
  bind_rows(
    summarise(., across(where(is.numeric), sum),
              keyword_topic = "Total")
  ) %>%
  # 计算每格在各列合计中的百分比，并拼成 "n (x.x%)" 格式
  mutate(
    across(
      c(`A Female-gendered`, `B Neutral`, `C Male-gendered`, Total),
      ~ {
        col_total <- sum(.[keyword_topic != "Total"])
        paste0(
          formatC(., format = "d", big.mark = ","),
          " (",
          formatC(. / col_total * 100, digits = 1, format = "f"),
          "%)"
        )
      }
    )
  ) %>%
  gt(rowname_col = "keyword_topic") %>%
  tab_header(title = "Table 3: Number of Comments by Topic and Framing Group") %>%
  tab_spanner(label = "Framing Group",
              columns = c("A Female-gendered", "B Neutral", "C Male-gendered")) %>%
  tab_style(
    style     = cell_text(weight = "bold"),
    locations = cells_body(rows = keyword_topic == "Total")
  ) %>%
  tab_source_note(md(SOURCE_NOTE))

save_gt_html(tbl3, file.path(OUTPUT_DIR, "table3_comments_crosstab.html"))
cat("Table 3 saved\n")



cat("Three original-style tables saved in: ", OUTPUT_DIR, "\n", sep = "")
