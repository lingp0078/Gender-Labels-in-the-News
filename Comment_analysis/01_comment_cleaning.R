# ============================================================
# 微博评论数据清洗脚本 v3
# 假设范围：H1, H5（含Word2Vec延伸）
#
# 输入字段说明（新版Excel）：
#   post_mid, comment_text, comment_date, comment_like_count,
#   comment_reply_count, commenter_id, commenter_name,
#   commenter_gender, commenter_verified, commenter_followers_count,
#   is_reply, reply_to_user, keyword_group, keyword_topic,
#   search_keyword
#   注：comment_id已删除；comment_text已手动清理前缀；
#       commenter_followers_count已转为纯数字
#
# 输出：comments_flagged.csv（全量含标记）
#        comments_clean.csv（清洗后保留行）
# ============================================================

library(readxl)
library(dplyr)
library(stringr)

# 修改为本地实际路径
comments <- read_excel(
  "Weibo_Comments_select.xlsx",
  col_types = "text"
)

# 数值列转换
comments <- comments %>%
  mutate(
    comment_like_count        = as.integer(comment_like_count),
    comment_reply_count       = as.integer(comment_reply_count),
    commenter_followers_count = as.numeric(commenter_followers_count),
    is_reply                  = as.integer(is_reply),
    commenter_verified        = commenter_verified == "TRUE"
  )

cat("原始评论数：", nrow(comments), "\n\n")


# ── 水评论词表 ────────────────────────────────────────────────
water_list <- c(
  "转发微博", "转发", "转",
  "关注", "关注了", "关注关注", "持续关注！", "保持热度，持续关注", "持续关注中","持续关注下",
  "早上好", "上午好", "中午好", "下午好", "晚上好", "晚安",
  "早安", "早上好呀", "晚上好呀",
  "点赞", "点赞了", "赞", "赞赞",
  "支持", "支持一下",
  "感谢", "谢谢", "谢谢了",
  "加油", "好的", "嗯", "嗯嗯", "是的",
  "持续关注", "了解", "注意安全", "安全第一",
  "图片评论", "吃瓜",
  "\U0001f44d", "\U0001f4aa", "\U0001f64f",
  "\U0001f64f\U0001f64f", "\U0001f64f\U0001f64f\U0001f64f"
)

# ── 垃圾广告词表 ─────────────────────────────────────────────
spam_list <- c(
  "加微信", "加我微信", "私信我", "扫码", "扫一扫",
  "立即购买", "点击购买", "点击链接", "代购", "优惠券",
  "限时优惠", "限时特惠", "厂家直供",
  "http", "www\\."
)

# ── E7 刷屏细化：含以下特征的清理，其余保留 ─────────────────
robot_drop_pattern <- paste(c(
  "@",
  "【|】",
  "#",
  "关注",
  "晚上好|早上好|早安|上午好|中午好|下午好|晚安",
  "网页链接",
  '^["\u201c].+["\u201d]$'
), collapse = "|")


# ── 清洗标记（全部在一个 mutate 块里）────────────────────────
# 注：comment_text 已由用户手动清理"回复@xxx的表态:"前缀，无需脚本处理
comments <- comments %>%
  mutate(
    comment_text_clean = str_trim(as.character(comment_text)),
    
    # E1：空内容
    mark_empty = ifelse(
      is.na(comment_text) | comment_text_clean == "",
      1L, 0L
    ),
    
    # E2：纯数字
    mark_numeric = ifelse(
      str_detect(comment_text_clean, "^[0-9\\s]+$"),
      1L, 0L
    ),
    
    # E3：纯标点/emoji
    mark_symbol = ifelse(
      str_detect(comment_text_clean,
                 "^[^\\u4e00-\\u9fa5a-zA-Z0-9]+$") &
        comment_text_clean != "",
      1L, 0L
    ),
    
    # E4：净汉字 <= 1
    text_for_len = str_remove_all(
      str_remove_all(comment_text_clean, "\\[[^\\]]{1,15}\\]"),
      "\\s+"
    ),
    net_cjk = str_count(text_for_len, "[\\u4e00-\\u9fa5]"),
    mark_tooshort = ifelse(
      net_cjk <= 1 &
        mark_empty == 0 & mark_numeric == 0 & mark_symbol == 0,
      1L, 0L
    ),
    
    # E5：水评论（词表 + 纯话题标签 + 含关注@ + 含【或#去除后无实质内容 + 纯@提及）
    text_no_topic = str_trim(
      str_remove_all(comment_text_clean, "#[^#]+#")
    ),
    text_no_bracket = str_trim(str_remove_all(
      str_remove_all(
        str_remove_all(comment_text_clean, "【[^】]*】"),
        "#[^#]+#+"
      ),
      "https?://\\S+|网页链接"
    )),
    net_cjk_bracket = str_count(
      str_remove_all(text_no_bracket, "\\s+"),
      "[\\u4e00-\\u9fa5]"
    ),
    text_no_at = str_trim(
      str_remove_all(comment_text_clean, "@\\S+")
    ),
    
    mark_water = ifelse(
      # 精确匹配：整条评论等于水评论词表中的词
      str_trim(comment_text_clean) %in% water_list |
        # 包含匹配：只要含"谢谢分享"或"感谢分享"字眼就删
        str_detect(comment_text_clean, "谢谢分享|感谢分享") |
        text_no_topic == "" |
        (str_detect(comment_text_clean, "关注") &
           str_detect(comment_text_clean, "@")) |
        (str_detect(comment_text_clean, "【|#") &
           net_cjk_bracket <= 2) |
        (text_no_at == "" &
           str_detect(comment_text_clean, "@")),
      1L, 0L
    ),
    
    # E6：垃圾广告
    mark_spam = ifelse(
      str_detect(comment_text_clean,
                 paste(spam_list, collapse = "|")),
      1L, 0L
    )
  )


# ── E7：刷屏细化 ──────────────────────────────────────────────
freq_table <- comments %>%
  filter(mark_empty == 0 & mark_water == 0) %>%
  count(text_for_len, name = "freq") %>%
  filter(freq >= 5)

comments <- comments %>%
  left_join(freq_table, by = "text_for_len") %>%
  mutate(freq = coalesce(as.integer(freq), 0L))

comments <- comments %>%
  mutate(
    mark_robot = ifelse(
      freq >= 5 &
        str_detect(comment_text_clean, robot_drop_pattern),
      1L, 0L
    ),
    freq = NULL
  )


# ── 汇总删除标记 ─────────────────────────────────────────────
comments <- comments %>%
  mutate(
    mark_delete = ifelse(
      mark_empty == 1 | mark_numeric  == 1 | mark_symbol   == 1 |
        mark_tooshort == 1 | mark_water == 1 | mark_spam    == 1 |
        mark_robot == 1,
      1L, 0L
    ),
    delete_reason = case_when(
      mark_empty    == 1 ~ "E1_空内容",
      mark_numeric  == 1 ~ "E2_纯数字",
      mark_symbol   == 1 ~ "E3_纯符号",
      mark_tooshort == 1 ~ "E4_净汉字<=1",
      mark_water    == 1 ~ "E5_水评论",
      mark_spam     == 1 ~ "E6_广告",
      mark_robot    == 1 ~ "E7_刷屏",
      TRUE               ~ "保留"
    )
  )


# ── 分析字段（精简版，仅 H1/H5 需要）────────────────────────
# 点赞排名分组（保留，后续分析备用）
comments <- comments %>%
  mutate(
    like_rank_group = case_when(
      comment_like_count == 0 ~ "low",
      comment_like_count >= quantile(
        comment_like_count[comment_like_count > 0], 0.5, na.rm = TRUE
      ) ~ "high",
      TRUE ~ "medium"
    ),
    
    # 情感分析占位（由 02_sentiment_analysis.py 填充）
    sentiment            = NA_integer_,
    sentiment_confidence = NA_real_,
    is_negative          = NA_integer_,
    
    # H5 规则字段占位（由 02_sentiment_analysis.py 填充）
    has_gender_lang  = NA_integer_,
    has_factual_lang = NA_integer_
  )


# ── 清洗统计 ─────────────────────────────────────────────────
total  <- nrow(comments)
n_del  <- sum(comments$mark_delete)
n_keep <- total - n_del

cat("════════════════════════════════\n")
cat("清洗统计\n")
cat("════════════════════════════════\n")
cat("原始评论：", total,  "\n")
cat("标记删除：", n_del,  "\n")
cat("保留：    ", n_keep, "\n\n")
cat("── 各规则删除数 ──\n")
cat("  E1 空内容：        ", sum(comments$mark_empty),    "\n")
cat("  E2 纯数字：        ", sum(comments$mark_numeric),  "\n")
cat("  E3 纯符号：        ", sum(comments$mark_symbol),   "\n")
cat("  E4 净汉字<=1：     ", sum(comments$mark_tooshort), "\n")
cat("  E5 水评论：        ", sum(comments$mark_water),    "\n")
cat("  E6 广告：          ", sum(comments$mark_spam),     "\n")
cat("  E7 刷屏（细化后）：", sum(comments$mark_robot),    "\n\n")

cat("── 按 keyword_group 清理率 ──\n")
print(
  comments %>%
    group_by(keyword_group) %>%
    summarise(
      总数   = n(),
      删除   = sum(mark_delete),
      清理率 = paste0(round(mean(mark_delete) * 100, 1), "%"),
      .groups = "drop"
    )
)


# ── 输出文件 ──────────────────────────────────────────────────
output_dir <- "."   # 修改为本地输出路径
drop_cols  <- c("comment_text_clean", "text_for_len", "net_cjk",
                "text_no_topic", "text_no_bracket",
                "net_cjk_bracket", "text_no_at")

write.csv(
  comments %>% select(-all_of(drop_cols)),
  file.path(output_dir, "comments_flagged.csv"),
  row.names = FALSE, fileEncoding = "UTF-8"
)
write.csv(
  comments %>%
    filter(mark_delete == 0) %>%
    select(-all_of(drop_cols)),
  file.path(output_dir, "comments_clean.csv"),
  row.names = FALSE, fileEncoding = "UTF-8"
)

cat(sprintf("\n✓ comments_flagged.csv（%d 行）\n", nrow(comments)))
cat(sprintf("✓ comments_clean.csv（%d 行）\n", n_keep))
cat("\n下一步：运行 02_sentiment_analysis.py\n")