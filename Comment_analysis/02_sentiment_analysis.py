"""
微博评论情感分析脚本 v6（精简版）
===================================
模型：IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment（二分类）

字段说明（精简，去除冗余字段）：
  sentiment              -1/0/1
  sentiment_confidence   模型置信度 0-1
  is_negative            0/1
    · 包含词典强制修正结果（仙女/蝈蝻/死刑/严惩等）
    · 词典修正直接并入 is_negative，不单独保留 corrected_by_dict
  has_gender_lang        0/1  （H5核心：含性别词/讽刺词即为1）
  has_factual_lang       0/1  （H5对照：含话题客观事实词）

移除字段：
  · is_extreme_negative（对H1/H5无用）
  · corrected_by_dict（词典修正直接进 is_negative）
  · gender_hostility（并入 has_gender_lang）
  · has_gender_discourse（合并到 has_gender_lang）
  · is_nonelite（H4不研究了）

标签映射：
  Negative 且 confidence >= 0.75  → -1
  Positive 且 confidence >= 0.75  →  1
  任意     且 confidence <  0.75  →  0（中性/模糊）
  含强制负面词                    →  直接 -1（不受阈值限制）

安装依赖（命令行）：
  pip install transformers torch pandas tqdm
"""

import pandas as pd
import torch
import re
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# ── 配置 ─────────────────────────────────────────────────────
INPUT_PATH  = "comments_clean.csv"      # 修改为本地实际路径
OUTPUT_PATH = "comments_sentiment.csv"

MODEL_NAME        = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
BATCH_SIZE        = 64
MAX_LENGTH        = 128
NEUTRAL_THRESHOLD = 0.75   # 低于此置信度 → 中性


# ════════════════════════════════════════════════════════════
# 词典定义
# ════════════════════════════════════════════════════════════

# ── 强制负面词（不受置信度阈值限制，直接 is_negative=1）────
FORCE_NEGATIVE_WORDS = [
    "死刑", "判死", "判死刑", "枪毙", "极刑",
    "严刑", "重刑", "严惩", "重判", "严判",
    "毙了", "杀之", "处死","畜牲", "离谱",
    "气愤", "气人", "愤怒", "愤慨", "义愤",
    "可耻", "无耻", "耻辱", "毒瘤",
    "该杀", "活该死",
]

# ── 性别攻击词（强制 is_negative=1 + has_gender_lang=1）────
GENDER_NEGATIVE_WORDS = [
    # 攻击女性
    "女拳", "打拳", "拳师", "田园女权", "圣母婊",
    "婆娘", "娘们", "臭婆娘","表子","小作文",
    "公主病", "女权疯子", "极端女权","渣女",
    "母狗", "母的", "女流氓", "泼妇","三八",
    # 攻击男性
    "蝈蝻", "男权癌", "直男癌","楠的",
    "渣男", "废物男", "男狗", "公狗",
    "臭男人", "死男人","狗奴","渣男","男宝",
    # 讽刺词（语境下100%负面）
    "仙女", "小仙女", "公主",
]

# ── 通用侮辱词（强制 is_negative=1）────────────────────────
GENERAL_HOSTILE_WORDS = [
    "zz", "szd", "nmsl","sb",
    "弱智", "智障", "傻逼", "傻b",
    "垃圾人", "人渣", "畜生","畜牲","牲口",
    "去死", "该死", "死吧",
]

# ── has_gender_lang：含性别词即为1（H5核心）────────────────
# 覆盖范围：直接性别词 + 群体泛化 + 性别攻击词
GENDER_LANG_PATTERNS = [
    # 直接性别词（最宽覆盖，只要提到性别就算边界划定）
    r"女性", r"男性", r"女人", r"男人",
    r"女的", r"男的",
    r"女司机", r"男司机",
    r"女生", r"男生",
    r"女孩", r"男孩",
    # 群体泛化表达
    r"你们[男女]",
    r"[男女][的人们]就是",
    r"[男女]人都是",
    r"所有[男女]",
]
GENDER_LANG_REGEX = re.compile("|".join(GENDER_LANG_PATTERNS))

# ── has_factual_lang：话题客观事实词（H5对照）───────────────
FACTUAL_WORDS = {
    "交通":    ["肇事", "全责", "赔偿", "保险", "违章", "超速",
                "闯红灯", "交警", "处罚", "吊销", "驾照",
                "碰撞", "追尾", "刮蹭", "酒驾", "醉驾"],
    "邻里纠纷": ["物业", "楼道", "噪音", "业主", "违规停放",
                "公共区域", "投诉", "物业规定"],
    "消费纠纷": ["三包", "退款", "消费者权益", "工商", "12315",
                "赔偿标准", "合同"],
    "职业学历": ["学术不端", "论文", "职称", "劳动法",
                "合同", "仲裁", "工资"],
    "治安事件": ["刑拘", "立案", "判决", "量刑", "证据",
                "报警", "处罚"],
}


# ════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════
def contains_any(text, word_list):
    t = str(text).lower()
    return any(w.lower() in t for w in word_list)


# ════════════════════════════════════════════════════════════
# 1. 加载模型
# ════════════════════════════════════════════════════════════
print("加载模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model     = model.to(device)
model.eval()

print(f"标签配置：{model.config.id2label}")
assert model.config.id2label[0] == "Negative", \
    f"标签顺序异常：{model.config.id2label}"
print()


# ════════════════════════════════════════════════════════════
# 2. 读取数据
# ════════════════════════════════════════════════════════════
print(f"读取数据：{INPUT_PATH}")
df = pd.read_csv(INPUT_PATH, dtype={"post_mid": str})
print(f"总评论数：{len(df)}\n")
df["comment_text"] = df["comment_text"].fillna("").astype(str)


# ════════════════════════════════════════════════════════════
# 3. 批量推理
# ════════════════════════════════════════════════════════════
def predict_batch(texts):
    encoded = tokenizer(
        texts,
        padding        = True,
        truncation     = True,
        max_length     = MAX_LENGTH,
        return_tensors = "pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        probs = softmax(model(**encoded).logits, dim=-1)
    return (
        probs.argmax(dim=-1).cpu().tolist(),
        probs.max(dim=-1).values.cpu().tolist()
    )

print("开始情感预测...")
start      = time.time()
texts_list = df["comment_text"].tolist()
all_lids   = []
all_confs  = []

for i in tqdm(range(0, len(texts_list), BATCH_SIZE), desc="推理进度"):
    lids, confs = predict_batch(texts_list[i: i + BATCH_SIZE])
    all_lids.extend(lids)
    all_confs.extend(confs)

print(f"推理完成，耗时：{(time.time()-start)/60:.1f} 分钟\n")


# ════════════════════════════════════════════════════════════
# 4. 情感标签映射
# ════════════════════════════════════════════════════════════
def map_sentiment(label_id, confidence):
    if confidence < NEUTRAL_THRESHOLD:
        return 0
    return -1 if label_id == 0 else 1

df["sentiment"]            = [map_sentiment(l, c)
                               for l, c in zip(all_lids, all_confs)]
df["sentiment_confidence"] = [round(c, 4) for c in all_confs]
df["is_negative"]          = (df["sentiment"] == -1).astype(int)

print("模型原始情感分布：")
print(df["sentiment"].value_counts().sort_index())
print()


# ════════════════════════════════════════════════════════════
# 5. 词典后处理（直接修改 is_negative，不保留中间字段）
# ════════════════════════════════════════════════════════════
print("词典后处理...")
cnt = 0

for idx, row in df.iterrows():
    text = str(row["comment_text"])
    sent = row["sentiment"]

    # 强制负面词
    if contains_any(text, FORCE_NEGATIVE_WORDS) and sent != -1:
        df.at[idx, "sentiment"]            = -1
        df.at[idx, "is_negative"]          = 1
        df.at[idx, "sentiment_confidence"] = 0.85
        cnt += 1

    # 性别攻击词 + 通用侮辱词（无条件负面）
    all_hostile = GENDER_NEGATIVE_WORDS + GENERAL_HOSTILE_WORDS
    if contains_any(text, all_hostile) and df.at[idx, "sentiment"] != -1:
        df.at[idx, "sentiment"]            = -1
        df.at[idx, "is_negative"]          = 1
        df.at[idx, "sentiment_confidence"] = 0.85
        cnt += 1

print(f"词典修正条数：{cnt}\n")
print("词典修正后情感分布：")
print(df["sentiment"].value_counts().sort_index())
print()


# ════════════════════════════════════════════════════════════
# 6. 规则字段生成（向量化）
# ════════════════════════════════════════════════════════════
print("生成规则字段...")
texts_s = df["comment_text"].fillna("").astype(str)

# has_gender_lang：含性别词/讽刺词即为1
# 同时覆盖 GENDER_NEGATIVE_WORDS（仙女/蝈蝻等算性别词）
gender_neg_pattern = re.compile(
    "|".join(re.escape(w) for w in GENDER_NEGATIVE_WORDS)
)
df["has_gender_lang"] = (
    texts_s.str.contains(GENDER_LANG_REGEX,    na=False) |
    texts_s.str.contains(gender_neg_pattern,   na=False)
).astype(int)

# has_factual_lang：话题客观事实词
def check_factual(row):
    words = FACTUAL_WORDS.get(str(row["keyword_topic"]), [])
    if not words:
        return 0
    t = str(row["comment_text"]).lower()
    return int(any(w in t for w in words))

df["has_factual_lang"] = df.apply(check_factual, axis=1)

print(f"has_gender_lang=1：  {df['has_gender_lang'].sum()}")
print(f"has_factual_lang=1： {df['has_factual_lang'].sum()}\n")


# ════════════════════════════════════════════════════════════
# 7. 分组统计报告
# ════════════════════════════════════════════════════════════
print("════════════════════════════════")
print("分组分析报告")
print("════════════════════════════════\n")

print("── 各 keyword_group ──")
print(df.groupby("keyword_group").agg(
    总数          = ("sentiment", "count"),
    负面率        = ("is_negative", "mean"),
    性别话语率    = ("has_gender_lang", "mean"),
    事实话语率    = ("has_factual_lang", "mean"),
).round(4))

print("\n── 各话题 × 分组 负面率 ──")
print(df.groupby(["keyword_topic", "keyword_group"])[
    "is_negative"].mean().unstack().round(3))

print("\n── 各话题 × 分组 性别话语率 ──")
print(df.groupby(["keyword_topic", "keyword_group"])[
    "has_gender_lang"].mean().unstack().round(3))


# ════════════════════════════════════════════════════════════
# 8. 保存结果
# ════════════════════════════════════════════════════════════
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n✓ 已保存：{OUTPUT_PATH}（{len(df)} 行）")
print("下一步：运行 04_statistical_analysis.R")
