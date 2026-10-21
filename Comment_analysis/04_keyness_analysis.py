"""
Keyness Analysis Script
========================
Method: Log-likelihood keyness (Dunning 1993), standard corpus-linguistics approach.
Comparisons:
  A (female-gendered) vs B (neutral) — primary hypothesis group
  C (male-gendered)   vs B (neutral)
  A vs C              — direct gendered-frame contrast

Corpora:
  Primary analysis : comments where is_negative=1 AND has_gender_lang=1
  Supplementary    : all comments where is_negative=1

Outputs:
  1. Keyness statistics tables (LL, p-value, effect size)
  2. A vs B keyword bar chart
  3. C vs B keyword bar chart
  4. A / C keyword word clouds
  5. Topic-level keyword heatmap

Dependencies: pip install jieba scipy wordcloud matplotlib pandas seaborn
"""

import pandas as pd
import numpy as np
import jieba
import re
import os
from collections import Counter
from scipy.stats import chi2_contingency
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ── Configuration ───────────────────────────────────────────
INPUT_PATH = os.environ.get("WEIBO_COMMENTS_PATH", "comments_sentiment.csv")
OUTPUT_DIR = os.environ.get("WEIBO_KEYNESS_OUTPUT", "keyness_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chinese font path — update for your OS
# Windows: C:/Windows/Fonts/simhei.ttf  or  msyh.ttc
# Mac:     /System/Library/Fonts/PingFang.ttc
# Linux:   /usr/share/fonts/truetype/wqy/wqy-microhei.ttc
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

matplotlib.rcParams["font.family"]        = ["DejaVu Sans", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Chinese stopwords — particles, conjunctions, filler words, and platform-specific noise
STOPWORDS = set([
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "这", "那", "还", "来",
    "吗", "呢", "吧", "啊", "哦", "嗯", "呀", "哈","出来",
    "但是", "因为", "所以", "然后", "如果", "虽然","还有", "觉得",
    "可以", "这个", "那个", "这样", "那样", "什么", "怎么",
    "一个", "没有","自己","一定", "有点","真是","这是","是不是",
    "http", "转发", "微博", "网页", "链接","知道","不会", "一样",
    "这种", "这么", "不是", "应该", "真的", "就是", "必须", "的话",
    "还是", "不能", "现在", "直接", "不要", "可能", "为什么",
    "不用", "哈哈哈","有没有","怎样",
])


# ════════════════════════════════════════════════════════════
# 1. Load and pre-process data
# ════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(INPUT_PATH, dtype={"post_mid": str})
df["comment_text"] = df["comment_text"].fillna("").astype(str)

# Reconstruct has_gender_lang from raw text if not already present in the CSV
if "has_gender_lang" not in df.columns:
    PATTERNS = [
        r"女性", r"男性", r"女人", r"男人",
        r"女的", r"男的", r"女司机", r"男司机",
        r"女生", r"男生", r"女孩", r"男孩",
        r"你们[男女]", r"[男女][的人们]就是",
        r"[男女]人都是", r"所有[男女]",
    ]
    HOSTILE_WORDS = [
        "女拳","打拳","拳师","田园女权","圣母婊",
        "婆娘","娘们","臭婆娘","公主病","女权疯子","极端女权",
        "母狗","母的","女流氓","泼妇",
        "蝈蝻","男权癌","直男癌","渣男","废物男",
        "男狗","公狗","臭男人","死男人",
        "仙女","小仙女","公主",
    ]
    regex_g   = re.compile("|".join(PATTERNS))
    regex_h   = re.compile("|".join(re.escape(w) for w in HOSTILE_WORDS))
    df["has_gender_lang"] = (
        df["comment_text"].str.contains(regex_g, na=False) |
        df["comment_text"].str.contains(regex_h, na=False)
    ).astype(int)

# Define corpora for each framing group and analysis type
def get_corpus(group, require_gender=True):
    mask = (df["keyword_group"] == group) & (df["is_negative"] == 1)
    if require_gender:
        mask = mask & (df["has_gender_lang"] == 1)
    return df[mask]["comment_text"].tolist()

corpus = {
    "A_neg_gender": get_corpus("A_gendered", require_gender=True),
    "B_neg_gender": get_corpus("B_neutral",  require_gender=True),
    "C_neg_gender": get_corpus("C_gendered", require_gender=True),
    "A_neg_all":    get_corpus("A_gendered", require_gender=False),
    "B_neg_all":    get_corpus("B_neutral",  require_gender=False),
    "C_neg_all":    get_corpus("C_gendered", require_gender=False),
}

print("Corpus sizes:")
for k, v in corpus.items():
    print(f"  {k}: {len(v)} comments")
print()


# ════════════════════════════════════════════════════════════
# 2. Chinese tokenisation using jieba
# ════════════════════════════════════════════════════════════
print("Tokenising...")

# Add compound terms to the jieba dictionary to prevent incorrect segmentation
custom_words = [
    "女司机", "男司机", "小仙女", "女拳", "直男癌",
    "蝈蝻", "田园女权", "公主病", "圣母婊", "女乘客",
    "女性化", "男性化", "性别歧视", "厌女",
]
for w in custom_words:
    jieba.add_word(w, freq=100)   # freq parameter required in older jieba versions

def tokenize(texts):
    """Tokenise a list of texts; return a word-frequency Counter."""
    all_tokens = []
    for text in texts:
        text = re.sub(r"https?://\S+|@\S+|\[[\w\s·]{1,15}\]|#[^#]+#", "", text)
        tokens = list(jieba.cut(text))   # jieba.cut returns a generator; convert to list
        tokens = [t for t in tokens
                  if len(t) >= 2
                  and t not in STOPWORDS
                  and not t.isdigit()
                  and re.search(r"[\u4e00-\u9fa5]", t)]
        all_tokens.extend(tokens)
    return Counter(all_tokens)

freq = {k: tokenize(v) for k, v in corpus.items()}
print("Tokenisation complete.\n")


# ════════════════════════════════════════════════════════════
# 3. Keyness calculation (Log-likelihood method)
# ════════════════════════════════════════════════════════════
def log_likelihood(freq_focus, freq_ref, total_focus, total_ref):
    """
    Compute the likelihood-ratio G^2 for the union of the two vocabularies.
    focus = target corpus (e.g. Group A), ref = reference corpus (e.g. Group B).
    Returns a DataFrame with LL, p-value, and %DIFF effect size.
    Positive LL = more frequent in focus (keyword of focus corpus).
    Negative LL = more frequent in reference corpus.
    """
    if total_focus <= 0 or total_ref <= 0:
        raise ValueError("Both corpora must contain at least one retained token.")
    all_words = sorted(set(freq_focus.keys()) | set(freq_ref.keys()))
    results = []

    for word in all_words:
        a = freq_focus.get(word, 0)   # count in focus corpus
        b = freq_ref.get(word, 0)     # count in reference corpus
        c = total_focus - a           # all other tokens in focus corpus
        d = total_ref - b             # all other tokens in reference corpus

        if a + b == 0 or c + d == 0:
            continue

        # G^2 = 2 * sum(O * log(O/E)); zero cells contribute zero.
        # The p-value uses the asymptotic chi-square distribution with df=1.
        # SciPy's default is Pearson X^2, so lambda_ must be explicit.
        table = [[a, b], [c, d]]
        g2, p, _, _ = chi2_contingency(
            table, correction=False, lambda_="log-likelihood"
        )

        # Direction: positive if word is more frequent in focus than reference
        direction = 1 if (a / total_focus) >= (b / total_ref) else -1
        ll        = g2 * direction

        # Effect size: %DIFF (per-million normalised frequency difference)
        norm_a    = (a / total_focus) * 1_000_000   # frequency per million tokens
        norm_b    = (b / total_ref)   * 1_000_000
        pct_diff  = ((norm_a - norm_b) / (norm_b + 0.001)) * 100

        results.append({
            "word":      word,
            "freq_focus": a,
            "freq_ref":   b,
            "norm_focus": round(norm_a, 2),
            "norm_ref":   round(norm_b, 2),
            # Retain full precision for ranking and significance selection.
            # LL is signed G^2; G2 is the unsigned test statistic.
            "LL":         ll,
            "G2":         g2,
            "p_value":    p,
            "pct_diff":   round(pct_diff, 2),
        })

    result_df = pd.DataFrame(results, columns=[
        "word", "freq_focus", "freq_ref", "norm_focus", "norm_ref",
        "LL", "G2", "p_value", "pct_diff"
    ])
    result_df = result_df.sort_values(
        ["LL", "word"], ascending=[False, True]
    ).reset_index(drop=True)
    return result_df


print("Computing keyness (A vs B, negative + gender language comments)...")
total_A = sum(freq["A_neg_gender"].values())
total_B = sum(freq["B_neg_gender"].values())
total_C = sum(freq["C_neg_gender"].values())

kn_AB = log_likelihood(freq["A_neg_gender"], freq["B_neg_gender"],
                       total_A, total_B)
kn_CB = log_likelihood(freq["C_neg_gender"], freq["B_neg_gender"],
                       total_C, total_B)
kn_AC = log_likelihood(freq["A_neg_gender"], freq["C_neg_gender"],
                       total_A, total_C)

# Keep only significant keywords (p < .05, positive LL, minimum 3 occurrences in focus)
def sig_keys(kn_df, min_freq=3):
    return kn_df[
        (kn_df["p_value"] < 0.05) &
        (kn_df["LL"] > 0) &
        (kn_df["freq_focus"] >= min_freq)
    ].copy()

keys_A    = sig_keys(kn_AB)   # Group A keywords relative to Group B
keys_C    = sig_keys(kn_CB)   # Group C keywords relative to Group B
keys_AC_A = sig_keys(kn_AC)   # Group A keywords relative to Group C

print(f"Group A keywords (vs B): {len(keys_A)}")
print(f"Group C keywords (vs B): {len(keys_C)}")
print(f"Group A keywords (vs C): {len(keys_AC_A)}\n")

# Save full keyness tables as CSV for reference and supplementary appendix
kn_AB.to_csv(os.path.join(OUTPUT_DIR, "keyness_A_vs_B.csv"),
             index=False, encoding="utf-8-sig")
kn_CB.to_csv(os.path.join(OUTPUT_DIR, "keyness_C_vs_B.csv"),
             index=False, encoding="utf-8-sig")
kn_AC.to_csv(os.path.join(OUTPUT_DIR, "keyness_A_vs_C.csv"),
             index=False, encoding="utf-8-sig")
print("✓ Keyness tables saved.\n")

# Print top-20 keywords to console for quick inspection
print("=== Group A: Top 20 keywords (vs Group B) ===")
print(keys_A[["word","freq_focus","norm_focus","norm_ref","LL","p_value","pct_diff"]].head(20).to_string(index=False))
print()
print("=== Group C: Top 20 keywords (vs Group B) ===")
print(keys_C[["word","freq_focus","norm_focus","norm_ref","LL","p_value","pct_diff"]].head(20).to_string(index=False))
print()
print("=== Group A: Top 20 keywords (vs Group C) ===")
print(keys_AC_A[["word","freq_focus","norm_focus","norm_ref","LL","p_value","pct_diff"]].head(20).to_string(index=False))


# ════════════════════════════════════════════════════════════
# 4. Visualisation
# ════════════════════════════════════════════════════════════

# ── Bidirectional bar chart: positive bars = focus keywords, negative = reference keywords
GROUP_NAMES = {"A": "Female-gendered", "B": "Gender-neutral", "C": "Male-gendered"}


# English glosses are display labels, not inputs to tokenization or testing.
KEYWORD_TRANSLATIONS = {'一点': 'a little', '一看': 'at a glance', '不了': 'cannot', '严惩': 'severe punishment', '中国': 'China', '主人': 'owner', '事情': 'matter', '二十分钟': 'twenty minutes', '二级': 'second level', '付出': 'contribution', '仙女': 'fairy', '低头': 'looking down', '作文': 'short essay', '你们': 'you (plural)', '保卫处': 'campus security office', '保安': 'security guard', '信任': 'trust', '儿子': 'son', '允许': 'allow', '公主': 'princess', '公共安全': 'public safety', '养狗': 'keeping dogs', '出租车': 'taxi', '医生': 'doctor', '协和': 'Peking Union Medical College', '危害': 'harm', '危险': 'danger', '原谅': 'forgive', '双方': 'both parties', '司机': 'driver', '吃软饭': 'living off a partner', '吃饭': 'eating', '坚决': 'firmly', '处罚': 'punishment', '外卖': 'food delivery', '大车': 'large vehicle', '女儿': 'daughter', '女司机': 'female driver', '女士': 'lady', '女子': 'woman', '女方': 'female party', '她们': 'they (female)', '妻子': 'wife', '姐姐': 'older sister', '婆婆': 'mother-in-law', '媒体': 'media', '学校': 'school', '孩子': 'children', '安全': 'safety', '完全': 'completely', '宣传': 'publicity', '家暴': 'domestic violence', '小仙女': 'little fairy', '小姐': 'young lady', '小时': 'hour', '就算': 'even if', '屌丝': 'loser', '工作': 'work', '师大': 'normal university', '年龄': 'age', '店家': 'shop owner', '强奸': 'rape', '性别': 'gender', '性骚扰': 'sexual harassment', '恶心': 'disgusting', '恶意': 'malice', '愿意': 'willing', '我们': 'we', '我家': 'my home', '手术': 'surgery', '手机': 'mobile phone', '打假': 'exposing counterfeit products', '技术': 'skill', '抑郁症': 'depression', '投毒': 'poisoning', '护士': 'nurse', '报警': 'calling the police', '抵制': 'boycott', '拘留': 'detention', '摔死': 'fatal fall', '擦边': 'borderline sexual content', '教授': 'professor', '文明': 'civility', '无语': 'speechless', '时候': 'when', '时间': 'time', '智驾': 'intelligent driving', '有些': 'some', '条件': 'conditions', '极端': 'extreme', '校长': 'school principal', '案子': 'case', '楼下': 'downstairs', '死刑': 'death penalty', '殴打': 'beating', '毫无': 'none at all', '没事': 'nothing wrong', '泼妇': 'shrew', '漂亮': 'beautiful', '炸弹': 'bomb', '牢饭': 'prison food', '犯罪': 'crime', '猥亵': 'sexual molestation', '王一博': 'Wang Yibo', '男子': 'man', '百亿': 'ten billion', '真不该': 'really should not', '知假': 'knowing it is counterfeit', '知心': 'understanding', '社区': 'community', '离场': 'leaving the venue', '离婚': 'divorce', '离开': 'leave', '私生': 'obsessive fan', '粉丝': 'fans', '精神': 'mental state', '结婚': 'marriage', '网友': 'internet users', '老头': 'old man', '老师': 'teacher', '老板': 'boss', '能力': 'ability', '补偿': 'compensation', '补贴': 'subsidy', '要求': 'request', '订婚': 'engagement', '诬告': 'false accusation', '货车': 'truck', '超标': 'exceeding the limit', '超雄': 'hypermasculine', '这女': 'this woman', '这群': 'this group', '远离': 'stay away', '造谣': 'spreading rumors', '酒店': 'hotel', '酒驾': 'drunk driving', '重伤': 'serious injury', '重点': 'main point', '重男轻女': 'favoring sons over daughters', '金店': 'jewelry store', '铁路': 'railway', '问题': 'problem', '鞋架': 'shoe rack', '顺风': 'ride-sharing', '马路': 'road', '驿站': 'parcel pickup station', '骂人': 'verbal abuse', '鸽子': 'pigeon'}

def bilingual_keyword(word):
    import textwrap
    if word not in KEYWORD_TRANSLATIONS:
        raise ValueError(f"Missing English display gloss: {word}")
    english = textwrap.fill(KEYWORD_TRANSLATIONS[word], width=40)
    return f"{english}\n({word})"

def render_keyness_chart(kn_df, focus, reference, color_pos, color_neg,
                         tokens_focus, tokens_ref, comments_focus, comments_ref,
                         corpus_label, output_path, topic=None, top_n=12):
    """Bilingual ranked keyness display; statistics and selection are unchanged."""
    from matplotlib.patches import Patch
    import textwrap
    positive = kn_df[kn_df["LL"] > 0].head(top_n)
    negative = kn_df[kn_df["LL"] < 0].tail(top_n).iloc[::-1]
    plot_df = pd.concat([positive, negative])
    if plot_df.empty:
        return
    # The footer shares the plotting area's left edge. Font sizes are set for
    # two-line bilingual labels, rather than shrinking to accommodate them.
    fig, ax = plt.subplots(figsize=(12, max(12, len(plot_df) * .60 + 1.5)))
    left, right = .34, .975
    colors = [color_pos if x > 0 else color_neg for x in plot_df["LL"]]
    bars = ax.barh([bilingual_keyword(w) for w in plot_df["word"]],
                   plot_df["LL"], color=colors, alpha=.88, height=.78)
    ax.axvline(0, color="#555555", linewidth=1)
    extent = max(abs(plot_df["LL"]).max(), 1)
    offset = extent * .018
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        x = bar.get_width()
        count = row["freq_focus"] if x > 0 else row["freq_ref"]
        ax.text(x + (offset if x > 0 else -offset),
                bar.get_y() + bar.get_height() / 2, f"n={int(count)}",
                va="center", ha="left" if x > 0 else "right",
                fontsize=12, color="#555555", fontfamily="DejaVu Sans")
    ax.margins(x=.18, y=.012)
    ax.set_xlabel(r"Signed log-likelihood ($G^2$)", fontsize=15,
                  fontfamily="DejaVu Sans", labelpad=10)
    ax.tick_params(axis="y", labelsize=14, length=0, pad=9)
    ax.tick_params(axis="x", labelsize=13)
    for label in ax.get_xticklabels():
        label.set_fontfamily("DejaVu Sans")
    ax.spines[["top", "right"]].set_visible(False)
    legend = [Patch(facecolor=color_pos, alpha=.88,
                    label=f"Group {focus} ({GROUP_NAMES[focus]})"),
              Patch(facecolor=color_neg, alpha=.88,
                    label=f"Group {reference} ({GROUP_NAMES[reference]})")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(left, .115),
               frameon=False, ncol=2, borderaxespad=0, columnspacing=1.2,
               handlelength=1.4, handletextpad=.5,
               prop={"family":"DejaVu Sans", "size":13})
    fig.suptitle("Distinctive Vocabulary by Headline Group", fontsize=19,
                 fontweight="bold", fontfamily="DejaVu Sans", y=.988)
    subtitle=f"Group {focus} vs Group {reference}"
    if topic:
        subtitle=f"{topic}: " + subtitle
    fig.text(.5, .958, subtitle, ha="center", fontsize=14,
             fontfamily="DejaVu Sans")
    scope = ("all negative comments within the indicated topic"
             if topic else "negative comments containing gender language")
    notes = [
        f"Note: Based on {scope}.",
        "Bars show signed G²; n indicates word occurrences in the corresponding group."
    ]
    fig.text(left, .065, "\n".join(notes), ha="left", va="top",
             fontsize=13, linespacing=1.4, fontfamily="DejaVu Sans")
    fig.subplots_adjust(left=left, right=right, top=.925, bottom=.18)
    output_path=os.fspath(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.splitext(output_path)[0]+".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {os.path.basename(output_path)} saved")

def plot_keyness_bar(kn_df, title, color_pos, color_neg, filename, top_n=25):
    comparisons = {"fig1_keyness_A_vs_B.png": ("A", "B", total_A, total_B),
                   "fig2_keyness_C_vs_B.png": ("C", "B", total_C, total_B),
                   "fig3_keyness_A_vs_C.png": ("A", "C", total_A, total_C)}
    focus, reference, n_focus, n_ref = comparisons[filename]
    counts = df.loc[(df["is_negative"] == 1) & (df["has_gender_lang"] == 1), "keyword_group"].value_counts()
    codes = {"A": "A_gendered", "B": "B_neutral", "C": "C_gendered"}
    render_keyness_chart(kn_df, focus, reference, color_pos, color_neg,
        n_focus, n_ref, int(counts.get(codes[focus], 0)), int(counts.get(codes[reference], 0)),
        "negative comments containing gender language", os.path.join(OUTPUT_DIR, filename), top_n=top_n // 2)

print("\nGenerating figures...")
plot_keyness_bar(
    kn_AB,
    "Keywords: Group A (Female-gendered) vs Group B (Neutral)\nNegative comments containing gender language",
    color_pos="#D64C6F", color_neg="#5B8DB8",
    filename="fig1_keyness_A_vs_B.png"
)
plot_keyness_bar(
    kn_CB,
    "Keywords: Group C (Male-gendered) vs Group B (Neutral)\nNegative comments containing gender language",
    color_pos="#4CAF7D", color_neg="#5B8DB8",
    filename="fig2_keyness_C_vs_B.png"
)
plot_keyness_bar(
    kn_AC,
    "Keywords: Group A (Female-gendered) vs Group C (Male-gendered)\nNegative comments containing gender language",
    color_pos="#D64C6F", color_neg="#4CAF7D",
    filename="fig3_keyness_A_vs_C.png"
)


# ── Word clouds weighted by LL value — larger words = stronger keyness ────
def plot_keyness_wordcloud(keys_df, title, cmap, filename):
    if keys_df.empty:
        print(f"  {filename} skipped — no significant keywords found")
        return
    freq_dict = dict(zip(keys_df["word"], keys_df["LL"].clip(lower=0.1)))
    try:
        wc = WordCloud(
            font_path=FONT_PATH, width=900, height=500,
            background_color="white", colormap=cmap,
            max_words=60, min_font_size=10,
        ).generate_from_frequencies(freq_dict)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {filename} saved")
    except Exception as e:
        print(f"  {filename} failed (font error): {e}")

plot_keyness_wordcloud(
    keys_A,
    "Distinctive Vocabulary: Group A vs Group B\nWord size represents log-likelihood keyness",
    "RdPu", "fig4a_wordcloud_A.png"
)
plot_keyness_wordcloud(
    keys_C,
    "Distinctive Vocabulary: Group C vs Group B\nWord size represents log-likelihood keyness",
    "GnBu", "fig4b_wordcloud_C.png"
)


# ── Topic-level keyword heatmap: LL values for top A-group keywords across topics ──
print("  Generating topic-level keyness heatmap...")
topics = ["交通", "邻里纠纷", "消费纠纷", "职业学历"]

def get_topic_corpus(group, topic, require_gender=True):
    mask = ((df["keyword_group"] == group) &
            (df["keyword_topic"] == topic) &
            (df["is_negative"] == 1))
    if require_gender:
        mask = mask & (df["has_gender_lang"] == 1)
    return df[mask]["comment_text"].tolist()

# Compute A vs B keyness separately for each topic, collect top words per topic
topic_keys = {}
for topic in topics:
    corp_A_t = get_topic_corpus("A_gendered", topic)
    corp_B_t = get_topic_corpus("B_neutral",  topic)
    if len(corp_A_t) < 10 or len(corp_B_t) < 10:
        continue
    freq_A_t = tokenize(corp_A_t)
    freq_B_t = tokenize(corp_B_t)
    tot_A_t  = sum(freq_A_t.values())
    tot_B_t  = sum(freq_B_t.values())
    if tot_A_t == 0 or tot_B_t == 0:
        continue
    kn_t = log_likelihood(freq_A_t, freq_B_t, tot_A_t, tot_B_t)
    k_t  = sig_keys(kn_t, min_freq=2)
    if not k_t.empty:
        topic_keys[topic] = k_t.set_index("word")["LL"]

if len(topic_keys) >= 2:
    # Union of top-10 keywords from each topic — forms the heatmap columns
    top_words_per_topic = set()
    for kn_s in topic_keys.values():
        top_words_per_topic.update(kn_s.nlargest(10).index.tolist())

    top_words_list = sorted(top_words_per_topic)
    matrix = pd.DataFrame(index=list(topic_keys.keys()),
                          columns=top_words_list, data=0.0)
    for topic, kn_s in topic_keys.items():
        for word in top_words_list:
            matrix.loc[topic, word] = kn_s.get(word, 0.0)

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(max(14, len(top_words_list)*0.7), 5))
    sns.heatmap(matrix.astype(float), ax=ax, cmap="YlOrRd",
                linewidths=0.3, annot=False)
    ax.set_title("Topic-level Keyword Heatmap: Group A (Female-gendered) vs Group B (Neutral)\nLL values for significant keywords",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Word")
    ax.set_ylabel("Topic")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_topic_keyness_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig5_topic_keyness_heatmap.png saved")


# ════════════════════════════════════════════════════════════
# 5. Text report — saves all keyness results to a readable .txt file
# ════════════════════════════════════════════════════════════
report_path = os.path.join(OUTPUT_DIR, "keyness_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Keyness Analysis Report\n")
    f.write("Method: Log-likelihood ratio G^2 on a 2x2 table (Dunning 1993)\n")
    f.write("SciPy: correction=False, lambda_='log-likelihood'; df=1\n")
    f.write("LL = signed G^2; G2 = unsigned statistic; p_value = raw asymptotic p.\n")
    f.write("Keyword screen: raw p<.05 and frequency>=3 in the favoured corpus.\n")
    f.write("Exploratory tests: no multiple-testing correction; inspect sparse words.\n")
    f.write("Bar charts rank signed G^2 without this significance/frequency screen.\n")
    f.write("Effect size: %DIFF (Gabrielatos and Marchi 2012)\n")
    f.write("=" * 55 + "\n\n")

    for label, kn_df, n_focus, n_ref in [
        ("A Female-gendered vs B Neutral", kn_AB, total_A, total_B),
        ("C Male-gendered vs B Neutral",   kn_CB, total_C, total_B),
        ("A Female-gendered vs C Male-gendered", kn_AC, total_A, total_C),
    ]:
        f.write(f"【{label}】\n")
        f.write(f"  Focus corpus:     {n_focus} tokens\n")
        f.write(f"  Reference corpus: {n_ref} tokens\n\n")

        pos = sig_keys(kn_df)
        neg = kn_df[(kn_df["p_value"]<0.05) & (kn_df["LL"]<0) &
                    (kn_df["freq_ref"]>=3)].copy()

        f.write(f"  Significant keywords in focus corpus (Top 30):\n")
        for _, row in pos.head(30).iterrows():
            f.write(f"    {row['word']:10s}  LL={row['LL']:8.2f}  "
                    f"n={int(row['freq_focus']):4d}  "
                    f"norm={row['norm_focus']:8.1f}  "
                    f"%DIFF={row['pct_diff']:+.1f}%\n")

        f.write(f"\n  Reference-biased words (more frequent in reference, Top 15):\n")
        for _, row in neg.sort_values(["LL", "word"]).head(15).iterrows():
            f.write(f"    {row['word']:10s}  LL={row['LL']:8.2f}  "
                    f"n_ref={int(row['freq_ref']):4d}\n")
        f.write("\n" + "-"*55 + "\n\n")

print(f"\n✓ Report saved: {report_path}")
print(f"✓ Figures saved to: {OUTPUT_DIR}/")


# ════════════════════════════════════════════════════════════
# 6. EQ1：A 组和 C 组在各话题下的高频负面词
#    分析对象：各组 is_negative=1 的全部评论（不限 has_gender_lang）
#    目的：揭示女性化框架 vs 男性化框架下各话题特有的攻击词汇
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("EQ1: High-frequency negative words by topic — Group A vs Group C")
print("="*55)

ALL_TOPICS = ["交通", "消费纠纷", "邻里纠纷", "职业学历", "治安事件"]
TOPIC_EN   = {
    "交通":   "Traffic",
    "消费纠纷": "Consumer",
    "邻里纠纷": "Neighbourhood",
    "职业学历": "Professional",
    "治安事件": "Public Order",
}

TOPIC_DISPLAY = {"交通": "Traffic incidents", "消费纠纷": "Consumer disputes",
                 "邻里纠纷": "Neighborhood conflicts", "职业学历": "Professional and academic disputes",
                 "治安事件": "Crime and legal disputes"}

# EQ1-A: A vs C keyness by topic (female-gendered vs male-gendered)
# Output 1: bidirectional keyness bar charts per topic
# Output 2: Top-15 frequency bar charts (A and C side by side per topic)
# Output 3: keyness CSV tables per topic

eq1_dir = os.path.join(OUTPUT_DIR, "EQ1_topic_keywords")
os.makedirs(eq1_dir, exist_ok=True)

def top_freq_words(texts, top_n=30, min_count=3):
    """Return the top_n most frequent words as a list of (word, count) tuples."""
    freq = tokenize(texts)
    return [(w, c) for w, c in freq.most_common(top_n) if c >= min_count]

# ── EQ1-A: per-topic A vs C keyness bar charts ──────────────────────────
print("\n[EQ1-A] Per-topic A vs C keyness bar charts...")
for topic in ALL_TOPICS:
    corp_A = get_topic_corpus("A_gendered", topic, require_gender=False)
    corp_C = get_topic_corpus("C_gendered", topic, require_gender=False)
    if len(corp_A) < 10 or len(corp_C) < 10:
        print(f"  {topic}: insufficient corpus size, skipping")
        continue

    freq_A_t = tokenize(corp_A)
    freq_C_t = tokenize(corp_C)
    tot_A_t  = sum(freq_A_t.values())
    tot_C_t  = sum(freq_C_t.values())
    if tot_A_t == 0 or tot_C_t == 0:
        continue

    kn_AC_t = log_likelihood(freq_A_t, freq_C_t, tot_A_t, tot_C_t)

    # Save full keyness statistics to CSV for each topic
    fname_csv = f"keyness_A_vs_C_{TOPIC_EN[topic]}.csv"
    kn_AC_t.to_csv(os.path.join(eq1_dir, fname_csv),
                   index=False, encoding="utf-8-sig")

    fname_fig = f"fig_EQ1_keyness_AC_{TOPIC_EN[topic]}.png"
    render_keyness_chart(kn_AC_t, "A", "C", "#D64C6F", "#4CAF7D",
        tot_A_t, tot_C_t, len(corp_A), len(corp_C), "all negative comments",
        os.path.join(eq1_dir, fname_fig), topic=TOPIC_DISPLAY[topic])

# ── EQ1-B: Top-15 frequency bar charts — A and C side by side per topic ────────
print("\n[EQ1-B] Top-15 high-frequency negative words by topic...")
n_topics = len(ALL_TOPICS)
fig, axes = plt.subplots(n_topics, 2,
                          figsize=(14, n_topics * 3.5),
                          constrained_layout=True)

for row_i, topic in enumerate(ALL_TOPICS):
    for col_i, (grp, color, grp_label) in enumerate([
        ("A_gendered", "#D64C6F", "A Female-gendered"),
        ("C_gendered", "#4CAF7D", "C Male-gendered"),
    ]):
        ax = axes[row_i, col_i]
        corp = get_topic_corpus(grp, topic, require_gender=False)
        top  = top_freq_words(corp, top_n=15, min_count=3)

        if not top:
            ax.text(0.5, 0.5, "insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{TOPIC_DISPLAY[topic]}: {grp_label}", fontsize=9)
            continue

        words  = [w for w, _ in top]
        counts = [c for _, c in top]

        ax.barh(words[::-1], counts[::-1], color=color, alpha=0.82)
        ax.set_xlabel("Frequency")
        ax.set_title(f"{TOPIC_DISPLAY[topic]}: {grp_label}", fontsize=9, fontweight="bold")
        for i, (w, c) in enumerate(zip(words[::-1], counts[::-1])):
            ax.text(c + 0.3, i, str(c), va="center", fontsize=7.5, color="#333")

fig.suptitle(
    "Most Frequent Words in Negative Comments by Topic and Headline Group",
    fontsize=13, fontweight="bold", y=1.01
)
plt.savefig(os.path.join(eq1_dir, "fig_EQ1_top15_by_topic.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  fig_EQ1_top15_by_topic.png saved")

# ── EQ1-C: side-by-side word clouds per topic ─────────────────────────────
print("\n[EQ1-C] Per-topic A / C word cloud comparison...")
for topic in ALL_TOPICS:
    corp_A = get_topic_corpus("A_gendered", topic, require_gender=False)
    corp_C = get_topic_corpus("C_gendered", topic, require_gender=False)

    freq_A_t = tokenize(corp_A)
    freq_C_t = tokenize(corp_C)
    if not freq_A_t or not freq_C_t:
        continue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, freq_dict, title, cmap in [
        (ax1, dict(freq_A_t.most_common(80)),
         "Group A (Female-gendered)", "RdPu"),
        (ax2, dict(freq_C_t.most_common(80)),
         "Group C (Male-gendered)", "GnBu"),
    ]:
        try:
            wc = WordCloud(
                font_path=FONT_PATH, width=700, height=450,
                background_color="white", colormap=cmap,
                max_words=60, min_font_size=9,
            ).generate_from_frequencies(freq_dict)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Font error: {e}", ha="center", va="center")
            ax.set_title(title)

    fig.suptitle(
        f"Frequent Words in Negative Comments: {TOPIC_DISPLAY[topic]}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fname = f"fig_EQ1_wordcloud_{TOPIC_EN[topic]}.png"
    plt.savefig(os.path.join(eq1_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {fname} saved")

# ── EQ1-D: text summary report — top-10 words per topic per group ─────────
eq1_report = os.path.join(eq1_dir, "EQ1_report.txt")
with open(eq1_report, "w", encoding="utf-8") as f:
    f.write("EQ1: High-frequency Negative Words by Topic — Group A vs Group C\n")
    f.write("="*65 + "\n\n")
    for topic in ALL_TOPICS:
        f.write(f"[{TOPIC_EN[topic]}]\n")
        for grp, label in [("A_gendered","A Female-gendered"),
                            ("C_gendered","C Male-gendered")]:
            corp  = get_topic_corpus(grp, topic, require_gender=False)
            top   = top_freq_words(corp, top_n=10, min_count=2)
            f.write(f"  {label}:\n")
            for w, c in top:
                f.write(f"    {w:<10}  n={c}\n")
        f.write("\n")

print(f"\n✓ EQ1 report saved: {eq1_report}")
print(f"✓ EQ1 figures saved to: {eq1_dir}/")
print("\nScript complete.")
