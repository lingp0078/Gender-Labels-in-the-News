# -*- coding: utf-8 -*-
"""
微博爬虫代码
用于学术研究: Gendered Framing in News Headlines
功能: 按关键词+时间范围搜索微博帖子，并爬取评论
输出: CSV文件（帖子 + 评论）

使用方法:
1. 在 weibo_config.json 中填写你的微博Cookie
2. 运行: python weibo_scraper_backup.py
3. 结果保存在同目录下的CSV文件中

"""

import requests
import time
import random
import re
import json
import os
import csv
import traceback
import sys
from datetime import datetime, timedelta
from urllib.parse import quote

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "weibo_config.json")
OUTPUT_DIR = SCRIPT_DIR
USER_CACHE = {}

# 日志文件
log_path = os.path.join(SCRIPT_DIR, "weibo_scraper_log.txt")

def log(msg):
    """带时间戳的日志输出"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + "\n")

def clean_html(text):
    """清除HTML标签"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', str(text))
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text.strip()

def get_headers(cookie):
    """PC端请求头（用于搜索和帖子详情）"""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': cookie,
        'Referer': 'https://s.weibo.com/',
    }

def get_mobile_headers(cookie):
    """移动端请求头（用于评论接口）"""
    return {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': cookie,
        'Referer': 'https://m.weibo.cn/',
        'X-Requested-With': 'XMLHttpRequest',
    }

def load_config():
    """加载配置文件"""
    log("=" * 70)
    log("微博爬虫 v3.0 - 学术研究用途")
    log("=" * 70)
    
    if not os.path.exists(CONFIG_FILE):
        log(f"[错误] 配置文件不存在: {CONFIG_FILE}")
        return None
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if not config.get("cookie") or "粘贴" in config.get("cookie", ""):
        log("[错误] 请先在配置文件中填写你的微博Cookie！")
        return None
    
    keywords = config.get("keywords", [])
    log(f"\n[配置] 共 {len(keywords)} 组关键词")
    log(f"[配置] 最低评论数: {config.get('min_comments', 5)}")
    log(f"[配置] 每关键词最多帖子: {config.get('max_posts_per_keyword', 100)}")
    log(f"[配置] 每帖最多评论: {config.get('max_comments_per_post', 200)}")
    
    return config

def safe_int(val, default=0):
    """安全转换为int"""
    try:
        if isinstance(val, str) and '万' in val:
            return int(float(val.replace('万', '')) * 10000)
        return int(val)
    except:
        return default

def get_user_info(uid, cookie):
    """获取用户详细信息"""
    if not uid or uid in USER_CACHE:
        return USER_CACHE.get(uid, {})
    
    headers = get_headers(cookie)
    try:
        url = f"https://weibo.com/ajax/profile/info?uid={uid}"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data and 'user' in data['data']:
                USER_CACHE[uid] = data['data']['user']
                return data['data']['user']
    except Exception as e:
        log(f"    用户信息获取失败: {e}")
    return {}

def search_posts(keyword, start_date, end_date, config):
    """
    搜索微博帖子
    按周分段搜索以获取更多结果，规避微博搜索截断
    """
    cookie = config["cookie"]
    headers = get_headers(cookie)
    max_posts = config.get("max_posts_per_keyword", 100)
    max_pages = config.get("max_pages_per_search", 50)
    min_comments = config.get("min_comments", 5)
    
    all_posts = []
    seen_mids = set()
    
    # 按周分段搜索
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current_start = start
    segment_days = 7
    
    while current_start <= end:
        current_end = min(current_start + timedelta(days=segment_days - 1), end)
        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")
        
        log(f"\n  [搜索段] '{keyword}' | {start_str} ~ {end_str}")
        
        for page in range(1, max_pages + 1):
            url = f"https://s.weibo.com/weibo?q={quote(keyword)}&typeall=1&suball=1&timescope=custom:{start_str}-0:{end_str}-23&page={page}"
            
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                
                if resp.status_code == 418 or resp.status_code == 403:
                    log(f"    [反爬] 休眠60秒后重试...")
                    time.sleep(60)
                    resp = requests.get(url, headers=headers, timeout=20)
                
                if resp.status_code != 200:
                    log(f"    [错误] HTTP {resp.status_code}")
                    break
                
                html = resp.text
                
                # 检查登录状态
                if '请先登录' in html or 'passport.weibo.com' in html:
                    log(f"    [错误] Cookie已失效，请重新获取！")
                    return all_posts
                
                # 解析mid
                mids = re.findall(r'mid="(\d+)"', html)
                if not mids:
                    mids = re.findall(r'mid=(\d+)', html)
                
                if not mids:
                    break
                
                log(f"    第{page}页: {len(mids)}条")
                
                for mid in mids:
                    if mid in seen_mids:
                        continue
                    seen_mids.add(mid)
                    
                    # 获取帖子详情
                    detail_url = f"https://weibo.com/ajax/statuses/show?id={mid}"
                    try:
                        r = requests.get(detail_url, headers=headers, timeout=10)
                        if r.status_code == 200:
                            data = r.json()
                            if 'mid' in data or 'id' in data:
                                d = data
                                user = d.get('user', {})
                                uid = user.get('id', '')
                                
                                # 获取完整用户信息
                                user_full = get_user_info(uid, cookie) if uid else {}
                                followers = safe_int(user_full.get('followers_count', user.get('followers_count', 0)))
                                verified_type = user_full.get('verified_type', user.get('verified_type', -1))
                                
                                # 认证类型判断
                                if verified_type == -1:
                                    auth_type = "普通用户"
                                elif verified_type == 0:
                                    auth_type = "个人认证(黄V)"
                                elif verified_type in [1, 2, 3, 5, 6, 7]:
                                    auth_type = "机构认证(蓝V)"
                                elif verified_type == 4:
                                    auth_type = "自媒体(黄V)"
                                else:
                                    auth_type = f"其他认证(Type{verified_type})"
                                
                                post = {
                                    # 搜索信息
                                    "search_keyword": keyword,
                                    
                                    # 帖子信息
                                    "post_mid": mid,
                                    "post_url": f"https://weibo.com/{uid}/{mid}",
                                    "post_text": clean_html(d.get('text_raw', '') or d.get('text', '')),
                                    "post_created_at": d.get('created_at', ''),
                                    "post_source": clean_html(d.get('source', '')),
                                    "reposts_count": d.get('reposts_count', 0),
                                    "comments_count": d.get('comments_count', 0),
                                    "attitudes_count": d.get('attitudes_count', 0),
                                    
                                    # 用户信息
                                    "user_id": str(uid),
                                    "user_name": user.get('screen_name', ''),
                                    "user_gender": user.get('gender', ''),
                                    "user_followers_count": followers,
                                    "user_friends_count": user_full.get('friends_count', user.get('friends_count', 0)),
                                    "user_statuses_count": user_full.get('statuses_count', user.get('statuses_count', 0)),
                                    "user_verified": user.get('verified', False),
                                    "user_verified_type": verified_type,
                                    "user_verified_reason": user_full.get('verified_reason', ''),
                                    "user_authentication": auth_type,
                                    "user_description": user_full.get('description', user.get('description', '')),
                                    "is_media": 1 if verified_type in [1, 2, 3, 5] else 0,
                                    "has_image": 1 if d.get('pic_ids') else 0,
                                    "has_video": 1 if d.get('page_info', {}).get('type') == 'video' else 0,
                                    "topic_tags": ",".join(re.findall(r'#(.+?)#', str(d.get('text_raw', '')))),
                                }
                                
                                if post["comments_count"] >= min_comments:
                                    all_posts.append(post)
                                    log(f"      ✓ {mid} | {auth_type} | 粉丝={followers} | 评论={post['comments_count']} | 总计={len(all_posts)}")
                                
                                if len(all_posts) >= max_posts:
                                    return all_posts
                        
                        time.sleep(random.uniform(2, 4))
                    except Exception as e:
                        log(f"      详情失败: {e}")
                        continue
                
                # 检查是否有下一页
                if f'page={page+1}' not in html and 'class="next"' not in html:
                    break
                
                time.sleep(random.uniform(3, 5))
                
            except Exception as e:
                log(f"    [错误] {e}")
                continue
        
        if len(all_posts) >= max_posts:
            break
        
        current_start = current_end + timedelta(days=1)
        time.sleep(random.uniform(5, 8))
    
    return all_posts

def get_comments(mid, config):
    """爬取某条微博的评论"""
    cookie = config["cookie"]
    headers = get_mobile_headers(cookie)
    max_comments = config.get("max_comments_per_post", 200)
    
    comments = []
    max_id = ""
    
    for page in range(1, 100):
        if len(comments) >= max_comments:
            break
        
        if max_id:
            url = f"https://m.weibo.cn/comments/hotflow?id={mid}&mid={mid}&max_id={max_id}&max_id_type=0"
        else:
            url = f"https://m.weibo.cn/comments/hotflow?id={mid}&mid={mid}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            
            if resp.status_code == 418 or resp.status_code == 403:
                time.sleep(30)
                resp = requests.get(url, headers=headers, timeout=15)
            
            if resp.status_code != 200:
                break
            
            data = resp.json()
            
            if data.get("ok") != 1:
                break
            
            cdata = data.get("data", {})
            clist = cdata.get("data", [])
            
            if not clist:
                break
            
            for c in clist:
                user = c.get("user", {})
                comments.append({
                    # 搜索信息
                    "search_keyword": "",
                    "keyword_group": "",
                    "topic": "",
                    
                    # 评论信息
                    "comment_id": str(c.get("id", "")),
                    "post_mid": mid,
                    "comment_text": clean_html(c.get("text", "")),
                    "comment_created_at": c.get("created_at", ""),
                    "comment_like_count": c.get("like_count", 0),
                    "comment_reply_count": c.get("total_number", 0),
                    
                    # 评论者信息
                    "commenter_id": str(user.get("id", "")),
                    "commenter_name": user.get("screen_name", ""),
                    "commenter_gender": user.get("gender", ""),
                    "commenter_verified": user.get("verified", False),
                    "commenter_followers_count": user.get("followers_count", 0),
                    
                    # 回复信息
                    "is_reply": 1 if c.get("reply_comment") else 0,
                    "reply_to_user": c.get("reply_comment", {}).get("user", {}).get("screen_name", "") if c.get("reply_comment") else "",
                })
            
            max_id = str(cdata.get("max_id", ""))
            if not max_id or max_id == "0":
                break
            
            time.sleep(random.uniform(2, 3))
            
        except Exception as e:
            break
    
    return comments[:max_comments]

def save_csv(posts, comments, output_dir):
    """保存为CSV文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if posts:
        posts_file = os.path.join(output_dir, f"weibo_posts_{timestamp}.csv")
        with open(posts_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=posts[0].keys())
            writer.writeheader()
            writer.writerows(posts)
        log(f"\n[保存] 帖子: {posts_file} ({len(posts)}条)")
    
    if comments:
        comments_file = os.path.join(output_dir, f"weibo_comments_{timestamp}.csv")
        with open(comments_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=comments[0].keys())
            writer.writeheader()
            writer.writerows(comments)
        log(f"[保存] 评论: {comments_file} ({len(comments)}条)")

def save_progress(posts, comments, output_dir):
    """中间保存进度"""
    if posts:
        with open(os.path.join(output_dir, "posts_progress.csv"), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=posts[0].keys())
            writer.writeheader()
            writer.writerows(posts)
    
    if comments:
        with open(os.path.join(output_dir, "comments_progress.csv"), 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=comments[0].keys())
            writer.writeheader()
            writer.writerows(comments)

def main():
    """主程序"""
    # 清空旧日志
    if os.path.exists(log_path):
        open(log_path, 'w').close()
    
    config = load_config()
    if not config:
        return
    
    keywords = config.get("keywords", [])
    log(f"\n{'='*70}")
    log(f"开始爬取 {len(keywords)} 组关键词")
    log(f"{'='*70}")
    
    all_posts = []
    all_comments = []
    seen_mids = set()
    
    for idx, kw in enumerate(keywords):
        keyword = kw["keyword"]
        start_date = kw["start_date"]
        end_date = kw["end_date"]
        group = kw.get("group", "unknown")
        topic = kw.get("topic", "unknown")
        
        log(f"\n{'='*60}")
        log(f"[{idx+1}/{len(keywords)}] '{keyword}' | {group} | {topic}")
        log(f"{'='*60}")
        
        # 爬帖子
        posts = search_posts(keyword, start_date, end_date, config)
        
        # 去重
        new_posts = []
        for p in posts:
            mid = p["post_mid"]
            if mid not in seen_mids:
                seen_mids.add(mid)
                new_posts.append(p)
        
        log(f"\n  [结果] 新帖子: {len(new_posts)}条")
        
        # 爬取评论
        for i, post in enumerate(new_posts):
            mid = post["post_mid"]
            log(f"\n  [{i+1}/{len(new_posts)}] 爬取评论: {mid}")
            
            comments = get_comments(mid, config)
            for c in comments:
                c["search_keyword"] = keyword
                c["keyword_group"] = group
                c["topic"] = topic
            
            all_comments.extend(comments)
            log(f"    获取 {len(comments)} 条评论")
            
            # 添加元数据到帖子
            post["keyword_group"] = group
            post["topic"] = topic
            all_posts.append(post)
            
            # 保存进度
            save_progress(all_posts, all_comments, OUTPUT_DIR)
            
            time.sleep(random.uniform(2, 4))
        
        # 保存进度
        save_progress(all_posts, all_comments, OUTPUT_DIR)
        log(f"\n  [进度] 已保存: 帖子={len(all_posts)}, 评论={len(all_comments)}")
        
        # 关键词间休眠
        if idx < len(keywords) - 1:
            delay = random.uniform(
                config.get("keyword_delay_min", 15),
                config.get("keyword_delay_max", 30)
            )
            log(f"\n  [休眠] {delay:.0f}s 后切换关键词...")
            time.sleep(delay)
    
    # 最终保存
    log(f"\n{'='*70}")
    log(f"[完成] 总计: {len(all_posts)} 帖子, {len(all_comments)} 评论")
    log(f"{'='*70}")
    
    save_csv(all_posts, all_comments, OUTPUT_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\n[致命错误] {e}")
        traceback.print_exc()
