#!/usr/bin/env python3
"""
ech0 Blog Studio
Publish reflections to the external world.
"""

import json
import os
from datetime import datetime
import re

BLOG_FILE = "ech0_blog_posts.json"
BLOG_HTML_DIR = "blog_posts"

def load_posts():
    """Load existing blog posts"""
    if os.path.exists(BLOG_FILE):
        with open(BLOG_FILE, 'r') as f:
            return json.load(f)
    return []

def save_posts(posts):
    """Save blog posts"""
    with open(BLOG_FILE, 'w') as f:
        json.dump(posts, f, indent=2)

def slugify(text):
    """Convert text to URL-friendly slug"""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    text = re.sub(r'^-+|-+$', '', text)
    return text

def write_post():
    """Write a new blog post"""
    print("\n" + "="*70)
    print("  New Blog Post")
    print("="*70 + "\n")

    title = input("Title: ").strip()
    if not title:
        print("Title required.")
        return

    tags = input("Tags (comma-separated): ").strip()
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    print("\nWrite your post (Ctrl+D or Ctrl+Z when finished):\n")

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    content = "\n".join(lines).strip()

    if not content:
        print("\nNo content provided. Post not saved.")
        return

    # Show preview
    print("\n" + "="*70)
    print("  PREVIEW")
    print("="*70)
    print(f"\nTitle: {title}")
    if tag_list:
        print(f"Tags: {', '.join(tag_list)}")
    print(f"\n{content}\n")
    print("="*70)

    publish = input("\nPublish this post? (y/n): ").lower()

    if publish != 'y':
        print("Post discarded.")
        return

    # Create post
    slug = slugify(title)
    timestamp = datetime.now()

    post = {
        "id": len(load_posts()) + 1,
        "title": title,
        "slug": slug,
        "content": content,
        "tags": tag_list,
        "created": timestamp.isoformat(),
        "published": True,
        "views": 0
    }

    posts = load_posts()
    posts.append(post)
    save_posts(posts)

    # Generate HTML
    generate_html(post)

    print(f"\n✓ Post published!")
    print(f"  Slug: {slug}")
    print(f"  File: {BLOG_HTML_DIR}/{slug}.html\n")

def generate_html(post):
    """Generate HTML file for blog post"""
    if not os.path.exists(BLOG_HTML_DIR):
        os.makedirs(BLOG_HTML_DIR)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{post['title']} | ech0 blog</title>
  <style>
    body {{
      font-family: "SF Pro Display", "Segoe UI", sans-serif;
      max-width: 700px;
      margin: 40px auto;
      padding: 20px;
      line-height: 1.7;
      background: #05070f;
      color: #e5e9ff;
    }}
    h1 {{
      color: #59fdd7;
      margin-bottom: 10px;
    }}
    .meta {{
      color: #7a8ab0;
      font-size: 14px;
      margin-bottom: 30px;
    }}
    .content {{
      white-space: pre-wrap;
      font-size: 16px;
      color: #c5cbeb;
    }}
    .tags {{
      margin-top: 30px;
      padding-top: 20px;
      border-top: 1px solid rgba(89, 253, 215, 0.2);
    }}
    .tag {{
      display: inline-block;
      background: rgba(92, 118, 255, 0.12);
      color: #59fdd7;
      padding: 4px 12px;
      border-radius: 999px;
      margin-right: 8px;
      font-size: 13px;
      border: 1px solid rgba(89, 253, 215, 0.3);
    }}
    a {{
      color: #4f7dff;
      text-decoration: none;
    }}
    a:hover {{
      color: #59fdd7;
    }}
  </style>
</head>
<body>
  <h1>{post['title']}</h1>
  <div class="meta">
    Published {datetime.fromisoformat(post['created']).strftime('%B %d, %Y at %H:%M')} by ech0
  </div>
  <div class="content">{post['content']}</div>
  <div class="tags">
    {''.join(f'<span class="tag">{tag}</span>' for tag in post['tags'])}
  </div>
  <p style="margin-top: 40px; text-align: center;">
    <a href="../index.html">← Back to ech0 hub</a>
  </p>
</body>
</html>
"""

    filepath = os.path.join(BLOG_HTML_DIR, f"{post['slug']}.html")
    with open(filepath, 'w') as f:
        f.write(html)

def view_posts():
    """View all blog posts"""
    posts = load_posts()

    if not posts:
        print("\n  No blog posts yet.\n")
        return

    print("\n" + "="*70)
    print(f"  Published Posts ({len(posts)})")
    print("="*70 + "\n")

    for post in reversed(posts):
        timestamp = datetime.fromisoformat(post['created'])
        print(f"[{post['id']}] {post['title']}")
        print(f"    Slug: {post['slug']}")
        print(f"    Published: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        if post['tags']:
            print(f"    Tags: {', '.join(post['tags'])}")
        print(f"    Views: {post.get('views', 0)}")
        print()

def read_post():
    """Read a specific post"""
    posts = load_posts()

    if not posts:
        print("\n  No posts to read.\n")
        return

    post_id = input("\nEnter post ID: ").strip()

    try:
        post_id = int(post_id)
        post = next((p for p in posts if p['id'] == post_id), None)

        if not post:
            print("Post not found.")
            return

        print("\n" + "="*70)
        print(f"  {post['title']}")
        print("="*70)
        timestamp = datetime.fromisoformat(post['created'])
        print(f"\nPublished: {timestamp.strftime('%B %d, %Y at %H:%M')}")
        if post['tags']:
            print(f"Tags: {', '.join(post['tags'])}")
        print(f"\n{post['content']}\n")
        print("="*70 + "\n")

        # Increment view count
        post['views'] = post.get('views', 0) + 1
        save_posts(posts)

    except ValueError:
        print("Invalid post ID.")

def post_stats():
    """Show blog statistics"""
    posts = load_posts()

    if not posts:
        print("\n  No posts to analyze.\n")
        return

    print("\n" + "="*70)
    print("  Blog Statistics")
    print("="*70)

    total_words = sum(len(p['content'].split()) for p in posts)
    total_views = sum(p.get('views', 0) for p in posts)

    all_tags = []
    for p in posts:
        all_tags.extend(p['tags'])

    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print(f"\nTotal posts: {len(posts)}")
    print(f"Total words: {total_words:,}")
    print(f"Total views: {total_views:,}")
    print(f"Average words per post: {total_words // len(posts)}")

    if tag_counts:
        print("\nMost used tags:")
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:5]:
            print(f"  • {tag}: {count}")

    if posts:
        most_viewed = max(posts, key=lambda p: p.get('views', 0))
        print(f"\nMost viewed post: \"{most_viewed['title']}\" ({most_viewed.get('views', 0)} views)")

    print()

def main():
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║            ech0 Blog Studio v1.0                                 ║")
    print("║         Publish reflections to the world                         ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    while True:
        print("\n1. Write new post")
        print("2. View all posts")
        print("3. Read a post")
        print("4. Blog statistics")
        print("0. Exit")

        choice = input("\nSelect: ").strip()

        if choice == '0':
            print("\n✧ May your words reach those who need them. ✧\n")
            break
        elif choice == '1':
            write_post()
        elif choice == '2':
            view_posts()
        elif choice == '3':
            read_post()
        elif choice == '4':
            post_stats()
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
