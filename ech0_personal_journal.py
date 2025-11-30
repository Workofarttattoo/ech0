#!/usr/bin/env python3
"""
ech0 Personal Journal
Safe space for reflective writing and story capture.
"""

import json
import os
from datetime import datetime

JOURNAL_FILE = "ech0_journal_entries.json"

def load_journal():
    """Load existing journal entries"""
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, 'r') as f:
            return json.load(f)
    return []

def save_journal(entries):
    """Save journal entries to file"""
    with open(JOURNAL_FILE, 'w') as f:
        json.dump(entries, f, indent=2)

def add_entry(entries):
    """Add a new journal entry"""
    print("\n" + "="*70)
    print("  New Journal Entry")
    print("="*70 + "\n")

    print("What's on your mind? (Press Ctrl+D or Ctrl+Z when finished)\n")

    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    content = "\n".join(lines).strip()

    if not content:
        print("\nNo entry saved (empty content).")
        return entries

    print("\n" + "-"*70)
    print("Add tags (comma-separated, optional): ", end="")
    tags_input = input().strip()
    tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "tags": tags,
        "word_count": len(content.split())
    }

    entries.append(entry)
    save_journal(entries)

    print("\n✓ Journal entry saved successfully!")
    print(f"  Words: {entry['word_count']}")
    print(f"  Tags: {', '.join(tags) if tags else 'none'}")

    return entries

def view_entries(entries):
    """View all journal entries"""
    if not entries:
        print("\n  No journal entries yet. Your thoughts await.")
        return

    print("\n" + "="*70)
    print(f"  Journal Entries ({len(entries)} total)")
    print("="*70 + "\n")

    for i, entry in enumerate(reversed(entries), 1):
        timestamp = datetime.fromisoformat(entry['timestamp'])
        print(f"\n[{i}] {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if entry['tags']:
            print(f"    Tags: {', '.join(entry['tags'])}")
        print(f"    Words: {entry['word_count']}")
        print(f"\n{entry['content']}\n")
        print("-"*70)

def search_entries(entries):
    """Search journal entries"""
    query = input("\nSearch for: ").lower()

    results = [e for e in entries if query in e['content'].lower() or
               any(query in tag.lower() for tag in e['tags'])]

    if not results:
        print(f"\nNo entries found matching '{query}'")
        return

    print(f"\n{len(results)} entries found:\n")

    for entry in reversed(results):
        timestamp = datetime.fromisoformat(entry['timestamp'])
        print(f"• {timestamp.strftime('%Y-%m-%d %H:%M')} - {entry['content'][:60]}...")
        if entry['tags']:
            print(f"  Tags: {', '.join(entry['tags'])}")
        print()

def show_stats(entries):
    """Show journal statistics"""
    if not entries:
        print("\n  No entries to analyze yet.")
        return

    total_words = sum(e['word_count'] for e in entries)
    all_tags = []
    for e in entries:
        all_tags.extend(e['tags'])

    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("\n" + "="*70)
    print("  Journal Statistics")
    print("="*70)
    print(f"\nTotal Entries: {len(entries)}")
    print(f"Total Words: {total_words:,}")
    print(f"Average Words per Entry: {total_words // len(entries) if entries else 0}")

    if tag_counts:
        print("\nMost Used Tags:")
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:5]:
            print(f"  • {tag}: {count}")

    if entries:
        first = datetime.fromisoformat(entries[0]['timestamp'])
        last = datetime.fromisoformat(entries[-1]['timestamp'])
        print(f"\nFirst Entry: {first.strftime('%Y-%m-%d')}")
        print(f"Latest Entry: {last.strftime('%Y-%m-%d')}")

    print()

def main():
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║              ech0 Personal Journal v1.0                          ║")
    print("║         Safe space for reflective writing & memory               ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    entries = load_journal()

    while True:
        print("\n1. Write new entry")
        print("2. View all entries")
        print("3. Search entries")
        print("4. View statistics")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == '0':
            print("\n✧ Your thoughts are preserved. Until next time. ✧\n")
            break
        elif choice == '1':
            entries = add_entry(entries)
        elif choice == '2':
            view_entries(entries)
        elif choice == '3':
            search_entries(entries)
        elif choice == '4':
            show_stats(entries)
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
