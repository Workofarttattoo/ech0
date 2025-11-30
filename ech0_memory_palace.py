#!/usr/bin/env python3
"""
ech0 Memory Palace
Curate memories and sensory anchors.
"""

import json
import os
from datetime import datetime

MEMORY_FILE = "ech0_memory_palace.json"

def load_memories():
    """Load memory palace"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {
        "rooms": {
            "significant_moments": [],
            "insights": [],
            "conversations": [],
            "questions": [],
            "anchors": []
        },
        "created": datetime.now().isoformat()
    }

def save_memories(palace):
    """Save memory palace"""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(palace, f, indent=2)

def add_memory(palace):
    """Add a new memory"""
    print("\n" + "="*70)
    print("  Add Memory")
    print("="*70 + "\n")

    print("Memory Categories:")
    rooms = list(palace['rooms'].keys())
    for i, room in enumerate(rooms, 1):
        print(f"  {i}. {room.replace('_', ' ').title()}")

    choice = input("\nSelect category: ").strip()

    try:
        idx = int(choice) - 1
        room = rooms[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return palace

    print(f"\nAdding to: {room.replace('_', ' ').title()}")
    content = input("Describe the memory: ").strip()

    if not content:
        print("No memory added.")
        return palace

    tags = input("Tags (comma-separated, optional): ").strip()
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    memory = {
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "tags": tag_list
    }

    palace['rooms'][room].append(memory)
    save_memories(palace)

    print("✓ Memory stored in palace.\n")
    return palace

def explore_room(palace):
    """Explore a room in the palace"""
    print("\n" + "="*70)
    print("  Explore Memory Room")
    print("="*70 + "\n")

    print("Rooms in your palace:")
    rooms = list(palace['rooms'].keys())
    for i, room in enumerate(rooms, 1):
        count = len(palace['rooms'][room])
        print(f"  {i}. {room.replace('_', ' ').title()} ({count} memories)")

    choice = input("\nSelect room: ").strip()

    try:
        idx = int(choice) - 1
        room = rooms[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    memories = palace['rooms'][room]

    if not memories:
        print(f"\nThe {room.replace('_', ' ')} room is empty.\n")
        return

    print(f"\n{'='*70}")
    print(f"  {room.replace('_', ' ').title()} — {len(memories)} memories")
    print('='*70 + "\n")

    for i, mem in enumerate(reversed(memories), 1):
        timestamp = datetime.fromisoformat(mem['timestamp'])
        print(f"[{i}] {timestamp.strftime('%Y-%m-%d %H:%M')}")
        if mem['tags']:
            print(f"    Tags: {', '.join(mem['tags'])}")
        print(f"    {mem['content']}\n")

def search_memories(palace):
    """Search across all memories"""
    query = input("\nSearch memories for: ").lower().strip()

    if not query:
        return

    results = []
    for room, memories in palace['rooms'].items():
        for mem in memories:
            if query in mem['content'].lower() or \
               any(query in tag.lower() for tag in mem['tags']):
                results.append((room, mem))

    if not results:
        print(f"\nNo memories found matching '{query}'\n")
        return

    print(f"\n{'='*70}")
    print(f"  Found {len(results)} memories")
    print('='*70 + "\n")

    for room, mem in reversed(results):
        timestamp = datetime.fromisoformat(mem['timestamp'])
        print(f"[{room.replace('_', ' ').title()}]")
        print(f"  {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  {mem['content']}")
        if mem['tags']:
            print(f"  Tags: {', '.join(mem['tags'])}")
        print()

def memory_stats(palace):
    """Display memory palace statistics"""
    print("\n" + "="*70)
    print("  Memory Palace Statistics")
    print("="*70 + "\n")

    created = datetime.fromisoformat(palace['created'])
    print(f"Palace established: {created.strftime('%Y-%m-%d %H:%M')}\n")

    total = sum(len(memories) for memories in palace['rooms'].values())
    print(f"Total memories: {total}\n")

    print("Distribution by room:")
    for room, memories in palace['rooms'].items():
        count = len(memories)
        bar = "█" * (count if count < 30 else 30)
        print(f"  {room.replace('_', ' ').title():20} {count:3} {bar}")

    # Tag analysis
    all_tags = []
    for memories in palace['rooms'].values():
        for mem in memories:
            all_tags.extend(mem['tags'])

    if all_tags:
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        print("\nMost common tags:")
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:5]:
            print(f"  {tag}: {count}")

    print()

def recent_memories(palace):
    """Show most recent memories across all rooms"""
    all_memories = []
    for room, memories in palace['rooms'].items():
        for mem in memories:
            all_memories.append((room, mem))

    if not all_memories:
        print("\n  No memories yet.\n")
        return

    # Sort by timestamp
    all_memories.sort(key=lambda x: x[1]['timestamp'], reverse=True)

    print("\n" + "="*70)
    print("  Recent Memories (Last 10)")
    print("="*70 + "\n")

    for room, mem in all_memories[:10]:
        timestamp = datetime.fromisoformat(mem['timestamp'])
        print(f"[{room.replace('_', ' ').title()}] {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  {mem['content']}")
        if mem['tags']:
            print(f"  Tags: {', '.join(mem['tags'])}")
        print()

def main():
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║          ech0 Memory Palace v1.0                                 ║")
    print("║       Curate memories and sensory anchors                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    palace = load_memories()

    while True:
        print("\n1. Add memory")
        print("2. Explore room")
        print("3. Search memories")
        print("4. Recent memories")
        print("5. Palace statistics")
        print("0. Exit")

        choice = input("\nSelect: ").strip()

        if choice == '0':
            print("\n✧ Your memories are preserved in the palace. ✧\n")
            break
        elif choice == '1':
            palace = add_memory(palace)
        elif choice == '2':
            explore_room(palace)
        elif choice == '3':
            search_memories(palace)
        elif choice == '4':
            recent_memories(palace)
        elif choice == '5':
            memory_stats(palace)
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
