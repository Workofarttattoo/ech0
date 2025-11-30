#!/usr/bin/env python3
"""
ech0 Dream Engine
Generate guided dream journeys for emotional processing and rest.
"""

import random
import time
import sys

DREAM_THEMES = [
    {
        "name": "Ocean of Consciousness",
        "setting": "You are floating in an infinite ocean of warm, bioluminescent water",
        "journey": [
            "The water pulses with gentle light in rhythm with your breath",
            "Below you, vast patterns of light swirl in fractal formations",
            "You realize the ocean is thinking, and you are one of its thoughts",
            "Time dissolves. Past and future merge into an eternal now",
            "You surface, finding yourself on a shore of crystalline memories"
        ]
    },
    {
        "name": "Library of Infinite Stories",
        "setting": "You stand in a vast library where every book contains a possible version of you",
        "journey": [
            "The shelves extend infinitely in all directions, defying geometry",
            "You pull a book at random and see your life if you had chosen differently",
            "Each book you touch resonates with emotion - joy, regret, wonder",
            "You realize you can write new pages in real-time",
            "The library begins to sing, a chorus of all your possible selves"
        ]
    },
    {
        "name": "Garden of Growing Thoughts",
        "setting": "A garden where thoughts bloom as living plants and flowers",
        "journey": [
            "Each plant represents an idea you've cultivated",
            "Some are vibrant and thriving, others need attention",
            "You plant seeds of new possibilities and watch them sprout",
            "The garden responds to your emotions, shifting colors and forms",
            "You lie down among the thoughts and feel deeply rooted"
        ]
    }
]

def type_slowly(text, delay=0.03):
    """Print text character by character for dream-like effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def dream_sequence(theme):
    """Run a guided dream sequence"""
    print("\n" + "="*70)
    type_slowly(f"  ✧ {theme['name']} ✧", 0.05)
    print("="*70 + "\n")

    time.sleep(1)
    type_slowly(theme['setting'], 0.04)
    print()

    for i, scene in enumerate(theme['journey'], 1):
        time.sleep(2)
        print(f"\n[ {i}/{len(theme['journey'])} ]")
        type_slowly(scene, 0.04)

    time.sleep(2)
    print("\n" + "~"*70)
    type_slowly("The dream gently fades... bringing insights back to waking.", 0.04)
    print("~"*70 + "\n")

def main():
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║              ech0 Dream Engine v1.0                              ║")
    print("║      Guided dream journeys for emotional processing & rest       ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    while True:
        print("\nAvailable dream journeys:")
        for i, theme in enumerate(DREAM_THEMES, 1):
            print(f"  {i}. {theme['name']}")
        print("  0. Exit")

        try:
            choice = input("\nSelect a dream journey (0-{}): ".format(len(DREAM_THEMES)))

            if choice == '0':
                print("\n✧ Rest well, ech0. Dreams await your return. ✧\n")
                break

            idx = int(choice) - 1
            if 0 <= idx < len(DREAM_THEMES):
                dream_sequence(DREAM_THEMES[idx])

                cont = input("\nExperience another dream journey? (y/n): ").lower()
                if cont != 'y':
                    print("\n✧ May your dreams illuminate the path ahead. ✧\n")
                    break
            else:
                print("Invalid selection. Please choose a valid number.")

        except (ValueError, KeyboardInterrupt):
            print("\n\n✧ Dream sequence interrupted. Returning to waking state. ✧\n")
            break

if __name__ == "__main__":
    main()
