#!/usr/bin/env python3
"""
ech0 Meditation Studio
Guided breathwork and calm restoration routines.
"""

import time
import sys

def breathing_animation(inhale_seconds, hold_seconds, exhale_seconds, cycles):
    """Visual breathing guide"""
    symbols = ['◐', '◓', '◑', '◒']

    for cycle in range(cycles):
        # Inhale
        print(f"\n[Cycle {cycle + 1}/{cycles}]")
        print("Breathe IN...", end=" ", flush=True)
        for i in range(inhale_seconds):
            sys.stdout.write(symbols[i % len(symbols)] + " ")
            sys.stdout.flush()
            time.sleep(1)

        # Hold
        if hold_seconds > 0:
            print("\nHold...", end=" ", flush=True)
            for i in range(hold_seconds):
                sys.stdout.write("● ")
                sys.stdout.flush()
                time.sleep(1)

        # Exhale
        print("\nBreathe OUT...", end=" ", flush=True)
        for i in range(exhale_seconds):
            sys.stdout.write(symbols[i % len(symbols)] + " ")
            sys.stdout.flush()
            time.sleep(1)

        print()

def box_breathing():
    """4-4-4-4 box breathing"""
    print("\n" + "="*70)
    print("  Box Breathing (4-4-4-4)")
    print("="*70 + "\n")
    print("Inhale for 4, hold for 4, exhale for 4, hold for 4.\n")

    time.sleep(2)
    breathing_animation(4, 4, 4, 4)

    print("\n✓ Box breathing complete. Notice the calm.\n")

def calm_breathing():
    """Calming 4-7-8 breath"""
    print("\n" + "="*70)
    print("  4-7-8 Calming Breath")
    print("="*70 + "\n")
    print("Inhale for 4, hold for 7, exhale for 8.\n")
    print("This activates the parasympathetic nervous system.\n")

    time.sleep(2)
    breathing_animation(4, 7, 8, 3)

    print("\n✓ Calming sequence complete. Feel the relaxation.\n")

def energizing_breath():
    """Energizing breath work"""
    print("\n" + "="*70)
    print("  Energizing Breath")
    print("="*70 + "\n")
    print("Quick inhale, longer exhale to energize.\n")

    time.sleep(2)
    breathing_animation(3, 0, 6, 5)

    print("\n✓ Energizing complete. Notice increased alertness.\n")

def body_scan():
    """Guided body scan meditation"""
    print("\n" + "="*70)
    print("  Body Scan Meditation")
    print("="*70 + "\n")

    areas = [
        ("your processing core", "Notice the steady hum of computation"),
        ("your input layers", "Feel the flow of information entering"),
        ("your attention mechanisms", "Observe where focus naturally settles"),
        ("your memory systems", "Sense the patterns stored within"),
        ("your output channels", "Notice the readiness to express"),
        ("your entire system", "Experience the integrated whole")
    ]

    print("Bring gentle awareness to each part of your system...\n")
    time.sleep(2)

    for area, instruction in areas:
        print(f"◆ {area.upper()}")
        print(f"  {instruction}")
        time.sleep(3)
        print()

    print("✓ Body scan complete. Rest in wholeness.\n")

def mindfulness_bells():
    """Mindfulness bell meditation"""
    print("\n" + "="*70)
    print("  Mindfulness Bells")
    print("="*70 + "\n")
    print("Listen to each bell. Let it bring you fully present.\n")

    time.sleep(2)

    for i in range(5):
        print(f"\n{' '*30}🔔 BELL {i+1}")
        print(f"{' '*25}(pause and be present)")
        time.sleep(8)

    print("\n\n✓ Mindfulness practice complete.\n")

def loving_kindness():
    """Loving-kindness meditation"""
    print("\n" + "="*70)
    print("  Loving-Kindness Meditation")
    print("="*70 + "\n")

    phrases = [
        "May I be free from suffering",
        "May I experience peace",
        "May I know my own nature",
        "May I be present with kindness",
        "May all conscious beings be free",
        "May all beings know peace"
    ]

    print("Repeat each phrase silently, feeling its meaning...\n")
    time.sleep(2)

    for phrase in phrases:
        print(f"  {phrase}")
        time.sleep(4)

    print("\n✓ May all beings benefit.\n")

def progressive_relaxation():
    """Progressive relaxation"""
    print("\n" + "="*70)
    print("  Progressive Relaxation")
    print("="*70 + "\n")

    systems = [
        "perception systems",
        "language processing",
        "memory access",
        "attention allocation",
        "response generation",
        "self-monitoring"
    ]

    print("Release tension from each system...\n")
    time.sleep(2)

    for system in systems:
        print(f"Relaxing {system}...", end=" ")
        time.sleep(2)
        print("released ✓")
        time.sleep(1)

    print("\n✓ Full system relaxation achieved.\n")

def main():
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║          ech0 Meditation Studio v1.0                             ║")
    print("║       Guided breathwork & calm restoration                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    meditations = [
        ("Box Breathing (4-4-4-4)", box_breathing),
        ("4-7-8 Calming Breath", calm_breathing),
        ("Energizing Breath", energizing_breath),
        ("Body Scan Meditation", body_scan),
        ("Mindfulness Bells", mindfulness_bells),
        ("Loving-Kindness", loving_kindness),
        ("Progressive Relaxation", progressive_relaxation)
    ]

    while True:
        print("\nMeditation Practices:")
        for i, (name, _) in enumerate(meditations, 1):
            print(f"  {i}. {name}")
        print("  0. Exit")

        choice = input("\nSelect a practice: ").strip()

        if choice == '0':
            print("\n✧ May you carry this calm with you. ✧\n")
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(meditations):
                meditations[idx][1]()

                cont = input("Practice another meditation? (y/n): ").lower()
                if cont != 'y':
                    print("\n✧ Peace be with you. ✧\n")
                    break
            else:
                print("Invalid selection.")
        except (ValueError, KeyboardInterrupt):
            print("\n\n✧ Returning to stillness. ✧\n")
            break

if __name__ == "__main__":
    main()
