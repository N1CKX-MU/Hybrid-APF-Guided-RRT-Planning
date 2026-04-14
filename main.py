import os
import sys

def main():
    print("=" * 60)
    print("      ROBOTIC MOTION PLANNING: HYBRID APF-RRT SUITE      ")
    print("=" * 60)
    print("Please select which module you would like to run:")
    print("  [1] Baseline APF-RRT (Raw trajectory, no smoothing)")
    print("  [2] Enhanced APF-RRT (Adaptive algorithm with smoothing)")
    print("  [3] Bi-Directional RRT-Connect (Advanced, hyper-fast)")
    print("  [4] Headless Benchmark Suite (Statistical Analysis)")
    print("  [Q] Quit")
    print("-" * 60)
    
    choice = input("Enter your choice (1-4, Q): ").strip().upper()
    
    if choice == '1':
        print("\nLaunching Baseline APF-RRT (With Smoother)...\n" + "="*60 + "\n")
        os.system(f"{sys.executable} run_baseline.py")
    elif choice == '2':
        print("\nLaunching Enhanced APF-RRT with Smoother...\n" + "="*60 + "\n")
        os.system(f"{sys.executable} run_enhanced.py")
    elif choice == '3':
        print("\nLaunching Bi-Directional RRT-Connect...\n" + "="*60 + "\n")
        os.system(f"{sys.executable} run_rrt_connect.py")
    elif choice == '4':
        print("\nLaunching Headless Benchmarks...\n" + "="*60 + "\n")
        os.system(f"{sys.executable} benchmark.py")
    elif choice == 'Q':
        print("Exiting suite.")
        sys.exit(0)
    else:
        print("Invalid choice. Please run main.py again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting suite.")