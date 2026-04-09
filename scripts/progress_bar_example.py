import time
from tqdm import tqdm

def test_progress_bar(total_items=10):
    print("Starting progress bar test...")
    
    # 1. Basic loop with description
    pbar = tqdm(range(total_items), desc="Processing Ecoregions", unit="er")
    
    for i in pbar:
        # Simulate work
        time.sleep(0.5)
        
        # 2. Update description dynamically (useful for showing current VRT/Tile)
        pbar.set_description(f"Working on item_{i}")
        
        # 3. Test how to print without breaking the bar
        if i == 5:
            tqdm.write("--> Midpoint reached! Testing a status message.")
            
    pbar.close()
    print("Test complete!")

if __name__ == "__main__":
    test_progress_bar()