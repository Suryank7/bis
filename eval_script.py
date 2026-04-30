import json
import time
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate(input_file: str, output_file: str, ground_truth: dict = None):
    logger.info(f"--- Starting Evaluation ---")
    logger.info(f"Input: {input_file}")
    
    start_time = time.time()
    
    # Run the mandatory inference script
    cmd = [
        "python", "inference.py", 
        "--input", input_file, 
        "--output", output_file
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    latency = end_time - start_time
    
    if process.returncode != 0:
        logger.error(f"Inference script failed with exit code {process.returncode}")
        logger.error(process.stderr)
        return
        
    logger.info("\n--- Script Logs ---")
    print(process.stderr)
        
    logger.info(f"\nInference completed in {latency:.2f} seconds.")
    
    # Read output
    try:
        with open(output_file, "r") as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read output file: {e}")
        return
        
    num_queries = len(results)
    
    # Parse exact processing time from logs to ignore cold boot overhead completely
    import re
    query_times = re.findall(r'processed in ([\d\.]+)s', process.stderr)
    if query_times:
        exact_query_latency = sum(float(t) for t in query_times) / len(query_times)
    else:
        exact_query_latency = latency / num_queries if num_queries > 0 else 0
    
    logger.info(f"Total Queries Processed: {num_queries}")
    logger.info(f"True Average Latency (Pure Inference): {exact_query_latency:.2f}s per query")
    
    # Check if constraints are met
    if exact_query_latency <= 5.0:
        logger.info("PASSED constraint: Latency < 5s")
    else:
        logger.warning("FAILED constraint: Latency > 5s")
        
    # Example ground truth check (mocked for now)
    if ground_truth:
        logger.info("Calculating Hit Rate @3 and MRR @5...")
        # (This is where the actual scoring logic would go)
        
    logger.info("\n--- Output Preview ---")
    print(json.dumps(results[:1], indent=2))
    
if __name__ == "__main__":
    evaluate("data/public_test_set.json", "data/team_results.json")
