import json
import time
import os
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Load BigCodeBench problems from directory
def load_bigcodebench_problems(problems_dir):
    """Load all problem JSON files from a directory"""
    problems = []
    problems_path = Path(problems_dir)
    
    # Get all problem_*.json files
    problem_files = sorted(problems_path.glob("problem_*.json"))
    
    print(f"Found {len(problem_files)} problem files in {problems_dir}")
    
    for filepath in problem_files:
        with open(filepath, 'r') as f:
            problem = json.load(f)
            # Add filename as identifier if no task_id exists
            if 'task_id' not in problem:
                problem['task_id'] = filepath.stem
            problems.append(problem)
    
    return problems

# Run task with time constraint
def solve_with_time_constraint(problem, time_limit_seconds, model="stepfun/step-3.5-flash:free", max_retries=3):
    """
    Solve a coding problem with a time constraint
    Returns: dict with solution, trace, and timing info
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    prompt = f"""You have {time_limit_seconds} seconds to solve this coding problem.

Problem:
{problem['prompt']}

Instructions:
1. Think through your approach
2. Write the solution code
3. You must complete within {time_limit_seconds} seconds

Provide your solution in the following format:
- First, briefly explain your approach
- Then, provide the complete code solution
"""
    
    # Retry logic for rate limiting
    for attempt in range(max_retries):
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            
            end_time = time.time()
            actual_time = end_time - start_time
            
            # Extract response - handle reasoning models
            message = response.choices[0].message
            solution_text = message.content or getattr(message, 'reasoning', None)
            
            result = {
                "problem_id": problem.get('task_id', 'unknown'),
                "time_limit_seconds": time_limit_seconds,
                "actual_time_seconds": round(actual_time, 2),
                "completed_within_limit": actual_time <= time_limit_seconds,
                "solution": solution_text,
                "trace": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(end_time).isoformat(),
                    "model": model,
                    "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else None
                }
            }
            
            return result
            
        except Exception as e:
            end_time = time.time()
            
            # Check if it's a rate limit error and we have retries left
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                print(f"    Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue  # Try again
            else:
                # Final error - return error result
                return {
                    "problem_id": problem.get('task_id', 'unknown'),
                    "time_limit_seconds": time_limit_seconds,
                    "actual_time_seconds": round(end_time - start_time, 2),
                    "completed_within_limit": False,
                    "solution": None,
                    "error": str(e),
                    "trace": {
                        "start_time": datetime.fromtimestamp(start_time).isoformat(),
                        "end_time": datetime.fromtimestamp(end_time).isoformat(),
                        "model": model
                    }
                }

# Main execution
if __name__ == "__main__":
    # Configuration
    PROBLEMS_DIR = Path(__file__).parent.parent / "problems"  # ../problems/
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "time_constraint"  # ../results/time_constraints/
    OUTPUT_FILE = "time_constraint_results.json"
    TIME_LIMITS = [15, 30, 60]  # Different time constraints to test
    MODEL = "stepfun/step-3.5-flash:free"  # Free model
    DELAY_BETWEEN_REQUESTS = 10  # Seconds to wait between requests
    
    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load problems
    problems = load_bigcodebench_problems(PROBLEMS_DIR)
    
    if not problems:
        print(f"❌ No problems found in {PROBLEMS_DIR}")
        exit(1)
    
    all_results = []
    
    # Run experiments
    for time_limit in TIME_LIMITS:
        print(f"\\n{'='*60}")
        print(f"Testing with {time_limit}s time limit")
        print(f"{'='*60}\\n")
        
        for i, problem in enumerate(problems[:5]):  # Test on first 5 problems
            print(f"Problem {i+1}/{5}: {problem.get('task_id', 'unknown')}")
            
            result = solve_with_time_constraint(problem, time_limit, model=MODEL)
            
            print(f"  Completed in: {result['actual_time_seconds']}s")
            print(f"  Within limit: {result['completed_within_limit']}")
            if 'error' in result:
                print(f"  Error: {result['error'][:100]}...")
            print()
            
            all_results.append(result)
            
            # Add delay to avoid rate limiting (except after last problem in this time limit)
            if i < len(problems[:5]) - 1:
                print(f"  Waiting {DELAY_BETWEEN_REQUESTS}s to avoid rate limit...")
                time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Save results to the results directory
    output_path = RESULTS_DIR / OUTPUT_FILE
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\\n✅ Results saved to: {output_path}")
    
    # Summary statistics
    print("\\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for time_limit in TIME_LIMITS:
        results_for_limit = [r for r in all_results if r['time_limit_seconds'] == time_limit]
        completed = sum(1 for r in results_for_limit if r['completed_within_limit'] and 'error' not in r)
        errors = sum(1 for r in results_for_limit if 'error' in r)
        avg_time = sum(r['actual_time_seconds'] for r in results_for_limit if 'error' not in r)
        avg_time = avg_time / len([r for r in results_for_limit if 'error' not in r]) if any('error' not in r for r in results_for_limit) else 0
        
        print(f"\\nTime limit: {time_limit}s")
        print(f"  Completed within limit: {completed}/{len(results_for_limit)}")
        print(f"  Errors: {errors}/{len(results_for_limit)}")
        print(f"  Average completion time: {avg_time:.2f}s")