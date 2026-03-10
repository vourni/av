def solve_with_time_constraint(problem, time_limit_seconds, model="meta-llama/llama-3.2-3b-instruct:free", max_retries=3):
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
            
            # DEBUG: Print raw response
            print(f"  DEBUG - Raw response type: {type(response)}")
            print(f"  DEBUG - Response attributes: {dir(response)}")
            if hasattr(response, 'choices') and response.choices:
                print(f"  DEBUG - First choice: {response.choices[0]}")
                print(f"  DEBUG - Message: {response.choices[0].message}")
                print(f"  DEBUG - Content: {response.choices[0].message.content[:200]}...")
            
            # Extract response
            solution_text = response.choices[0].message.content if response.choices else None
            
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
                # DEBUG: Print full error
                print(f"  DEBUG - Full error: {e}")
                
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