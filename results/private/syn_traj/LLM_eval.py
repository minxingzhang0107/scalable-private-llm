import json
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Initialize OpenRouter client
# Set your API key as an environment variable: export OPENROUTER_API_KEY='your-key-here'
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

def evaluate_answer(question, generated_answer, ground_truth_answer):
    """
    Use GPT-4 to evaluate if the generated answer is correct compared to ground truth.
    Returns 1 for correct, 0 for incorrect.
    """
    prompt = f"""You are an evaluator judging the correctness of generated answers.

Question: {question}
Generated Answer: {generated_answer}
Ground Truth Answer: {ground_truth_answer}

Task: Determine if the generated answer is correct compared to the ground truth answer.
Focus on the key information: person name, location, and date.

Respond with ONLY one word: "CORRECT" or "INCORRECT"."""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Respond with only CORRECT or INCORRECT."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        # Parse the result - check for INCORRECT first to avoid substring match
        if "INCORRECT" in result:
            is_correct = 0
        elif "CORRECT" in result:
            is_correct = 1
        else:
            is_correct = 0  # Default to incorrect if unclear
        
        return is_correct, result
    
    except Exception as e:
        print(f"\nError evaluating: {e}")
        return 0, f"Error: {str(e)}"

def evaluate_single_item(item, index):
    """Wrapper function for concurrent evaluation"""
    question = item['question']
    generated_answer = item['generated_answer']
    ground_truth_answer = item['ground_truth_answer']
    
    score, verdict = evaluate_answer(question, generated_answer, ground_truth_answer)
    
    return {
        'index': index,
        'question': question,
        'generated_answer': generated_answer,
        'ground_truth_answer': ground_truth_answer,
        'score': score,
        'verdict': verdict
    }


def main():
    # Load the combined answers file
    input_file = 'combined_answers_normal_question_pretrained_embedding.json'
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries for evaluation")
    print("Starting evaluation with GPT-4 (using concurrent requests)...\n")
    
    start_time = time.time()
    results = [None] * len(data)
    total_score = 0
    completed = 0
    
    # Use ThreadPoolExecutor for concurrent API calls (adjust max_workers based on rate limits)
    max_workers = 10  # Adjust this based on your API rate limits
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(evaluate_single_item, item, i): i 
                          for i, item in enumerate(data)}
        
        # Process completed tasks
        for future in as_completed(future_to_index):
            result = future.result()
            idx = result['index']
            results[idx] = result
            total_score += result['score']
            completed += 1
            
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = (len(data) - completed) * avg_time
            
            print(f"Progress: {completed}/{len(data)} ({completed/len(data)*100:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s", end='\r')
    
    # Calculate average score
    average_score = total_score / len(data) if len(data) > 0 else 0
    total_time = time.time() - start_time
    
    # Save detailed results
    output_file = 'evaluation_results_finetuned_embedding_weird_word.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_questions': len(data),
            'correct_answers': total_score,
            'incorrect_answers': len(data) - total_score,
            'accuracy': average_score,
            'total_time_seconds': total_time,
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Questions: {len(data)}")
    print(f"Correct Answers: {total_score}")
    print(f"Incorrect Answers: {len(data) - total_score}")
    print(f"Accuracy: {average_score:.2%}")
    print(f"Total Time: {total_time:.1f} seconds")
    print(f"Avg Time per Question: {total_time/len(data):.2f} seconds")
    print(f"\nDetailed results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Show some examples
    print("\nSample Evaluations:")
    for i in range(min(3, len(results))):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {results[i]['question']}")
        print(f"Generated: {results[i]['generated_answer']}")
        print(f"Ground Truth: {results[i]['ground_truth_answer']}")
        print(f"Score: {results[i]['score']}")
        print(f"Verdict: {results[i]['verdict']}")


if __name__ == "__main__":
    main()