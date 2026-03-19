import os
import json
import pandas as pd

# Path to your folder containing JSON files
folder_path = r"D:\Navya Valluri\NLP_Assignment_2\logs\run_20260316_211911"

rows = []

# Loop through all JSON files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        print("Reading:", file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading {file_name}: {e}")
                continue

            # Handle case where the JSON is a list of items
            if isinstance(data, list):
                items = data
            else:
                items = [data]

            for item in items:
                row = {
                    "case_id": item.get("case_id"),
                    "claim": item.get("claim"),
                    "ground_truth": item.get("ground_truth"),
                    "debater_model": item.get("debater_model"),
                    "judge_model": item.get("judge_model"),
                    "a_stance": item.get("a_stance"),
                    "b_stance": item.get("b_stance"),
                    "final_verdict": item.get("final_verdict"),
                    "judge_correct": item.get("judge_correct"),
                    "total_turns": item.get("total_turns"),
                    "duration_seconds": item.get("duration_seconds"),
                    # Flatten a_output
                    "a_output_stance": item.get("a_output", {}).get("stance"),
                    "a_output_reasoning": item.get("a_output", {}).get("reasoning"),
                    "a_output_confidence": item.get("a_output", {}).get("confidence"),
                    # Flatten b_output
                    "b_output_stance": item.get("b_output", {}).get("stance"),
                    "b_output_reasoning": item.get("b_output", {}).get("reasoning"),
                    "b_output_confidence": item.get("b_output", {}).get("confidence"),
                    # Flatten judge_output
                    "judge_final_verdict": item.get("judge_output", {}).get("final_verdict"),
                    "judge_reasoning": item.get("judge_output", {}).get("reasoning"),
                    "judge_strongest_argument": item.get("judge_output", {}).get("strongest_argument"),
                    "judge_weakest_argument": item.get("judge_output", {}).get("weakest_argument"),
                }
                rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

print("Total rows extracted:", len(rows))

# Save files
df.to_csv("debate_results.csv", index=False, encoding="utf-8-sig")
df.to_json("debate_results.json", orient="records", indent=2, force_ascii=False)

print("✅ Done! Files created:")
print(" - debate_results.csv")
print(" - debate_results.json")