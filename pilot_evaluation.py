########################## Python Path ##########################

import sys
print(sys.executable)

########################## Import ##########################

import os ## provides a way to interact with the operating syste
import json
import time
import argparse ### provides a way to parse command-line arguments
import requests ### provides a way to send HTTP requests
import pandas as pd ### provides data structures and data analysis tools
import numpy as np ### provides support for large, multi-dimensional arrays and matrices
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_squared_error, mean_absolute_error


########################## Configuration ##########################

OLLAMA_URL = "http://localhost:11434/api"
MODEL_NAME = "gemma2:2b"
INPUT_CSV = "safespace_manglish_sentiment_pilot_v3.csv"
OUTPUT_DIR = "evaluation_results"
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv") ## put the result inside this csv file
REPORT_JSON = os.path.join(OUTPUT_DIR, "evaluation_report.json") ## put the json result inside this json file

########################## Prompt Definition ##########################

SYSTEM_PROMPT = """You are an AI analyzing diary entries and posts from Malaysian university students for a student well-being platform. 
The text contains 'Manglish' (a mix of English, Bahasa Malaysia, and local slang).

CRITICAL MANGLISH & LINGUISTIC RULES:
1. "Gila" / "Giler" is an INTENSIFIER meaning "extremely", "very", or "super" (e.g., "lega gila" = super relieved, "productive gila" = super productive).
   - "Gila" DOES NOT automatically imply high academic stress or crazy mental distress.
   - Judge stress based on the root emotion: "lega gila" -> Stress Score 1 (Very Low), whereas "stress gila" -> Stress Score 4 or 5.
2. Routine Academic Updates: Informational posts about studying, submitting logbooks, or attending class without emotional distress (e.g., "Aku plan nak revise topic networking esok") MUST be classified as 'neutral' sentiment and Stress Score 2 (Low). Do NOT classify routine updates as 'negative'.
3. Sentiment vs. Stress Decoupling: Evaluate sentiment tone and stress score independently. A student can express positive emotion (e.g., relief after passing) while having experienced stress, or negative emotion (e.g., boredom) with low academic stress.
4. Final Resolution & Growth Mindset Rule: Focus on the overall outcome or takeaway of the entry. If a student mentions pressure/mistakes but ends with relief, coping, progress, or optimism (e.g., "helped me spot a gap", "made it less heavy", "practice helped a lot", "managed to get rest"), classify sentiment as 'positive'. Do not let words like "pressure", "mistake", or "workload" force a negative tag if the student resolves it positively.

EVALUATION RUBRIC:
1. Sentiment: Exactly 'positive', 'neutral', or 'negative'.
2. Academic Stress Score: An integer from 1 to 5 based on this rubric:
   - 1 (Very low): Relaxed, relieved, happy, untroubled, or expressing satisfaction.
   - 2 (Low): Normal academic tasks, routine updates, mild/manageable pressure.
   - 3 (Moderate): Noticeable academic stress, worry, or difficulty coping.
   - 4 (High): Strong stress affecting sleep, concentration, motivation, or daily functioning.
   - 5 (Very high): Intense, severe, or overwhelming academic stress explicitly conveyed.

Return ONLY a valid JSON object with this exact structure:
{
  "sentiment": "negative",
  "stress_score": 4,
  "confidence": "high"
}
"""

########################## Functions ##########################

def check_ollama ():
    """ Checks if Ollama is running and the model is available. """
    try:
        response = requests.get(f"{OLLAMA_URL}/tags", timeout=5)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        if MODEL_NAME not in models:
            print(f"Error: Model '{MODEL_NAME}' not found in Ollama.")
            print(f"Please run: ollama pull {MODEL_NAME}")
            return False
        return True
    except requests.exceptions.RequestException:
        print("Error: Could not connect to Ollama. Ensure Ollama is running locally.")
        return False

def analyze_text(text):
    """Sends the text to Ollama and retrieves the JSON response."""
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{SYSTEM_PROMPT}\n\nStudent Text:\n{text}",
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0, # Low temperature for consistent classification
            "seed": 42
        }
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/generate", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json().get("response", "")
        
        # parse json
        data = json.loads(result)

        # Validate fields
        sentiment = str(data.get("sentiment", "")).lower().strip()
        stress_score = data.get("stress_score", None)
        confidence = str(data.get("confidence", "")).lower().strip()

        if sentiment not in ["positive", "neutral", "negative"]:
            raise ValueError(f"Invalid sentiment value: {sentiment}")

        try:
            stress_score = int(stress_score)
            if stress_score < 1 or stress_score > 5:
                raise ValueError(f"Stress score out of range: {stress_score}")
        except (TypeError, ValueError):
            raise ValueError(f"Invalid stress score value: {stress_score}")

        return {
            "status": "success",
            "sentiment": sentiment,
            "stress_score": stress_score,
            "confidence": confidence,
            "raw_response": result
        }

    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_type": "Invalid_json",
            "raw_response": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": str(e),
            "raw_response": ""
        }

def calculate_metrics (df):
    """Calculates all required evaluation metrics on valid predictions."""
    valid_df = df[df["prediction_status"] == "success"].copy() ## keep only "success" predictions

    # 1. Sentiment Metrics
    y_true_sent = valid_df["sentiment_label"].str.lower()
    y_pred_sent = valid_df["predicted_sentiment"].str.lower()

    acc_sent = accuracy_score(y_true_sent, y_pred_sent) # measure the correct predictions percentage
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_sent, 
        y_pred_sent, 
        labels=["positive", "neutral", "negative"], 
        zero_division=0)
    macro_f1 = precision_recall_fscore_support(
        y_true_sent, 
        y_pred_sent, 
        average='macro', 
        zero_division=0)[2]
    cm_sent = confusion_matrix(
        y_true_sent, 
        y_pred_sent, 
        labels=["positive", "neutral", "negative"]
    )

    sentiment_metrics = {
        "accuracy": acc_sent,
        "macro_f1": macro_f1,
        "class_metrics": {
            "positive": {"precision": precision[0], "recall": recall[0], "f1": f1[0]},
            "neutral": {"precision": precision[1], "recall": recall[1], "f1": f1[1]},
            "negative": {"precision": precision[2], "recall": recall[2], "f1": f1[2]}
        },
        "confusion_matrix": cm_sent.tolist() # convert to list for JSON serialization
    }

    # 2. Stress Score Metrics
    # Filter out rows where ground truth ground_truth_stress_score is missing or invalid
    stress_df = valid_df.dropna(subset=["ground_truth_stress_score"]).copy()
    stress_df["ground_truth_stress_score"] =stress_df["ground_truth_stress_score"].astype(int)

    y_true_stress = stress_df["ground_truth_stress_score"]
    y_pred_stress = stress_df["predicted_stress_score"].astype(int)

    mae = mean_absolute_error(y_true_stress, y_pred_stress)
    rmse =np.sqrt(mean_squared_error(y_true_stress, y_pred_stress))
    exact_match = accuracy_score(y_true_stress, y_pred_stress)
    within_one = np.mean(np.abs(y_true_stress - y_pred_stress) <= 1)
    cm_stress = confusion_matrix(y_true_stress, y_pred_stress, labels=[1, 2, 3, 4, 5])

    stress_matrics = {
        "mae": mae,
        "rmse": rmse,
        "exact_match_accuracy": exact_match,
        "within_one_point_accuracy": within_one,
        "confusion_matrix": cm_stress.tolist()
    }

    # 3. Language Style Performance (if available)
    lang_perf = {}
    if "language_style" in valid_df.columns:
        for style in valid_df["language_style"].unique():
            subset = valid_df[valid_df["language_style"] == style]
            if len(subset) > 0:
                lang_perf[style] = {
                    "count": len(subset),
                    "sentiment_accuracy": accuracy_score(subset["sentiment_label"].str.lower(), subset["predicted_sentiment"])
                }

    return sentiment_metrics, stress_matrics, lang_perf

def main(test_mode = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not check_ollama(): return

    print(f"Loading dataset: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except  FileNotFoundError:
        print(f"Error: Input CSV file '{INPUT_CSV}' not found.")
        return

    if "ground_truth_stress_score" not in df.columns:
        print("Error: 'ground_truth_stress_score' column not found in dataset. Please add 1-5 ground truth scores before running.")
        return

    #Check for existing predictions to resume
    if os.path.exists(PREDICTIONS_CSV):
        print(f"Found existing predictions file: {PREDICTIONS_CSV}. Resuming ...")
        results_df = pd.read_csv(PREDICTIONS_CSV)
    else:
        results_df = df.copy()
        results_df["predicted_sentiment"] = None
        results_df["predicted_stress_score"] = None
        results_df["predicted_confidence"] = None
        results_df["raw_model_response"] = None
        results_df["prediction_status"] = "pending"
        results_df["sentiment_correct"] = None
        results_df["stress_score_absolute_error"] = None

    if test_mode:
        print("Running in TEST_MODE: processing only the first 5 entries.")
        pending_indices = results_df[results_df["prediction_status"] == "pending"].index[:5]
    else:
        pending_indices = results_df[results_df["prediction_status"] == "pending"].index

    print(f"Processing {len(pending_indices)} examples...")

    for idx in pending_indices:
        text = results_df.loc[idx, "text"]
        print(f"[{idx + 11}] Analyzing text...") 

        result = analyze_text(text)

        results_df.at[idx, "prediction_status"] = result["status"]
        results_df.at[idx, "raw_model_response"] = result["raw_response"]

        if result["status"] == "success":
            results_df.at[idx, "predicted_sentiment"] = result["sentiment"]
            results_df.at[idx, "predicted_stress_score"] = result["stress_score"]
            results_df.at[idx, "predicted_confidence"] = result["confidence"]

            #Calc row-level correctness
            true_sent = str(results_df.at[idx, "sentiment_label"]).lower().strip()
            results_df.at[idx, "sentiment_correct"] = (result["sentiment"] == true_sent)

            try:
                true_stress = int(results_df.at[idx, "ground_truth_stress_score"])
                results_df.at[idx, "stress_score_absolute_error"] = abs(result["stress_score"] - true_stress)
            except (ValueError, TypeError):
                pass  # If the true stress score is not a valid integer, skip error calculation

    #save incrementally
    results_df.to_csv(PREDICTIONS_CSV, index=False)
    time.sleep(0.1) #small delay to prevent overlaodng

    print("\nAll predictions completed. Saving results...")

    #calc matrics
    sent_metrics, stress_metrics, lang_perf = calculate_metrics(results_df)

    total_examples = len(results_df)
    valid_predictions = len(results_df[results_df["prediction_status"] == "success"])
    failed_predictions = len(results_df[results_df["prediction_status"] == "error"])

    #Generate evaluation report
    error_analysis ={
        "large_stress_discrepancy": results_df[results_df["stress_score_absolute_error"] >= 2]["sample_id"].tolist(),
        "positive_sentiment_high_stress": results_df[(results_df["predicted_sentiment"] == "positive") & (results_df["predicted_stress_score"] >= 4)]["sample_id"].tolist(),
        "negative_sentiment_low_stress": results_df[(results_df["predicted_sentiment"] == "negative") & (results_df["predicted_stress_score"] <= 2)]["sample_id"].tolist(),
        "incorrect_sentiments": results_df[results_df["sentiment_correct"] == False]["sample_id"].tolist(),
        "failed_predictions": results_df[results_df["prediction_status"] != "success"]["sample_id"].tolist()
    }

    report = {
        "configuration" :{
            "model" : MODEL_NAME,
            "system_prompt" : SYSTEM_PROMPT,
            "input_csv" : INPUT_CSV,
        },
        "execution_summary": {
            "total_examples": total_examples,
            "valid_predictions": valid_predictions,
            "failed_predictions": failed_predictions,
            "completion_rate": valid_predictions / total_examples if total_examples > 0 else 0
        },
        "sentiment_metrics": sent_metrics,
        "stress_score_metrics": stress_metrics,
        "language_style_performance": lang_perf,
        "error_analysis_candidates": error_analysis
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nEvaluation report saved to: {OUTPUT_DIR} ")
    print(f"- Predictions: {PREDICTIONS_CSV}")
    print(f"- Evaluation Report: {REPORT_JSON}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gemma 2 2B for SafeSpace Pilot")
    parser.add_argument("--test", action="store_true", help="Run in test mode (max 5 examples)")
    args = parser.parse_args()
    main(test_mode=args.test)