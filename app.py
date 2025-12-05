from flask import Flask, request, render_template
import os
import datetime
import time # NEW: For time.sleep()
from dotenv import load_dotenv
from eval.metrics import sanitize_label # Assuming this function handles the raw output
import google.generativeai as genai


load_dotenv()

# --- Gemini Configuration (Only runs once on app startup) ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEN_CFG = {"temperature": 0.2, "top_p": 0.95, "top_k": 40}
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME, generation_config=GEN_CFG)

app = Flask(__name__)

# --- Reusable function for safe API call with Retry and Exponential Backoff ---
def safe_gemini_call(prompt_text: str, max_retries: int = 4) -> str:
    """
    Attempts to call the Gemini API, retrying on rate limit errors (429) 
    with exponential backoff to ensure service stability.
    """
    delay = 2  # Initial wait time in seconds
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(prompt_text)
            # Successfully got a response
            return (resp.text or "").strip() 

        except Exception as e:
            # Check for Rate Limit or Resource Exhausted errors
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                if attempt < max_retries - 1:
                    print(f"⚠️ Rate limit hit. Attempt {attempt + 1}/{max_retries}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponentially increase the delay
                else:
                    # Last attempt failed due to rate limit
                    print(f"🛑 Max retries reached. Quota exceeded.")
                    return "ERROR: Quota Exceeded. Please try again later."
            else:
                # Handle other errors (e.g., bad request, permission denied, network issues)
                print(f"❌ Error: {e}")
                return f"ERROR: API Communication Failed: {str(e)}"
    
    return "ERROR: Classification failed after all retries."




@app.route('/', methods=['GET', 'POST'])
def index():
    raw_result = None
    result_class = None
    
    if request.method == 'POST':
        user_text = request.form['text']
        
        # 1. Build the prompt
        # Using a highly constrained format for reliability
        prompt = f"You are a precise sentiment classifier. Output only one token: POSITIVE or NEGATIVE.\nClassify the sentiment of the text below.\n---\n{user_text}\n---\nRespond with only: POSITIVE or NEGATIVE."
        
        # 2. Call the safe function
        raw_output = safe_gemini_call(prompt)
        
        # 3. Process and sanitize the output
        if raw_output.startswith("ERROR:"):
            raw_result = "Service Unavailable: Please wait a moment and try again."
            result_class = "ERROR"
        else:
            # Use the existing sanitize logic from your project
            sanitized_result = sanitize_label(raw_output)
            raw_result = sanitized_result
            
            # Set the class for styling
            if "POSITIVE" in sanitized_result:
                result_class = "POSITIVE"
            elif "NEGATIVE" in sanitized_result:
                result_class = "NEGATIVE"
            else:
                raw_result = f"Uncertain: {raw_output}" # Fallback for unexpected model output
                result_class = "ERROR"

    year = datetime.datetime.now().year
    return render_template('index.html', raw_result=raw_result, result_class=result_class, year=year)

if __name__ == '__main__':
    # Use production mode for Heroku
    app.run(host='0.0.0.0', port=5050, debug=False)