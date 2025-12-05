
# Enterprise Prompt Engineering

Production-ready prompt engineering portfolio featuring sentiment classification with Google Gemini, a modern Flask web interface, automated evaluation, and Docker deployment.

## 🚀 Project Overview
This project demonstrates prompt engineering and evaluation using Google Gemini LLM for sentiment classification. It features:

- Zero-shot and few-shot prompt variants
- Automated evaluation of accuracy, latency, and cost
- Logging and reproducibility for robust experimentation
- A modern Flask web interface with Bootstrap for user-friendly sentiment classification

## 🛠️ Setup Instructions

1. **Clone the repository**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Add your Gemini API key to `.env`:**
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
4. **Run the Flask web app locally:**
   ```sh
   python app.py
   ```
   Then visit [http://localhost:5050](http://localhost:5050) in your browser.

## 🐳 Docker Setup

You can run the app in a Docker container:

```sh
docker build -t sentiment-app .
docker run -p 5050:5050 sentiment-app
```

## 📁 Project Structure

- `app.py`: Flask web app for interactive sentiment classification
- `templates/index.html`: Jinja2 template with Bootstrap styling
- `tasks/classification/run.py`: Main script for running experiments
- `tasks/classification/prompts.yaml`: Prompt templates and few-shot examples
- `tasks/classification/dataset.jsonl`: Sentiment dataset
- `tasks/classification/results/`: Output CSVs and logs
- `ablations/compare.py`: Compare results between variants
- `Dockerfile`: Container setup for deployment

## 📊 Sample Results

Example output from the web app and logs:
```
RAW OUTPUT for 'Loved the service and the staff!': POSITIVE
RAW OUTPUT for 'Awful wait time and cold food.': NEGATIVE
...
[v1_zero_shot] saved -> .../results/v1_zero_shot.csv
Summary: {'n': 5, 'acc': 1.0, 'latency_s_avg': 0.805, 'cost_total_usd': 0.0}
```

## 📈 Performance Analysis

- **Accuracy:** High accuracy (1.0) indicates the model correctly classifies sentiment for all examples.
- **Latency:** Average latency per prediction is logged for each variant.
- **Cost:** Cost is tracked (if applicable) for each run.

## 🎨 Customization

- Modify `prompts.yaml` to experiment with different prompt styles or few-shot examples.
- Add more data to `dataset.jsonl` for broader evaluation.
- Use `ablations/compare.py` to compare different model variants.
- Edit `templates/index.html` for UI changes (Bootstrap supported).

## 🚢 Deployment Options

You cannot host Flask apps on GitHub Pages. For free hosting, use:

- **Heroku** ([heroku.com](https://heroku.com)) — Free tier for small apps
- **Render** ([render.com](https://render.com)) — Free web service tier

Both support Docker and Python web apps. See their docs for step-by-step deployment.

---

For questions or improvements, feel free to reach out!


# Prompt Engineering Portfolio

Small, reproducible examples of **prompt design + evaluation**:
- Tasks: classification (sentiment) — more to come (retrieval, code-refactor)
- Variants: zero-shot vs few-shot with clear, measured **ACC / latency / cost**
- Ablations: show exactly what changed and why it helped

## ▶️ How to run
```bash
pip install -r requirements.txt
python tasks/classification/run.py
python ablations/compare.py
```

## 📄 License 
MIT

## 👤 Author
Hakim Murphy