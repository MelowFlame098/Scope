from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Scope AI Service is running", "service": "AI/ML"})

@app.route('/analyze', methods=['POST'])
def analyze():
    # Placeholder for analysis logic
    return jsonify({"status": "analysis_pending"})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
