from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Flask Backend!"

@app.route('/rag-pipeline', methods=['POST'])
def rag_pipeline():
    data = request.json
    if not data or 'videoId' not in data:
        return jsonify({"status": "error", "message": "feature is not available for this video"}), 500
    
    
    
    

if __name__ == '__main__':
    app.run(debug=True)