from flask import Flask, request, jsonify
from helper import ask_question

app = Flask(__name__)

@app.route('/query_video', methods=['POST'])
def query_video():
    data = request.json
    video_id = data.get('videoId')
    query = data.get('query')

    if not video_id or not query:
        return jsonify({'error': 'videoId and query are required'}), 400

    try:
        answer = ask_question(video_id, query)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
