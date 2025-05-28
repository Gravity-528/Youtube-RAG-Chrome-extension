from flask import Flask, request, jsonify
app = Flask(__name__)

# @app.route('/videoId', methods=['POST'])
# def receive_video_id():
#     data = request.json
#     video_id = data.get('videoId')
#     if video_id:
#         print("Received videoId:", video_id)
#         return jsonify({"status": "success", "videoId": video_id})
#     return jsonify({"status": "error", "message": "No videoId provided"}), 400

@app.route('/')
def home():
    return "Welcome to Flask Backend!"


if __name__ == '__main__':
    app.run(debug=True)