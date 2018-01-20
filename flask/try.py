from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello():
    return 'Hello World'

@app.route('/api/test', methods=['POST'])
def face_info():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    print request.json

    return flask.jsonify(res='ok')
    
if __name__ == '__main__':
    app.run()