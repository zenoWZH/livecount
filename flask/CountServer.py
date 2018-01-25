# coding: utf-8
#!/usr/bin/env  python2
import time
from flask import Flask, jsonify, request
import pdb
import os
import json
import sys
def excepthook(*args):
    print >> STDERR, 'caught'
    print >> STDERR, args

sys.excepthook = excepthook


app = Flask(__name__)

@app.route('/name/<name>.json')
def hello_world(name):
  greet = "Hello %s from flask!" % name
  result = {
    "Result":{
      "Greeting": greet
      }
  }
  return jsonify(ResultSet=result)

@app.route('/hello')
def say_hello():
  greet = "Hello from flask!"
  #pdb.set_trace()
  result = {
    "Result":{
      "Greeting": greet
      }
  }
  return jsonify(ResultSet=result)
@app.route('/init')
def initial():
  #import pdb
  #pdb.set_trace()
  time.sleep(3)
  dic = {}
  dic['initflag'] = True
  outfile = 'init.json'
  print(dic)
  with open(outfile,'w') as f:
    #json.dumps(dic,outfile)
    f.write(json.dumps(dic)+'\n')
    f.close()
    result = {
      "Result":{
        "initflag": True
      }
  }
  #with open('init.log','w') as f:
  #  f.write('\n')
  #  f.close()
  return jsonify(result)

@app.route('/api/test', methods=['POST'])
def face_info():
    def excepthook(*args):
      print >> STDERR, 'caught'
      print >> STDERR, args

    sys.excepthook = excepthook
    #pdb.set_trace()
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    print(request.json)
    filename='recieved'
    with open(filename+'.json','w') as outfile:
        json.dump(request.json,outfile,ensure_ascii=False)#,encoding='utf-8')
        outfile.write('\n')
    outfile.close()
    print('json received and saved')

    print('running count.py')
    os.popen('python count.py')
    print("counted")


    return jsonify(res='ok')
@app.route('/counter_status', methods=['GET'])
def count_result():
    #pdb.set_trace()
    class state:
      NO_REP = 1
      IN_REP = 2
      COOLDOWN = 3
    filename='global'
    with open(filename+'.json','r') as outfile:
        glob=json.load(outfile)#,encoding='utf-8')
        outfile.close()
    cur_state = glob['cur_state']
    out_time = glob['out_time']
    in_time = glob['in_time']
    global_counter = glob['global_counter']
    actions_counter = glob['actions_counter']
    in_frame_num = glob['in_frame_num']

    result = {'status': 'None'}
    out_time = in_frame_num/15
    if  ((cur_state == state.IN_REP) and (((out_time-in_time)<4) or (global_counter<5))):
      result['status'] = 'new hypothesis'
      result['global_counter'] = global_counter
      result['actions_counter'] = actions_counter
    if ((cur_state == state.IN_REP) and ((out_time-in_time)>=4) and (global_counter>=5)):
      result['status'] = 'counting'
      result['global_counter'] = global_counter
      result['actions_counter'] = actions_counter
    if ((cur_state == state.COOLDOWN) and (global_counter>=5)):
      result['status'] = 'cooldown'
      result['global_counter'] = global_counter
      result['actions_counter'] = actions_counter

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
