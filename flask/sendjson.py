# coding: utf-8
#!/usr/bin/env  python2.7

# json を指定したurlへPOSTする例

import urllib.request, json

if __name__ == '__main__':
    url = "http://127.0.0.1:5000/api/test"
    method = "POST"
    headers = {"Content-Type" : "application/json"}

    # PythonオブジェクトをJSONに変換する
    #obj = {"xxx" : "xxxx", 123 : 123}
    obj = {"10": 0.006044479086995125, "interval": 9, "3": 0.05119388550519943, "5": 0.08947592973709106, "4": 0.013541867956519127, "7": 0.0252132136374712, "6": 0.13018670678138733, "9": 0.6771517992019653, "8": 0.00719203008338809, "cur_std": 26.154877329664306}

    json_data = json.dumps(obj).encode("utf-8")

    # httpリクエストを準備してPOST
    request = urllib.request.Request(url, data=json_data, method=method, headers=headers)
    with urllib.request.urlopen(request) as response:
        response_body = response.read().decode("utf-8")
