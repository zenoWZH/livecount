import json

obj = {'8': -2.890625, '6': -0.07257080078125, '3': -1.748046875, '4': -1.00390625, 
'7': 0.28466796875, '9': -0.106689453125, 'cur_std': 0.001198957723379135, '10': 0.669921875, 
'5': 1.8515625}
filename='recieve'
with open(filename+'.json','w') as outfile:
    json.dump(obj,outfile,ensure_ascii=False)#,encoding='utf-8')
    outfile.write('\n')


