from flask import Flask
from flask import request, send_from_directory
from utils import init_context, init_conversation, process_output, ids2text
import numpy as np
import redisai


app = Flask(__name__, static_url_path='/frontend')
con = redisai.Client()
context = init_context()


@app.route('/')
def frontend():
    return send_from_directory('frontend', 'index.html')


@app.route('/<path:path>')
def frontend_assets(path):
    return send_from_directory('frontend', path)


@app.route('/next')
def next_():
    last = request.args.get('lastid')
    if last == 'null':
        breakpoint()
    premise = request.args.get('premise')
    if last and premise:
        return {"error": "You shouldn't send both ``last`` and ``context``"}
    elif not any([last, premise]):
        return {"error": "You must send something"}
    elif last:
        last = np.array([[int(last)]])
    elif premise:
        con.tensorset('context', context)
        last = init_conversation(premise)
    wordids = []
    con.tensorset('last', last)
    con.modelrun('gptmodel', inputs=['last', 'context'], outputs=['out', 'context'])
    out = con.tensorget('out')
    last = process_output(out)
    wordids.append(last.item())
    words = ids2text(wordids)
    return {"next": words, "nextid": wordids[-1]}
