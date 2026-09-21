"""Forward subprocess stdout/stderr into notebook output with honest live progress."""
import collections
import html
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

PREFIX = '@@RX_PROGRESS@@'


def render_html(state):
    esc = lambda value: html.escape(str(value))
    def bar(title, current, total):
        known = isinstance(total,(int,float)) and total > 0 and current is not None
        attrs = f'value="{max(0,min(current,total))}" max="{total}"' if known else ''
        count = f'{current:,} / {total:,} ({100*current/total:.1f}%)' if known else 'working; percentage unavailable'
        return f'<div><b>{esc(title)}</b> — {esc(count)}<br><progress style="width:100%;height:22px" {attrs}></progress></div>'
    event = state.get('event',{})
    training = state.get('training')
    status = state['status']
    body = f'<h3>{esc(state["title"])} · {esc(status)}</h3>'
    body += f'<p>Elapsed: {state["elapsed"]:.0f}s · Last output: {state["silent"]:.0f}s ago</p>'
    if status == 'Running':
        body += bar(event.get('stage','Starting process'),event.get('current'),event.get('total'))
    if training:
        body += bar('Training optimizer steps',training.get('current'),training.get('total'))
        body += '<p>Epoch: '+esc(training.get('epoch','—'))+'</p>'
    if state.get('metrics'):body += '<p>'+esc(json.dumps(state['metrics']))+'</p>'
    if status == 'Running' and state['silent'] >= 60:
        body += '<p>No new output for over a minute. The process is still running; this does not prove it is progressing. Check the last stage and log below.</p>'
    if state.get('log_path'):body += '<p>Full log: '+esc(state['log_path'])+'</p>'
    body += '<pre style="max-height:280px;overflow:auto;white-space:pre-wrap">'+esc('\n'.join(state['tail']))+'</pre>'
    return '<div style="padding:16px;border:1px solid #aaa;border-radius:10px">'+body+'</div>'


def stop_process(proc):
    if proc.poll() is not None:return
    # Colab/Linux: stop the launched process group, including training children.
    for sig, wait in [(signal.SIGINT,5),(signal.SIGTERM,5),(signal.SIGKILL if os.name!='nt' else signal.SIGTERM,5)]:
        if proc.poll() is not None:break
        if os.name == 'nt':proc.terminate()
        else:os.killpg(proc.pid,sig)
        try:proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:pass


def run_live(command, cwd=None, title='Working', log_path=None, on_update=None):
    if on_update is None:
        from IPython.display import HTML, display
        handle = display(HTML('<p>Starting '+html.escape(title)+'…</p>'),display_id=True)
        on_update = lambda state: handle.update(HTML(render_html(state)))
    env = dict(os.environ,PYTHONUNBUFFERED='1',RX_PROGRESS='1',TOKENIZERS_PARALLELISM='false')
    proc = subprocess.Popen(list(map(str,command)),cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,encoding='utf-8',errors='replace',bufsize=1,start_new_session=os.name!='nt')
    lines=queue.Queue()
    def reader():
        try:
            for line in proc.stdout:lines.put(line.rstrip('\r\n'))
        finally:lines.put(None)
    worker=threading.Thread(target=reader,daemon=True);worker.start()
    started=time.monotonic();last_output=started;last_render=0;ended=False
    state={'title':title,'status':'Running','tail':collections.deque(maxlen=18),'elapsed':0,'silent':0,'log_path':str(log_path or '')}
    stream=None
    try:
        if log_path:
            Path(log_path).parent.mkdir(parents=True,exist_ok=True)
            stream=Path(log_path).open('a',encoding='utf-8',buffering=1)
            stream.write('\n--- '+title+' ---\n')
        while not ended or proc.poll() is None:
            try:
                line=lines.get(timeout=0.2)
                if line is None:ended=True
                else:
                    last_output=time.monotonic()
                    if stream:stream.write(line+'\n')
                    if line.startswith(PREFIX):
                        try:
                            event=json.loads(line[len(PREFIX):]);state['event']=event
                            if event['stage']=='Training':state['training']=event
                            if event.get('metrics'):state['metrics']=event['metrics']
                        except (ValueError,KeyError):state['tail'].append(line)
                    elif line.strip():state['tail'].append(line)
            except queue.Empty:pass
            now=time.monotonic()
            if now-last_render>=0.5:
                state.update(elapsed=now-started,silent=now-last_output);on_update(state);last_render=now
        code=proc.wait()
        state.update(status='Completed' if code==0 else f'Failed (exit {code})',elapsed=time.monotonic()-started,silent=0)
        on_update(state)
        if code:raise subprocess.CalledProcessError(code,command)
        return code
    except BaseException:
        stop_process(proc)
        if state['status']=='Running':
            state.update(status='Stopped',elapsed=time.monotonic()-started);on_update(state)
        raise
    finally:
        if stream:stream.close()
        proc.stdout.close()
        worker.join(timeout=2)
