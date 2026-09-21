import ast
import base64
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from src.prescription.notebook_runner import run_live, render_html
from src.prescription.progress import emit, progress_rows


class LiveRunnerTests(unittest.TestCase):
    def test_streams_progress_before_process_finishes(self):
        states=[]
        script="import time,json; print('@@RX_PROGRESS@@'+json.dumps({'stage':'Training','current':1,'total':2}),flush=True); time.sleep(1); print('finished',flush=True)"
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/'live.log'
            run_live([sys.executable,'-u','-c',script],log_path=log,on_update=lambda s:states.append(copy.deepcopy(s)))
            self.assertIn('finished',log.read_text())
        self.assertTrue(any(s['status']=='Running' and s.get('training',{}).get('current')==1 for s in states))
        self.assertEqual(states[-1]['status'],'Completed')
        self.assertIn('1 / 2 (50.0%)',render_html(states[-1]))

    def test_failure_does_not_look_complete(self):
        states=[]
        with self.assertRaises(subprocess.CalledProcessError):
            run_live([sys.executable,'-u','-c',"print('failure detail',flush=True); raise SystemExit(7)"],on_update=lambda s:states.append(copy.deepcopy(s)))
        self.assertEqual(states[-1]['status'],'Failed (exit 7)')
        self.assertIn('failure detail',states[-1]['tail'])

    def test_interrupt_stops_launched_process(self):
        import src.prescription.notebook_runner as runner
        actual=runner.subprocess.Popen
        children=[]
        def launch(*args,**kwargs):
            proc=actual(*args,**kwargs);children.append(proc);return proc
        def interrupt(state):
            if state['status']=='Running':raise KeyboardInterrupt()
        with patch.object(runner.subprocess,'Popen',side_effect=launch):
            with self.assertRaises(KeyboardInterrupt):
                run_live([sys.executable,'-u','-c','import time; time.sleep(30)'],on_update=interrupt)
        self.assertIsNotNone(children[0].poll())

    def test_unknown_work_is_indeterminate_and_html_is_escaped(self):
        rendered=render_html({'title':'<script>bad</script>','status':'Running','elapsed':70,'silent':65,'tail':['<img src=x>']})
        self.assertIn('percentage unavailable',rendered)
        self.assertNotIn('<script>',rendered)
        self.assertIn('does not prove it is progressing',rendered)


class ProgressTests(unittest.TestCase):
    def test_real_iteration_counts_and_opt_in(self):
        sink=io.StringIO()
        with patch.dict(os.environ,{'RX_PROGRESS':'1'}),contextlib.redirect_stdout(sink):
            self.assertEqual(list(progress_rows([1,2,3],'Verify',every=2)),[1,2,3])
        events=[json.loads(line.split('@@RX_PROGRESS@@')[1]) for line in sink.getvalue().splitlines()]
        self.assertEqual([e['current'] for e in events],[0,2,3])
        with patch.dict(os.environ,{'RX_PROGRESS':'0'}),contextlib.redirect_stdout(io.StringIO()) as quiet:
            emit('Test',1,1)
        self.assertEqual(quiet.getvalue(),'')

    def test_trainer_steps_and_validation_reset(self):
        from src.prescription.training_progress import TrainingProgress
        callback=TrainingProgress(5,2)
        state=SimpleNamespace(global_step=4,max_steps=10,epoch=.4)
        args=SimpleNamespace(gradient_accumulation_steps=16)
        control=SimpleNamespace(should_save=False)
        with patch('src.prescription.training_progress.emit') as event:
            callback.on_step_begin(args,state,control)
            self.assertEqual(event.call_args.args,('Training',4,10))
            for _ in range(3):callback.on_prediction_step(args,state,control)
            self.assertEqual(event.call_args.args,('Validation batches',3,3))
            callback.on_evaluate(args,state,control,metrics={'eval_loss':1})
            callback.on_prediction_step(args,state,control)
            self.assertEqual(event.call_args.args,('Validation batches',1,3))


class NotebookTests(unittest.TestCase):
    def test_notebook_snapshot_matches_sources_and_cells_parse(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/RxHandBD_TrOCR_Base_Colab.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            text=''.join(cell['source']);tree=ast.parse(text)
            self.assertEqual(cell['outputs'],[])
            if 'BUNDLE_SHA256 =' in text:
                bundle=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='BUNDLE')
                with zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle))) as z:
                    for name in z.namelist():
                        if (root/name).exists():self.assertEqual(z.read(name),(root/name).read_bytes(),name)
                        if name.endswith('.py'):ast.parse(z.read(name).decode())
        full='\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertIn("'-u', '-m'",full)
        self.assertIn("trocr-base-rxhandbd-v2",full)
        self.assertIn('RUN_FINAL_TEST = False',full)


if __name__=='__main__':unittest.main()
