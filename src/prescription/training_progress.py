"""Trainer events for notebook progress; percentages are based on actual steps."""
import math
from transformers import TrainerCallback
from .progress import emit


class TrainingProgress(TrainerCallback):
    def __init__(self, evaluation_size, batch_size):
        self.eval_total=math.ceil(evaluation_size/batch_size)
        self.eval_step=0
        self.micro_step=0

    def training(self,state):
        emit('Training',state.global_step,state.max_steps,epoch=round(state.epoch or 0,3))

    def on_train_begin(self,args,state,control,**kwargs):self.training(state)
    def on_step_begin(self,args,state,control,**kwargs):
        self.micro_step=0
        self.training(state)
    def on_substep_end(self,args,state,control,**kwargs):
        self.micro_step+=1
        emit('Accumulating gradient batches',self.micro_step,args.gradient_accumulation_steps)
    def on_step_end(self,args,state,control,**kwargs):
        self.training(state)
        if control.should_save:emit('Validation / checkpoint due')
    def on_log(self,args,state,control,logs=None,**kwargs):
        emit('Training metrics',metrics={k:v for k,v in (logs or {}).items() if isinstance(v,(int,float))})
    def on_prediction_step(self,args,state,control,**kwargs):
        self.eval_step+=1
        emit('Validation batches',self.eval_step,self.eval_total)
    def on_evaluate(self,args,state,control,metrics=None,**kwargs):
        emit('Validation complete',self.eval_step,self.eval_total,metrics=metrics)
        self.eval_step=0
        if control.should_save:emit('Writing checkpoint — Drive may be slow')
    def on_save(self,args,state,control,**kwargs):
        emit('Checkpoint saved',state.global_step,state.max_steps)
    def on_train_end(self,args,state,control,**kwargs):
        emit('Training stopped; selecting best checkpoint',state.global_step,state.max_steps)
