from pathlib import Path
BASE_SCRIPT = Path(__file__).with_name('resume_audit.py')
exec(BASE_SCRIPT.read_text())

import shutil
ROOT2 = ROOT / 'fixed_horizon'
ROOT2.mkdir(exist_ok=True)
def equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a, np.ndarray): return np.array_equal(a,b)
    return a == b
base['max_train_steps'] = 3
base['checkpoint_interval_steps'] = 1
train_identity_set = {(r.year,r.game_id,r.kyoku_index,r.seat,r.decision_index) for r in train_rows}
original_collate2 = original_collate
current_log = None
interrupt_after = None
train_batches = 0
class CanaryInterrupt(Exception): pass
def interrupting_collate(rows, *args, **kwargs):
    global train_batches
    identities = [(r.year,r.game_id,r.kyoku_index,r.seat,r.decision_index) for r in rows]
    is_train = all(x in train_identity_set for x in identities)
    if is_train and interrupt_after is not None and train_batches >= interrupt_after:
        raise CanaryInterrupt()
    if is_train:
        train_batches += 1
        if current_log is not None: current_log.extend(identities)
    return original_collate2(rows, *args, **kwargs)
t.collate_samples = interrupting_collate
def fixed_run(output, resume=None, stop_new_batches=None):
    global current_log, interrupt_after, train_batches
    cfg=copy.deepcopy(base); cfg['checkpoint_dir']=str(output); cfg['resume']=str(resume) if resume else None
    current_log=[]; interrupt_after=stop_new_batches; train_batches=0
    try: t.train_worker(0,1,cfg,DATA,output)
    except CanaryInterrupt: pass
    log=current_log; current_log=None; interrupt_after=None
    return torch.load(output/'latest.pt',map_location='cpu',weights_only=False),log
cont, cont_ids=fixed_run(ROOT2/'continuous')
ids_all=[]
a,ids=fixed_run(ROOT2/'restart',stop_new_batches=1); ids_all+=ids
b,ids=fixed_run(ROOT2/'restart',ROOT2/'restart'/'latest.pt',stop_new_batches=1); ids_all+=ids
c,ids=fixed_run(ROOT2/'restart',ROOT2/'restart'/'latest.pt'); ids_all+=ids
report={
 'fixed_scheduler_horizon':3,'continuous_global_step':cont['global_step'],'resumed_global_step':c['global_step'],
 'cursor_a':a['data_cursor'],'cursor_b':b['data_cursor'],'cursor_n':c['data_cursor'],
 'consumed_identity_sequence_equal':cont_ids==ids_all,'continuous_identities':cont_ids,'resumed_identities':ids_all,
 'model_exact':equal(cont['model'],c['model']),'optimizer_exact':equal(cont['optimizer'],c['optimizer']),
 'scheduler_exact':equal(cont['scheduler'],c['scheduler']),'torch_rng_exact':equal(cont['torch_rng'],c['torch_rng']),
 'numpy_rng_exact':equal(cont['numpy_rng'],c['numpy_rng']),'python_rng_exact':equal(cont['python_rng'],c['python_rng']),
}
(Path(__file__).resolve().parents[1]/'resume_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
shutil.rmtree(ROOT)
