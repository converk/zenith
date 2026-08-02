import json,time
from pathlib import Path
import torch
from riichi_ppo_v1.model import KyokuTransformerActorCritic,ModelConfig
from riichi_ppo_v1.sft.heuristic_evaluation import evaluate_against_heuristics
p=torch.load('checkpoints/train_riichi_v13_sft/best.pt',map_location='cpu',weights_only=False)
m=KyokuTransformerActorCritic(ModelConfig(**p['model_config'])); m.load_state_dict(p['model']); d=torch.device('cuda:0'); m.to(d)
c=dict(p['sft_config']); c.update({'heuristic_evaluation_hanchan_count':96,'heuristic_evaluation_parallel_hanchan_count':24,'heuristic_evaluation_seed_base':20260717})
OUT=Path(__file__).resolve().parents[1]
rows=[]
for i in (1,2):
 r=evaluate_against_heuristics(m,d,c,hanchan_count=96,cycle=0); rows.append(r)
 (OUT/f'heuristic_run_{i}.json').write_text(json.dumps(r,indent=2)+'\n')
 print('RUN',i,json.dumps(r,indent=2),flush=True)
comparison={'exact_equal_excluding_elapsed':{k:rows[0][k]==rows[1][k] for k in rows[0] if 'performance' not in k},'all_core_exact':all(rows[0][k]==rows[1][k] for k in rows[0] if 'performance' not in k)}
(OUT/'heuristic_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n'); print(json.dumps(comparison,indent=2))
