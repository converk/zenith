import hashlib,json,time
from pathlib import Path
import torch
from riichi_ppo_v1.model import KyokuTransformerActorCritic,ModelConfig
from riichi_ppo_v1.sft.train import evaluate

dataset=Path('datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16')
payload=torch.load('checkpoints/train_riichi_v13/sft/best.pt',map_location='cpu',weights_only=False)
model=KyokuTransformerActorCritic(ModelConfig(**payload['model_config']))
model.load_state_dict(payload['model']); device=torch.device('cuda:0'); model.to(device)
config=dict(payload['sft_config']); config['learner_gpus']=2
results={}
for label,maximum in [('fixed_150k',150000),('full',0)]:
  started=time.perf_counter(); metrics=evaluate(model,dataset,config,device,max_samples=maximum)
  results[label]={'elapsed_seconds':time.perf_counter()-started,'metrics':metrics}
# Independent fixed identity digest before bucketing.
h=hashlib.sha256(); count=0
from riichi_ppo_v1.sft.data import iter_split_samples
for s in iter_split_samples(dataset,'validation',seed=int(config['seed']),shuffle=False,include_critic=False):
  if count==150000: break
  h.update(json.dumps([s.year,s.game_id,s.kyoku_index,s.seat,s.decision_index],separators=(',',':')).encode()+b'\n'); count+=1
results['fixed_150k_identity_count']=count; results['fixed_150k_identity_sha256']=h.hexdigest()
(Path(__file__).resolve().parents[1]/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
