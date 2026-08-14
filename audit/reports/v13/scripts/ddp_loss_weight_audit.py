import json,random
from pathlib import Path
import numpy as np
from riichi_ppo_v1.sft.data import iter_split_samples
from riichi_ppo_v1.sft.train import length_bucketed_batches
root=Path('datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16'); groups=((0,1),(1,75),(75,76),(76,133),(133,170),(170,239),(239,240),(240,241))
iters=[]
for rank in (0,1):
 s=iter_split_samples(root,'train',seed=1,shuffle=True,rank=rank,world_size=2,include_critic=False)
 iters.append(iter(length_bucketed_batches(s,256,window_batches=32,rng=random.Random(1+rank))))
rows=[]
for step in range(32):
 counts=[]
 for rank in (0,1):
  batch=next(iters[rank]); legal=np.stack([x.legal_mask for x in batch]); teacher=np.stack([x.teacher_mask for x in batch]); available=np.stack([legal[:,a:b].any(1) for a,b in groups],1)
  counts.append({'rank':rank,'samples':len(batch),'group_eligible':int((available.sum(1)>=2).sum()),'rule_eligible':int(((teacher.sum(1)>=1)&(legal.sum(1)>=2)).sum())})
 rows.append({'step':step+1,'ranks':counts,'group_count_difference':counts[0]['group_eligible']-counts[1]['group_eligible'],'rule_count_difference':counts[0]['rule_eligible']-counts[1]['rule_eligible']})
report={'steps':rows,'steps_with_group_denominator_difference':sum(r['group_count_difference']!=0 for r in rows),'steps_with_rule_denominator_difference':sum(r['rule_count_difference']!=0 for r in rows),'conclusion':'DDP averages local eligible-means equally; whenever eligible counts differ this is not the global eligible-sample mean.'}
(Path(__file__).resolve().parents[1]/'ddp_loss_weight.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k!='steps'},indent=2))
