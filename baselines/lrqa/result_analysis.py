import matplotlib.pyplot as plt
import json
import os
file_name = 'trainer_state.json'
#file_path = '/home/yuanhuang/quality2/quality/baselines/models/dpr_roberta_base_temp2'
#result_file = os.path.join(file_path, file_name)
result_file = file_name
data = json.load(open(result_file))
log_hist = data["log_history"]
eval_acc = []
eval_loss = []
train_loss = []
epoches = []
for i in range(1,len(log_hist)):
    loss = "loss"
    e_loss = "eval_loss"
    if e_loss in log_hist[i].keys():
        eval_loss.append(log_hist[i]["eval_loss"])
        epoches.append(log_hist[i]["epoch"])
    elif loss in log_hist[i].keys():
        train_loss.append(log_hist[i]["loss"])
    else:
        continue

#max_acc = max(eval_acc)
#print(max_acc)
print(epoches)
print(eval_loss)
print(train_loss)
plt.plot(epoches,eval_loss,'b')
plt.plot(epoches[:-2],train_loss[:-1],'r')
plt.xlabel('epoch')
plt.ylabel('eval_acc')
plt.show()




