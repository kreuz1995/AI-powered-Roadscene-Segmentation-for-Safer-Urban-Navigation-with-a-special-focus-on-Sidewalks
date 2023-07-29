import sys
sys.path.append('../model_f')
from model import *


#max_score1 = 10
max_score2=0
for i in range(0, 100):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    if max_score2 < valid_logs['iou_score']:
        max_score2 = valid_logs['iou_score']
        torch.save(model,"Pytorch_Weights/DeepLabv3Plus_resnet101.pth")
        # torch.save(model.state_dict(), "for_onnx/DeepLabV3Plus_resnet101.pth")
        print('Validation Model saved!')

        
# torch.save("Total_Save-DLv3P.pth") 
if i == 40:
    optimizer.param_groups[0]['lr'] = 0.00005
    print('Decreasing decoder learning rate to 1e-5!')