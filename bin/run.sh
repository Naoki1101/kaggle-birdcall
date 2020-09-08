cd ../src


# python train.py -m 'resnet18_001' -c 'test'
# python train.py -m 'resnet18_002' -c 'epoch=10'
# python train.py -m 'resnet18_003' -c 'BCEWithLogitsLoss'
# python train.py -m 'resnet18_004' -c 'mod head'
# python train.py -m 'resnet18_005' -c 'epoch=50'
# python train.py -m 'resnet18_006' -c 'mod Dataset'


# python train.py -m 'resnet34_001' -c 'epoch=50'
# python train.py -m 'resnet34_002' -c 'BCEWithLogitsLoss, epoch=30'
# python train.py -m 'resnet34_003' -c 'BCEWithLogitsLoss, epoch=50'
# python train.py -m 'resnet34_004' -c 'epoch=30, mixup, micro'
# python train.py -m 'resnet34_005' -c 'epoch=50, mixup, micro'


# python train.py -m 'resnet50_001' -c 'epoch=50'
# python train.py -m 'resnet50_002' -c 'epoch=50'
# python train.py -m 'resnet50_003' -c 'mod head'
# python train.py -m 'resnet50_004' -c 'epoch=30'


# python train.py -m 'se_resnext50_32x4d_001' -c 'epoch=30'
# python train.py -m 'se_resnext50_32x4d_002' -c 'epoch=50'
# python train.py -m 'se_resnext50_32x4d_003' -c 'epoch=50, mixup'
# python train.py -m 'se_resnext50_32x4d_004' -c 'epoch=30, noise'
# python train.py -m 'se_resnext50_32x4d_005' -c 'epoch=50, noise'
# python train.py -m 'se_resnext50_32x4d_006' -c 'epoch=30, mod noise'
# python train.py -m 'se_resnext50_32x4d_007' -c 'epoch=30, mod noise'
# python train.py -m 'se_resnext50_32x4d_008' -c 'epoch=30, mixup, noise'
# python train.py -m 'se_resnext50_32x4d_009' -c 'epoch=30, mixup, noise, pin_memory=True'
# python train.py -m 'se_resnext50_32x4d_010' -c 'epoch=30, mixup, noise, water_noise, shift'
# python train.py -m 'se_resnext50_32x4d_011' -c 'epoch=30, mixup, shift'
# python train.py -m 'se_resnext50_32x4d_012' -c 'epoch=50, mixup, noise, water_noise, shift'
# python train.py -m 'se_resnext50_32x4d_013' -c 'epoch=30, mixup, noise, water_noise, shift, mod head'
# python train.py -m 'se_resnext50_32x4d_014' -c 'epoch=30, mixup, noise, water_noise, bus_noise, shift'
# python train.py -m 'se_resnext50_32x4d_015' -c 'epoch=30, mixup, water_noise, shift'
# python train.py -m 'se_resnext50_32x4d_016' -c 'epoch=30, mixup, water_noise, bus_noise, shift'
python train.py -m 'se_resnext50_32x4d_016' -c 'epoch=30, mixup, noise, water_noise, shift'


# python train.py -m 'resnest50_001' -c 'epoch=30'
# python train.py -m 'resnest50_002' -c 'epoch=30, mixup, noise, water_noise, shift'
# python train.py -m 'resnest50_frelu_001' -c 'epoch=30'


# python train.py -m 'efficientnetb3_001' -c 'test'
# python train.py -m 'efficientnetb3_002' -c 'epoch=30'
# python train.py -m 'efficientnetb3_003' -c 'epoch=30, mixup, noise'
# python train.py -m 'efficientnetb4_001' -c 'epoch=30, mixup, noise'
# python train.py -m 'efficientnetb4_002' -c 'epoch=30, mixup, noise, seed=2021'
# python train.py -m 'efficientnetb4_003' -c 'epoch=30, mixup, noise, water_noise, shift'


cd ../
git add -A
git commit -m '...'
git push origin master