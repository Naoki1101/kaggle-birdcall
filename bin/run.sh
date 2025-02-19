cd ../src

########################################################
# Final submission
########################################################
python train.py -m 'final_001' -c 'model1'
python train.py -m 'final_002' -c 'model2'
python train.py -m 'final_003' -c 'model3'


########################################################
# Experiments
########################################################
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
# python train.py -m 'se_resnext50_32x4d_016' -c 'epoch=30, mixup, noise, water_noise, shift'
# python train.py -m 'se_resnext50_32x4d_016' -c 'epoch=30, mixup, noise, water_noise, shift, n_mels=256'
# python train.py -m 'se_resnext50_32x4d_017' -c 'epoch=30, mixup, noise, water_noise, shift'
# python train.py -m 'se_resnext50_32x4d_018' -c 'epoch=30, mixup, shift'
# python train.py -m 'se_resnext50_32x4d_019' -c 'epoch=30, mixup, noise, water_noise, shift, seed=2021'
# python train.py -m 'se_resnext50_32x4d_020' -c 'epoch=30, mixup, noise, water_noise, shift, seed=2022'
# python train.py -m 'se_resnext50_32x4d_021' -c 'epoch=30, mixup, noise, water_noise, shift, seed=2023'
# python train.py -m 'se_resnext50_32x4d_022' -c 'epoch=30, mixup, noise, water_noise, walk_noise, shift'
# python train.py -m 'se_resnext50_32x4d_023' -c 'epoch=30, mixup, noise, water_noise, walk_noise, rain_noise, shift'
# python train.py -m 'se_resnext50_32x4d_024' -c 'epoch=30, mixup, noise, rain_noise, shift'
# python train.py -m 'se_resnext50_32x4d_025' -c 'epoch=30, mixup, water_noise, rain_noise, shift'
# python train.py -m 'se_resnext50_32x4d_026' -c 'epoch=30, mixup, noise, water_noise, rain_noise, shift'
# python train.py -m 'se_resnext50_32x4d_027' -c 'epoch=30, mixup, noise, water_noise, walk_noise, shift'
# python train.py -m 'se_resnext50_32x4d_028' -c 'epoch=30, mixup, noise, water_noise, shift, m=(3, 10)'
# python train.py -m 'se_resnext50_32x4d_028' -c 'epoch=30, mixup, noise, water_noise, shift, m=(1, 8)'
# python train.py -m 'se_resnext50_32x4d_029' -c 'epoch=30, mixup, noise, water_noise, shift, ReduceLROnPlateau'
# python train.py -m 'se_resnext50_32x4d_030' -c 'epoch=30, mixup, noise, motorcycle_noise, water_noise, shift'
# python train.py -m 'se_resnext50_32x4d_031' -c 'epoch=30, mixup, motorcycle_noise, water_noise, shift'
# python train.py -m 'se_resnext50_32x4d_032' -c 'epoch=50, mixup, noise, water_noise, shift, ReduceLROnPlateau'
# python train.py -m 'se_resnext50_32x4d_033' -c 'epoch=30, mixup, noise, water_noise, shift, RandomBrightnessContrast'
# python train.py -m 'se_resnext50_32x4d_034' -c 'epoch=30, mixup, noise, water_noise, shift, drop=0.3'
# python train.py -m 'se_resnext50_32x4d_035' -c 'epoch=30, mixup, noise, water_noise, shift, drop=0.'
# python train.py -m 'se_resnext50_32x4d_036' -c 'epoch=30, mixup, noise, water_noise, shift, Mish'
# python train.py -m 'se_resnext50_32x4d_037' -c 'epoch=30, mixup, noise, water_noise, shift, mixup<0.8'
# python train.py -m 'se_resnext50_32x4d_038' -c 'epoch=30, noise, water_noise, shift'


# python train.py -m 'resnest50_001' -c 'epoch=30'
# python train.py -m 'resnest50_002' -c 'epoch=30, mixup, noise, water_noise, shift'
# python train.py -m 'resnest50_frelu_001' -c 'epoch=30'


# python train.py -m 'efficientnetb3_001' -c 'test'
# python train.py -m 'efficientnetb3_002' -c 'epoch=30'
# python train.py -m 'efficientnetb3_003' -c 'epoch=30, mixup, noise'


# python train.py -m 'efficientnetb4_001' -c 'epoch=30, mixup, noise'
# python train.py -m 'efficientnetb4_002' -c 'epoch=30, mixup, noise, seed=2021'
# python train.py -m 'efficientnetb4_003' -c 'epoch=30, mixup, noise, water_noise, shift'
# python train.py -m 'efficientnetb4_004' -c 'epoch=30, mixup, shift'
