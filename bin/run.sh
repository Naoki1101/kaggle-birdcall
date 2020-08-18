cd ../src

# python train.py -m 'resnet18_001' -c 'test'
python train.py -m 'resnet18_002' -c 'epoch=50'

# python train.py -m 'resnet34_001' -c 'epoch=50'

# python train.py -m 'resnet50_001' -c 'epoch=50'


cd ../
git add -A
git commit -m '...'
git push origin master