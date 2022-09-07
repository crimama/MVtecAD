unzip '/content/drive/MyDrive/데이터공유폴더/open.zip'
unzip open/train.zip 
unzip open/test.zip
mv train ./open
mv test ./open 

pip install timm 

git clone https://github.com/crimama/MVtecAD.git 

cd './MVtecAD'
