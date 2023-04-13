while [ 1 ]
do
    nu=$(ps -ef | grep train_imagenet.py | grep -v grep -c)
    if [ $nu = 0 ]
    then
        CUDA_VISIBLE_DEVICES=0,1,3,7 nohup python train_imagenet.py --generator_type test --discriminator_type test --data_dir /raid/zhangwuxia/transform-imagenet/imagenet > wuxia.log 2>&1 &
        echo restart
    else
        echo testing
    fi
    sleep 30
done
