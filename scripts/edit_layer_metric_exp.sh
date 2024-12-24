# 循环次数
num_iterations=18

echo "kp"
for ((i=1; i<=num_iterations; i++))
do
  echo "$i"
  CUDA_VISIBLE_DEVICES=1 python test_metric.py --mode "kp" --edit_images_path "/home/liudongxv/workplace/GANInverter-dev/test_edit/edit/e4e/edit_optim_kp_l$i"
done

