# 循环次数
num_iterations=18

# 循环开始
for ((i=10; i<=$num_iterations; i++))
do
    echo "Iteration $i"
    CUDA_VISIBLE_DEVICES=1 python edit_test.py --config ../configs/edit/edit_seg.yaml --edit_layer "$i" --save_folder "edit_optim_seg_l$i"

done

echo "Loop finished."
