for mode in kp seg
do
  echo "$mode"
  for ((i=3; i<=$num_iterations; i++))
  do
    echo "$layer"
    # shellcheck disable=SC2154
    CUDA_VISIBLE_DEVICES=1 python edit_test.py --config "../configs/edit/edit_$mode.yaml" --edit_layer "$layer" --save_folder "edit_optim_$mode_l$layer_total"
  done
done