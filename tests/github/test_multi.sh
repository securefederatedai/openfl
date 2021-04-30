#!/bin/bash

output='./result1.md'
touch $output

printf "
| FRAMEWORK | COL_NUMBER | DATASET_MULT | IS_MULTI | WORKERS | GPU | MODE | REAL TIME | USER TIME | SYS TIME |
|-----------|------------|--------------|----------|---------|-----|------|-----------|-----------|----------|
" >> $output

echo "START"
function build_command() {
  framework=$1
  col_num=$2
  dataset_mult=$3
  is_multi=$4
  is_gpu=$5
  mode=$6
  max_workers=$7
  rounds_to_train=$8
  if [ "$framework" = "tf" ];
  then
    script="./python_native_tf.py"
  fi
  if [ "$framework" = "torch" ];
  then
    script="./python_native_torch.py"
  fi

  if [ "$is_multi" = "1" ];
  then
    is_multi="--is_multi"
  else
    is_multi=
  fi

  result=$({ time -p timeout 20m python "$script" --rounds_to_train="$rounds_to_train" --dataset_multiplier="$dataset_mult" --collaborators_amount="$col_num" --mode=\"$mode\" $is_multi > /dev/null 2>&1; } 2>&1 | tee /dev/tty)
  echo "$result"
}

function save_result() {
  framework=$1
  col_num=$2
  dataset_mult=$3
  is_multi=$4
  is_gpu=$5
  mode=$6
  max_workers=$7
  rounds_to_train=$8
  time=$9

  readarray foo <<<"$time"

  real_time_str=${foo[0]}
  user_time_str=${foo[1]}
  sys_time_str=${foo[2]}
  read field1 real_time <<< ${real_time_str}
  read field1 user_time <<< ${user_time_str}
  read field1 sys_time <<< ${sys_time_str}

  echo "$framework" "$col_num" "$dataset_mult" "$is_multi" "$is_gpu" "$mode" "$max_workers" "$real_time_str" "$user_time_str" "$sys_time_str" "\n"
  printf "| %s " "$framework" >> $output


  printf "| $col_num " >> $output
  printf "| $dataset_mult " >> $output
  if [ "$is_multi" = "1" ];
  then
    printf "| 1 " >> $output
  else
    printf "| 0 " >> $output
  fi

  printf '| %s ' "$max_workers" >> $output

  if [ "$is_gpu" = "1" ];
  then
    printf "| 1 " >> $output
  else
    printf "| 0 " >> $output
  fi

  printf "| $mode " >> $output


  printf "| $real_time " >> $output
  printf "| $user_time " >> $output
  printf "| $sys_time |\n" >> $output
}


FrameworkArray=("torch")
IsGPUArray=("1" "0")
ColNumArray=("10" "5" "2" "1" )
IsMultiArray=("1" "0")
ModeArray=("p=c" "p=c*r")
RoundsToTrain=("3")
dataset_mult="15"
Retry=1
total=46
c=0

for framework in "${FrameworkArray[@]}"; do
  for is_gpu in "${IsGPUArray[@]}"; do
    if [ "$is_gpu" = "1" ];
    then
      export CUDA_VISIBLE_DEVICES=0
    else
      export CUDA_VISIBLE_DEVICES=
    fi
    for col_num in "${ColNumArray[@]}"; do
      for is_multi in "${IsMultiArray[@]}"; do
        if [ "$is_multi" = "0" ];
        then
          ModeArray=("p=c*r")
        else
          ModeArray=("p=c" "p=c*r")
        fi
        for mode in "${ModeArray[@]}"; do
          if [ "$is_multi" = "1" ];
          then
            if [ "$framework" = "torch" ] && [ "$is_gpu" = "0" ];
            then

              case "$col_num" in
                "1" ) WorkerArray=("1");;
                "2" ) WorkerArray=("1" "2");;
                "5" ) WorkerArray=("1" "2");;
                "10" ) WorkerArray=("1" "2");;
                *)  WorkerArray=("1");;
              esac
            else
              case "$col_num" in
                "1" ) WorkerArray=("0");;
                "2" ) WorkerArray=("0" "1");;
                "5" ) WorkerArray=("0" "1" "2" "4");;
                "10" ) WorkerArray=("0" "1" "2" "5" "9");;
                *)  WorkerArray=("0");;
              esac
            fi
          else
            WorkerArray=("0")
          fi
          for workers in "${WorkerArray[@]}"; do
            for rounds in "${RoundsToTrain[@]}"; do
              for (( i=1; i<=$Retry; i++ )); do
                c=$((c + 1))
                echo $c/$total
                run_time=$(build_command "$framework" "$col_num" "$dataset_mult" "$is_multi" "$is_gpu" "$mode" "$workers" "$rounds")
                save_result "$framework" "$col_num" "$dataset_mult" "$is_multi" "$is_gpu" "$mode" "$workers" "$rounds" "$run_time"
              done
            done
          done
        done
      done
    done
  done
done
