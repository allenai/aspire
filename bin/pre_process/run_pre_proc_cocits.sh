#!/usr/bin/env bash
# Parse command line args.
model_name="cosentbert"
while getopts ":a:d:e:n:m:r:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        d) dataset=$OPTARG ;;
        e) experiment=$OPTARG ;;
        n) neg_method=$OPTARG ;;
        m) model_name=$OPTARG ;;
        r) run_name=$OPTARG ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done
# Make sure required arguments are passed.
if [[ "$action" == '' ]]; then
    echo "Must specify action (-a): filt_cocit_papers"
    exit 1
fi
if [[ "$dataset" == '' ]] ; then
    echo "Must specify dataset (-d): s2orccompsci/s2orcbiomed"
    exit 1
fi

# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
log_dir="$CUR_PROJ_DIR/logs/pre_process"
mkdir -p $log_dir

source_path="$CUR_PROJ_DIR/experiments/src/pre_process"
script_name="pre_proc_cocits"


if [[ "$action" == 'filt_cocit_papers' ]] || [[ "$action" == 'filt_cocit_sents' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}_logs.txt"
    run_path="$CUR_PROJ_DIR/datasets_raw/$dataset/"
    cmd="python3 -u $source_path/$script_name.py $action --run_path $run_path --dataset $dataset"
elif [[ "$action" == 'write_examples' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}-${model_name}_logs.txt"
    if [[ "$dataset" == 'treccovid' ]] || [[ "$dataset" == 'relish' ]]; then
      in_path="$CUR_PROJ_DIR/datasets_raw/s2orcbiomed/"
      sb_dataset="s2orcbiomed"
    else
      in_path="$CUR_PROJ_DIR/datasets_raw/$dataset/"
      sb_dataset="$dataset"
    fi
    out_path="$CUR_PROJ_DIR/datasets_proc/$dataset/$experiment"
    mkdir -p "$out_path"
    cmd="python3 -u $source_path/$script_name.py $action \
        --in_path $in_path \
        --out_path $out_path \
        --dataset $dataset --experiment $experiment --model_name $model_name"
    if [[ $run_name != '' ]]; then
      model_path="$CUR_PROJ_DIR/model_runs/$sb_dataset/$model_name/$run_name"
      cmd="$cmd --model_path $model_path"
    fi
fi

echo ${cmd} | tee ${log_file}
eval ${cmd} 2>&1 | tee -a ${log_file}
