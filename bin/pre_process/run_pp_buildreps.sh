#!/usr/bin/env bash
use_log_file=false
# Parse command line args.
while getopts ":a:e:d:r:l" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        # Directory where the trained model should be read from
        r) run_name=$OPTARG ;;
        # Command line switch to use a log file and not print to stdout.
        l) use_log_file=true ;;
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
if [[ "$action" == '' ]] || [[ "$dataset" == '' ]]; then
    echo "Must specify action (-a):"
    echo "Must specify dataset (-d):"
    exit 1
fi

script_name="pre_proc_buildreps"
source_path="$CUR_PROJ_DIR/experiments/src/pre_process"
log_dir="$CUR_PROJ_DIR/logs/pre_process"
mkdir -p "$log_dir"

if [[ $dataset == 'csfcube' ]]; then
    data_path="$CUR_PROJ_DIR/datasets_raw/s2orccompsci"
    model_base_path="$CUR_PROJ_DIR/model_runs/s2orccompsci"
elif [[ $dataset == 'relish' ]] || [[ $dataset == 'treccovid' ]]; then
    data_path="$CUR_PROJ_DIR/datasets_raw/s2orcbiomed"
    model_base_path="$CUR_PROJ_DIR/model_runs/s2orcbiomed"
elif [[ $dataset == 'scidcite' ]] || [[ $dataset == 'scidcocite' ]] ||
  [[ $dataset == 'scidcoread' ]] || [[ $dataset == 'scidcoview' ]]; then
    data_path="$CUR_PROJ_DIR/datasets_raw/s2orcbiomed"
    model_base_path="$CUR_PROJ_DIR/model_runs/s2orcbiomed"
else
    data_path="$CUR_PROJ_DIR/datasets_raw/${dataset}"
    model_base_path="$CUR_PROJ_DIR/model_runs/s2orcbiomed"
fi

if [[ $action == 'build_reps' ]]; then
    run_path="${data_path}/${experiment}"
    log_file="${run_path}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
    mkdir -p "$run_path"
    # Base command line call for all models.
    cmd="python3 -um src.pre_process.$script_name  $action \
                --model_name $experiment \
                --data_path $data_path \
                --dataset $dataset \
                --run_path $run_path"
    # For all the trained models: cospecter, cosentbert, ictsentbert
    if [[ $run_name != '' ]]; then
        model_path="${model_base_path}/${experiment}/${run_name}"
        log_dir="${data_path}/${experiment}/${run_name}"
        mkdir -p "$log_dir"
        # Copy this over so that generating nearest docs have access to it to compute scores.
        cp "$model_path/run_info.json" "$log_dir"
        log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
        cmd="$cmd --model_path $model_path --run_name $run_name"
    fi
    if [[ $use_log_file == true ]]; then
      cmd="$cmd --log_fname $log_file"
    fi
fi

if [[ $use_log_file == true ]]; then
  eval $cmd
else
  eval $cmd 2>&1 | tee ${log_file}
fi
