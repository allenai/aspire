#!/usr/bin/env bash
# Parse command line args.
use_log_file=false
caching_scorer=false
while getopts ":a:e:r:d:f:lct" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        # Trained model basepath name.
        r) run_name=$OPTARG ;;
        f) facet=$OPTARG ;;
        # Command line switch to use a log file and not print to stdout.
        l) use_log_file=true ;;
        # Switch to say to use the caching scorer or to generate reps and then score.
        c) caching_scorer=true ;;
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
    echo "Must specify action (-a)"
    exit 1
fi
if [[ "$dataset" == '' ]] ; then
    echo "Must specify dataset (-d)"
    exit 1
fi

# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
log_dir="$CUR_PROJ_DIR/logs/pre_process"
mkdir -p $log_dir

source_path="$CUR_PROJ_DIR/experiments/src/pre_process"
script_name="pp_gen_nearest"


if [[ "$action" == 'rank_pool' ]]; then
    if [[ $dataset == 'csfcube' ]]; then
        root_path="$CUR_PROJ_DIR/datasets_raw/s2orccompsci"
        model_base_path="$CUR_PROJ_DIR/model_runs/s2orccompsci"
    elif [[ $dataset == 'scidcite' ]] || [[ $dataset == 'scidcocite' ]] ||
     [[ $dataset == 'scidcoread' ]] || [[ $dataset == 'scidcoview' ]]; then
        root_path="$CUR_PROJ_DIR/datasets_raw/s2orcscidocs"
        model_base_path="$CUR_PROJ_DIR/model_runs/s2orcscidocs"
    elif [[ $dataset == 'relish' ]] || [[ $dataset == 'treccovid' ]] && [[ $tcr_train == false ]]; then
        root_path="$CUR_PROJ_DIR/datasets_raw/s2orcbiomed"
        model_base_path="$CUR_PROJ_DIR/model_runs/s2orcbiomed"
    else
        root_path="$CUR_PROJ_DIR/datasets_raw/${dataset}/"
    fi
    log_file="${root_path}/${experiment}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
    cmd="python3 -um src.pre_process.$script_name $action \
      --root_path $root_path --dataset $dataset --rep_type $experiment"
    if [[ $run_name != '' ]]; then
        model_path="${model_base_path}/${experiment}/${run_name}"
        log_dir="${root_path}/${experiment}/${run_name}"
        mkdir -p "$log_dir"
        # Copy this over so that generating nearest docs have access to it to compute scores.
        # Do it in update mode only though.
        cp -u "$model_path/run_info.json" "$log_dir"
        log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}_logs.txt"
        cmd="$cmd --run_name $run_name --model_path $model_path"
    fi
    if [[ $facet != '' ]]; then
        log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}-${facet}_logs.txt"
        cmd="$cmd --facet $facet"
    fi
    if [[ $use_log_file == true ]]; then
      cmd="$cmd --log_fname $log_file"
    fi
    if [[ $caching_scorer == true ]]; then
      cmd="$cmd --caching_scorer"
    fi
else
    echo "Unknown action."
    exit 1
fi

if [[ $use_log_file == true ]]; then
  eval $cmd
else
  eval $cmd 2>&1 | tee ${log_file}
fi