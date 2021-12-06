#!/usr/bin/env bash
# Parse command line args.
while getopts ":a:d:e:f:r:x:y:c:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        d) dataset=$OPTARG ;;
        e) experiment=$OPTARG ;;
        f) facet=$OPTARG ;;
        r) run_name=$OPTARG ;;
        c) comet_exp_key=$OPTARG ;;
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

source_path="$CUR_PROJ_DIR/experiments/src/evaluation"
script_name="ranking_eval"
if [[ "$action" == 'eval_pool_ranking' ]]; then
    if [[ "$dataset" == 'relish' ]] || [[ "$dataset" == 'treccovid' ]]; then
      data_path="$CUR_PROJ_DIR/datasets_raw/s2orcbiomed"
      run_path="$CUR_PROJ_DIR/datasets_raw/s2orcbiomed"
    elif [[ $dataset == 'scidcite' ]] || [[ $dataset == 'scidcocite' ]] ||
      [[ $dataset == 'scidcoread' ]] || [[ $dataset == 'scidcoview' ]]; then
          data_path="$CUR_PROJ_DIR/datasets_raw/s2orcscidocs"
          run_path="$CUR_PROJ_DIR/datasets_raw/s2orcscidocs"
    else
      data_path="$CUR_PROJ_DIR/datasets_raw/s2orccompsci"
      run_path="$CUR_PROJ_DIR/datasets_raw/s2orccompsci"
    fi
    log_file="${run_path}/${script_name}-${action}-${dataset}-${experiment}-eval_logs.txt"
    cmd="python3 -u $source_path/$script_name.py $action \
    --data_path $data_path --experiment $experiment --dataset $dataset"
    if [[ $run_name != '' ]]; then
        run_path="$data_path/$experiment/$run_name"
        log_file="${run_path}/${script_name}-${action}-${dataset}-${experiment}-eval_logs.txt"
        cmd="$cmd --run_path $run_path --run_name $run_name"
    fi
    if [[ $facet != '' ]]; then
        log_file="${run_path}/${script_name}-${action}-${dataset}-${experiment}-${facet}-eval_logs.txt"
        cmd="$cmd --facet $facet"
    fi
    if [[ $comet_exp_key != '' ]]; then
        cmd="$cmd --comet_exp_key $comet_exp_key"
    fi
fi


echo "$cmd" | tee ${log_file}
eval "$cmd" 2>&1 | tee -a ${log_file}
