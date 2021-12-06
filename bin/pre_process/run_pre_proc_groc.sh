#!/usr/bin/env bash
# Parse command line args.
while getopts ":a:e:d:s:n:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        s) special=$OPTARG ;;
        n) neg_method=$OPTARG ;;
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
if [[ "$action" != 'get_batch_pids' ]] && [[ "$dataset" == '' ]] ; then
    echo "Must specify dataset (-d)"
    exit 1
fi

# $CUR_PROJ_ROOT is a environment variable; manually set outside of the script.
log_dir="$CUR_PROJ_DIR/logs/pre_process"
mkdir -p $log_dir

source_path="$CUR_PROJ_DIR/experiments/src/pre_process"
script_name="pre_proc_gorc"


if [[ "$action" == 'filter_by_hostserv' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}_logs.txt"
    raw_meta_path="$CUR_PROJ_DIR/datasets_raw/gorc/gorc/20190928/metadata"
    filt_meta_path="$CUR_PROJ_DIR/datasets_raw/gorc/hostservice_filt/"
    cmd="python3 -u $source_path/$script_name.py $action \
    --raw_meta_path $raw_meta_path --filt_meta_path $filt_meta_path --dataset $dataset"
elif [[ "$action" == 'gather_by_hostserv' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}_logs.txt"
    in_meta_path="$CUR_PROJ_DIR/datasets_raw/gorc/hostservice_filt/"
    raw_data_path="$CUR_PROJ_DIR/datasets_raw/gorc/gorc/20190928/papers"
    cmd="python3 -u $source_path/$script_name.py $action \
    --in_meta_path $in_meta_path --raw_data_path $raw_data_path --dataset $dataset"
elif [[ "$action" == 'get_batch_pids' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}_logs.txt"
    in_path="$CUR_PROJ_DIR/datasets_raw/gorc/gorc/20190928/metadata"
    out_path="$CUR_PROJ_DIR/datasets_raw/gorc"
    mkdir -p "$out_path"
    cmd="python3 -u $source_path/$script_name.py $action --in_path $in_path --out_path $out_path"
elif [[ "$action" == 'gather_from_citationnw' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}_logs.txt"
    root_path="$CUR_PROJ_DIR/datasets_raw/s2orc/"
    cmd="python3 -u $source_path/$script_name.py $action \
    --root_path $root_path --dataset $dataset"
elif [[ "$action" == 'filter_area_citcontexts' ]] || [[ "$action" == 'gather_area_cocits' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}_logs.txt"
    root_path="$CUR_PROJ_DIR/datasets_raw/s2orc/"
    # Sloppily using dataset as the area.
    cmd="python3 -u $source_path/$script_name.py $action --root_path $root_path --area $dataset"
elif [[ "$action" == 'gather_filtcocit_corpus' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}_logs.txt"
    in_meta_path="$CUR_PROJ_DIR/datasets_raw/s2orc/hostservice_filt/"
    raw_data_path="$CUR_PROJ_DIR/datasets_raw/s2orc/gorc/20190928/papers"
    root_path="$CUR_PROJ_DIR/datasets_raw/s2orc/"
    out_path="$CUR_PROJ_DIR/datasets_raw/$dataset"
    mkdir -p $out_path
    cmd="python3 -u $source_path/$script_name.py $action --out_path $out_path\
    --in_meta_path $in_meta_path --raw_data_path $raw_data_path --dataset $dataset\
     --root_path $root_path"
else
    echo "Unknown action."
    exit 1
fi

echo ${cmd} | tee ${log_file}
eval ${cmd} 2>&1 | tee -a ${log_file}