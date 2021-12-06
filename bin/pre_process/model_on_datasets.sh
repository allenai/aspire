#!/usr/bin/env bash
# Runs scripts for embedding data with a model and for generating rankings for datasets.
# Parse command line args.
caching_socrer=false
while getopts ":e:d:r:c" opt; do
    case "$opt" in
        # This is the model_name.
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        r) run_name=$OPTARG ;;
        # Switch to say to use the caching scorer or to generate reps and then score.
        c) caching_socrer=true ;;
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

if [[ $dataset == 's2orcbiomed' ]]; then
  if [[ "$run_name" != '' ]]; then
    get_treccovid_reps="srun -p m40-long --mem=30GB --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
     -a build_reps -d treccovid -e $experiment -r $run_name -l"
    get_treccovid_neighs="srun -p m40-long --mem=30GB --gres=gpu:1 ./bin/pre_process/run_pp_gen_nearest.sh\
     -a rank_pool -d treccovid -e $experiment -r $run_name -l"
    get_relish_reps="srun -p m40-short --mem=30GB --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
     -a build_reps -d relish -e $experiment -r $run_name -l"
    get_relish_neighs="srun -p m40-short --mem=30GB --gres=gpu:1 ./bin/pre_process/run_pp_gen_nearest.sh\
     -a rank_pool -d relish -e $experiment -r $run_name -l"
  else
    get_treccovid_reps="srun -p m40-long --mem=30GB --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
         -a build_reps -d treccovid -e $experiment -l"
    get_treccovid_neighs="srun -p m40-long --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
     -a rank_pool -d treccovid -e $experiment -l"
    get_relish_reps="srun -p m40-short --mem=30GB --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
     -a build_reps -d relish -e $experiment -l"
    get_relish_neighs="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
     -a rank_pool -d relish -e $experiment -l"
  fi
  if [[ $caching_socrer == true ]]; then
    tc_seq="$get_treccovid_neighs -c"
    relish_seq="$get_relish_neighs -c"
    for cmd in "$tc_seq" "$relish_seq"; do
          echo "$cmd" && eval "$cmd" &
    done
  else
    tc_seq="$get_treccovid_reps && $get_treccovid_neighs"
    relish_seq="$get_relish_reps && $get_relish_neighs"
    echo "$tc_seq"
    eval "$tc_seq"
    echo "$relish_seq"
    eval "$relish_seq"
  fi
elif [[ $dataset == 's2orccompsci' ]]; then
  if [[ "$run_name" != '' ]]; then
    get_csfcube_reps="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
      -a build_reps -d csfcube -e $experiment -r $run_name -l"
    get_csfcube_neighs_b="srun --mem=30GB -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d csfcube -e $experiment -r $run_name -f background -l"
    get_csfcube_neighs_m="srun --mem=30GB -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d csfcube -e $experiment -r $run_name -f method -l"
    get_csfcube_neighs_r="srun --mem=30GB -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d csfcube -e $experiment -r $run_name -f result -l"
  else
    get_csfcube_reps="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
          -a build_reps -d csfcube -e $experiment -l"
    get_csfcube_neighs_b="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d csfcube -e $experiment -f background -l"
    get_csfcube_neighs_m="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d csfcube -e $experiment -f method -l"
    get_csfcube_neighs_r="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d csfcube -e $experiment -f result -l"
  fi
  if [[ $caching_socrer == false ]]; then
    echo "$get_csfcube_reps"
    eval "$get_csfcube_reps"
    # Run each one in parallel.
    for cmd in "$get_csfcube_neighs_b" "$get_csfcube_neighs_m" "$get_csfcube_neighs_r"; do
      echo "$cmd" && eval "$cmd" &
    done
  else
    # Run each one in parallel.
    for cmd in "$get_csfcube_neighs_b" "$get_csfcube_neighs_m" "$get_csfcube_neighs_r"; do
      echo "$cmd -c" && eval "$cmd -c" &
    done
  fi
elif [[ $dataset == 'scidocs' ]]; then
  if [[ "$run_name" != '' ]]; then
    get_reps_cite="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
              -a build_reps -d scidcite -e $experiment -r $run_name -l"
    get_neighs_cite="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcite -e $experiment -r $run_name -l"
    get_reps_cocite="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
              -a build_reps -d scidcocite -e $experiment -r $run_name -l"
    get_neighs_cocite="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcocite -e $experiment -r $run_name -l"
    get_reps_coread="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
              -a build_reps -d scidcoread -e $experiment -r $run_name -l"
    get_neighs_coread="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcoread -e $experiment -r $run_name -l"
    get_reps_coview="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
              -a build_reps -d scidcoview -e $experiment -r $run_name -l"
    get_neighs_coview="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcoview -e $experiment -r $run_name -l"
  else
    get_reps_cite="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
          -a build_reps -d scidcite -e $experiment -l"
    get_neighs_cite="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcite -e $experiment -l"
    get_reps_cocite="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
              -a build_reps -d scidcocite -e $experiment -l"
    get_neighs_cocite="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcocite -e $experiment -l"
    get_reps_coread="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
              -a build_reps -d scidcoread -e $experiment -l"
    get_neighs_coread="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcoread -e $experiment -l"
    get_reps_coview="srun -p m40-short --gres=gpu:1 ./bin/pre_process/run_pp_buildreps.sh\
              -a build_reps -d scidcoview -e $experiment -l"
    get_neighs_coview="srun -p m40-short --gres=gpu:1 --mem=30GB ./bin/pre_process/run_pp_gen_nearest.sh\
      -a rank_pool -d scidcoview -e $experiment -l"
  fi
  if [[ $caching_socrer == false ]]; then
    # Run each one in parallel.
    for cmd in "$get_reps_cite" "$get_reps_cocite" "$get_reps_cocite" "$get_reps_coread"; do
      echo "$cmd" && eval "$cmd" &
    done
    # Wait for the above step to complete.
    read -rsp $'Check build reps and press any key to continue on completion...\n' -n1 key
    # Run each one in parallel.
    for cmd in "$get_neighs_cite" "$get_neighs_cocite" "$get_neighs_coread" "$get_neighs_coview"; do
      echo "$cmd" && eval "$cmd" &
    done
  else
    # Run each one in parallel.
    for cmd in "$get_neighs_cite" "$get_neighs_cocite" "$get_neighs_coread" "$get_neighs_coview"; do
          echo "$cmd -c" && eval "$cmd -c" &
    done
  fi
fi

wait
