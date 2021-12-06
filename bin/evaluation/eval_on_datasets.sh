#!/usr/bin/env bash
# Parse command line args.
while getopts ":e:d:r:c:" opt; do
    case "$opt" in
        # This is the model_name.
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        r) run_name=$OPTARG ;;
        c) comet_id=$OPTARG ;;
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
    tc_eval="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
     -d treccovid -e $experiment -r $run_name"
    relish_eval="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
     -d relish -e $experiment -r $run_name"
  else
    tc_eval="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
         -d treccovid -e $experiment"
    relish_eval="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
       -d relish -e $experiment"
  fi
  if [[ "$comet_id" != '' ]]; then
    tc_eval="$tc_eval -c $comet_id"
    relish_eval="$relish_eval -c $comet_id"
  fi
  eval "$tc_eval"
  read -rsp $'Press any key to continue...\n' -n1 key
  eval "$relish_eval"
elif [[ $dataset == 's2orccompsci' ]]; then
  if [[ "$run_name" != '' ]]; then
    cs_eval_b="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
      -d csfcube -e $experiment -r $run_name -f background"
    cs_eval_m="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d csfcube -e $experiment -r $run_name -f method"
    cs_eval_r="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d csfcube -e $experiment -r $run_name -f result"
    cs_eval_a="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d csfcube -e $experiment -r $run_name -f all"
  else
    cs_eval_b="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
          -d csfcube -e $experiment -f background"
    cs_eval_m="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d csfcube -e $experiment -f method"
    cs_eval_r="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d csfcube -e $experiment -f result"
    cs_eval_a="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d csfcube -e $experiment -f all"
  fi
  if [[ "$comet_id" != '' ]]; then
      cs_eval_b="$cs_eval_b -c $comet_id"
      cs_eval_m="$cs_eval_m -c $comet_id"
      cs_eval_r="$cs_eval_r -c $comet_id"
      cs_eval_a="$cs_eval_a -c $comet_id"
  fi
  eval "$cs_eval_b"
  read -rsp $'Press any key to continue...\n' -n1 key
  eval "$cs_eval_m"
  read -rsp $'Press any key to continue...\n' -n1 key
  eval "$cs_eval_r"
  read -rsp $'Press any key to continue...\n' -n1 key
  eval "$cs_eval_a"
elif [[ $dataset == 'scidocs' ]]; then
  if [[ "$run_name" != '' ]]; then
    scid_eval_c="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
      -d scidcite -e $experiment -r $run_name"
    scid_eval_cc="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d scidcocite -e $experiment -r $run_name"
    scid_eval_cr="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d scidcoread -e $experiment -r $run_name"
    scid_eval_cv="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d scidcoview -e $experiment -r $run_name"
  else
    scid_eval_c="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
          -d scidcite -e $experiment"
    scid_eval_cc="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d scidcocite -e $experiment"
    scid_eval_cr="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d scidcoread -e $experiment"
    scid_eval_cv="srun --mem=30GB ./bin/evaluation/run_rank_eval.sh -a eval_pool_ranking\
        -d scidcoview -e $experiment"
  fi
  if [[ "$comet_id" != '' ]]; then
    scid_eval_c="$scid_eval_c -c $comet_id"
    scid_eval_cc="$scid_eval_cc -c $comet_id"
    scid_eval_cr="$scid_eval_cr -c $comet_id"
    scid_eval_cv="$scid_eval_cv -c $comet_id"
  fi
  eval "$scid_eval_c"
  read -rsp $'Press any key to continue...\n' -n1 key
  eval "$scid_eval_cc"
  read -rsp $'Press any key to continue...\n' -n1 key
  eval "$scid_eval_cr"
  read -rsp $'Press any key to continue...\n' -n1 key
  eval "$scid_eval_cv"
fi
