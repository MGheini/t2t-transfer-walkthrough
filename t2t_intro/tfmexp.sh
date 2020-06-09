#!/usr/bin/env bash

set -e


# A single run of tensor2tensor for vista cluster
# Author = Jibiao Shen (jibiaosh@isi.edu)
# Date = June 21, 2018

source ~/.bashrc ""
source activate t2tenv
# source activate tfenv

date

while getopts g:h:D:n:d:x:l:w:f:s:a:b:v:t:p:rN option
do
case ${option}
in
l) SRCLAN=${OPTARG};;
r) REVERSE=1;;
d) DATA=${OPTARG};;
x) PRFX=${OPTARG};;
w) WORKPATH=./${OPTARG};;
f) PARENT=${OPTARG};;
n) DIV=${OPTARG};;
s) TOTALSTEPS=${OPTARG};;
b) BATCHSIZE="--batch_size ${OPTARG}"
   BSIZE=${OPTARG};;
v) VOCABSIZE="--vocab_size ${OPTARG}"
   VSIZE=${OPTARG};;
t) RESTOKENS="--res_tokens ${OPTARG}";;
N) NEWPROBLEM=1;;
D) EXTRADECODE=1
   DECODESRC=${OPTARG};;
p) PROBLEM=${OPTARG};;
h) CUS_HPARAMS="--hparams ${OPTARG}";;
g) NUM_GPUS=${OPTARG};;
esac
done

WORKPATH=${WORKPATH:-./$SRCLAN"_"$BSIZE"_"$VSIZE"_"$(date +'%d-%H-%M-%S')}

if [ ! -d $WORKPATH ]
then
    mkdir -p $WORKPATH
fi

SRCLAN=${SRCLAN:-ben}
TGTLAN=en
DATA=${DATA:-/nas/material/users/jonmay/jibiaosh/tfmexp/elisa/"$SRCLAN"-en}
PRFX=${PRFX:-elisa}

SRC=$SRCLAN
TGT=en

if [ "$REVERSE" == "1" ]
then
    TGT=$SRCLAN
    SRC=en
fi

DIV=${DIV:-2}
SRCTRAIN=$DATA/"$PRFX".train_"$DIV".$SRC
TGTTRAIN=$DATA/"$PRFX".train_"$DIV".$TGT
# SRCTRAIN=$DATA/elisa.train_"$DIV".trimmed.$SRC
# TGTTRAIN=$DATA/elisa.train_"$DIV".trimmed.$TGT
SRCTUNE=$DATA/"$PRFX".tune.$SRC
TGTTUNE=$DATA/"$PRFX".tune.$TGT
SRCTEST=$DATA/"$PRFX".test.$SRC
TGTTEST=$DATA/"$PRFX".test.$TGT
#SRCTESTWR=$DATA/elisa.testwr.$SRC
#TGTTESTWR=$DATA/elisa.testwr.$TGT
#IDXTESTWR=$DATA/elisa.testwr.indices
#SAMPLEPATH=$DATA/testwr

for i in $SRCTRAIN $TGTTRAIN $SRCTUNE $TGTTUNE $SRCTEST $TGTTEST
do
	[ -f $i.tok ] || perl /home/nlg-05/gheini/libs/OpenNMT-py/tools/tokenizer.perl < $i > $i.tok
done

#TOKENIZER=~/t2twd/ulftok/ulf-eng-tok.sh

# Tensor 2 tensor

# PROBLEM=${PROBLEM:-translate_src_tgt32k}
PROBLEM=${PROBLEM:-translate_srctgt}
MODEL=transformer
HPARAMS=transformer_base
BEAM_SIZE=4
ALPHA=0.6
NUM_GPUS=${NUM_GPUS:-1}
TRAIN_STEPS=$( expr ${TOTALSTEPS:-128000} / $NUM_GPUS )

echo TRAIN_STEPS:$TRAIN_STEPS

if [ "$NEWPROBLEM" == "1" ]
then
    while read prob hps; do
        PROBLEM=$prob
        HPARAMS=${hps:-$HPARAMS}
    done <<< $(python problem_gen.py --save_problem $PROBLEM $BATCHSIZE $VOCABSIZE $CUS_HPARAMS $RESTOKENS)
else
    PROBLEM=$(python name_conv.py $PROBLEM)
fi

#
for i in data model results decodes
do
    if [ ! -d $WORKPATH/$i ]
    then
        mkdir -p $WORKPATH/$i
    fi
done

WORKPATH=$(realpath "$WORKPATH")

[ -f $WORKPATH/data/train.tok.src ] || ln -s $(realpath $SRCTRAIN.tok) $WORKPATH/data/train.tok.src
[ -f $WORKPATH/data/train.tok.tgt ] || ln -s $(realpath $TGTTRAIN.tok) $WORKPATH/data/train.tok.tgt

[ -f $WORKPATH/data/tune.tok.src ] || ln -s $(realpath $SRCTUNE.tok) $WORKPATH/data/tune.tok.src
[ -f $WORKPATH/data/tune.tok.tgt ] || ln -s $(realpath $TGTTUNE.tok) $WORKPATH/data/tune.tok.tgt

[ -f $WORKPATH/data/test.tok.src ] || ln -s $(realpath $SRCTEST.tok) $WORKPATH/data/test.tok.src
[ -f $WORKPATH/data/test.tok.tgt ] || ln -s $(realpath $TGTTEST.tok) $WORKPATH/data/test.tok.tgt

#[ -f $WORKPATH/data/testwr.tok.src ] || ln -s $(realpath SRCTESTWR.tok) $WORKPATH/data/testwr.tok.src

# this is a hack from Thamme Gowda (tg@isi.edu) -- to skip remote corpus download -- because our corpus is local
[ -f $WORKPATH/data/dummy.txt ] || touch $WORKPATH/data/dummy.txt

#Utils from Thamme Gowda (tg@isi.edu)
function log {
    printf "`date '+%Y-%m-%d %H:%M:%S'`:: $1\n" >> $WORKPATH/job.log
}
function decode {
    # accepts two args: <src-file> <out-file>
    FROM=$1
    TO=$2
    cmd="t2t-decoder \
        --data_dir=$WORKPATH/data \
        --problem=$PROBLEM \
        --model=$MODEL \
        --hparams_set=$HPARAMS \
        --output_dir=$WORKPATH/model \
        --decode_hparams=\"beam_size=$BEAM_SIZE,batch_size=2,alpha=$ALPHA\" \
        --decode_from_file=$FROM \
        --decode_to_file=$TO"
    log "$cmd"
    eval "$cmd"

}
function score {
    # usage: score <output> <reference>
    out=$1
    ref=$2
    # DETOK=~/libs/OpenNMT-py/tools/detokenize.perl
    DETOK=/home/nlg-05/gheini/scripts/detokenizer.perl
    BLEU=/home/nlg-05/gheini/libs/OpenNMT-py/tools/multi-bleu-detok.perl

    cat $out | $DETOK | sed 's/ @\([^@]\+\)@ /\1/g' > $out.detok
    cat $out.detok | $BLEU  $ref > $out.detok.tc.bleu
    cat $out.detok | $BLEU -lc $ref > $out.detok.lc.bleu
}


# copy this script for reproducibility
cp "${BASH_SOURCE[0]}"  $WORKPATH/job.sh.bak

#Data generation
if [[ -f $WORKPATH/._SUCCESS_DATAGEN ]]; then
    log "Step : Skipping data processing... Files already exists"
else
    cmd="t2t-datagen --data_dir=$WORKPATH/data --tmp_dir=$WORKPATH/data --problem=$PROBLEM"
    # cmd="~/libs/tensor2tensor/tensor2tensor/bin/t2t-datagen --data_dir=$WORKPATH/data --tmp_dir=$WORKPATH/data --problem=$PROBLEM"
    log "$cmd"
    if eval "$cmd"; then
        touch $WORKPATH/._SUCCESS_DATAGEN
    else
        log 'Failed... exiting'
        exit 1
    fi
fi

#Training
if [[ -f "$WORKPATH/._SUCCESS_TRAIN" ]]; then
    log "Step : Skipping Training... "
else
    date
    log "Step : Starting trainer"
    # If you run out of memory, add --hparams='batch_size=1024'.
    if [ "$PARENT" ]; then
        cmd="t2t-trainer --data_dir=$WORKPATH/data \
            --problem=$PROBLEM --model=$MODEL \
            --hparams_set=$HPARAMS --output_dir=$WORKPATH/model \
            --train_steps=$TRAIN_STEPS \
            --worker_gpu=$NUM_GPUS \
            --warm_start_from=$PARENT"
    else
        cmd="t2t-trainer --data_dir=$WORKPATH/data \
            --problem=$PROBLEM --model=$MODEL \
            --hparams_set=$HPARAMS --output_dir=$WORKPATH/model \
            --train_steps=$TRAIN_STEPS \
            --worker_gpu=$NUM_GPUS"
    fi
    log "$cmd"
    if eval "$cmd"; then
        touch "$WORKPATH/._SUCCESS_TRAIN"
    else
        log 'Failed....'
        exit 1
    fi
fi


:<<END
printf "$TGTTEST test
$TGTTUNE tune
" | while read ref split; do
    echo "Decoding and scoring $split"

    [[ -f $WORKPATH/results/$split.out ]] || decode $WORKPATH/data/$split.tok.src $WORKPATH/results/$split.out
    score $WORKPATH/results/$split.out $ref
done
END


for i in tune test
do decode $DATA/"$PRFX".$i.$SRC.tok $WORKPATH/results/$i.out.tok
done

score $WORKPATH/results/tune.out.tok $TGTTUNE
score $WORKPATH/results/test.out.tok $TGTTEST
#score $WORKPATH/results/testwr.out.tok $TGTTESTWR

#python wrgen.py $IDXTESTWR $WORKPATH/results/testwr.out.tok.detok $WORKPATH/

#for i in $(seq 0 999)
#do
#perl ~/OpenNMT/tools/multi-bleu-detok.perl $SAMPLEPATH/elisa.test.$i.en < $WORKPATH/test.$i.pred >> $WORKPATH/wrresults.txt
#done

#rm $WORKPATH/test.*.pred

#source activate python3
#python wranalyze.py $WORKPATH/wrresults.txt > $WORKPATH/analysis.txt
#source deactivate

date
