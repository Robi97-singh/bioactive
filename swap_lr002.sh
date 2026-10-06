L=/shared/ssd/logs/b-r-singh1
B=$L/backup_lr005_before_lr002_swap
[ "$APPLY" = 1 ] && mkdir -p $B/results $B/checkpoints
nr=0; nc=0
for src in $L/rerun_all_lr002 $L/rerun_bf_lr002; do
  for d in $src/results/bioact_*; do
    n=$(basename $d); nr=$((nr+1))
    if [ "$APPLY" = 1 ]; then
      [ -d $L/results/$n ] && mv $L/results/$n $B/results/$n
      cp -a $d $L/results/$n
    fi
  done
  for f in $src/checkpoints/*head_metrics.json; do
    n=$(basename $f); nc=$((nc+1))
    if [ "$APPLY" = 1 ]; then
      [ -f $L/checkpoints/$n ] && cp -a $L/checkpoints/$n $B/checkpoints/$n
      cp -f $f $L/checkpoints/$n
    fi
  done
done
echo "result dirs: $nr | checkpoint jsons: $nc | APPLY=${APPLY:-0}"
