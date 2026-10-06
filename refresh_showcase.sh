L=/shared/ssd/logs/b-r-singh1/results/cv
while read folder stem; do
  src=$L/$stem; dst=results_showcase/$folder
  [ -d "$src" ] && [ -d "$dst" ] || { echo "MISSING $folder / $stem"; continue; }
  n=0
  for f in $(ls $src); do
    [ "$f" = README.md ] && continue
    n=$((n+1))
    [ "$APPLY" = 1 ] && cp -f "$src/$f" "$dst/$f"
  done
  echo "$folder <- $stem : $n files"
done <<'LIST'
biomedclip_frozen_224 biomedclip_r224
celldino_frozen_224 celldino_r224
celldino_frozen_448 celldino_r448
clip_frozen_224 clip_r224
dinov2_frozen_224 dino_r224
dinov2_frozen_448 dino_r448
dinov2_large_frozen_224 dino_large_r224
dinov2_large_frozen_448 dino_large_r448
dinov2_small_frozen_224 dino_small_r224
dinov3_frozen_224 dinov3_r224
dinov3_frozen_448 dinov3_r448
LIST
