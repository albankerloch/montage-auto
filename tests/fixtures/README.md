# Fixtures vidéo

`rushes/` contient trois clips de 24 s générés par ffmpeg (8 scènes de 3 s
chacune, motifs et teintes distincts pour que la détection de scènes se
déclenche). Ils servent aux tests de bout en bout dans `test_pipeline_e2e.py`,
qui vérifient la sémantique du cache, l'export NLE et le rendu.

Les régénérer :

```bash
for i in 0 1 2; do
  : > /tmp/list_$i.txt
  for s in 0 1 2 3 4 5 6 7; do
    n=$(( i*8 + s ))
    case $(( n % 4 )) in
      0) SRC="testsrc=size=320x180:rate=25:duration=3";;
      1) SRC="smptebars=size=320x180:rate=25:duration=3";;
      2) SRC="testsrc2=size=320x180:rate=25:duration=3";;
      3) SRC="rgbtestsrc=size=320x180:rate=25:duration=3";;
    esac
    ffmpeg -y -loglevel error -f lavfi -i "$SRC" \
      -vf "drawtext=text='R${i}S${s}':fontsize=44:x=10:y=10,hue=h=$(( n * 43 ))" \
      -c:v libx264 -pix_fmt yuv420p -g 12 /tmp/p_${i}_${s}.mp4
    echo "file '/tmp/p_${i}_${s}.mp4'" >> /tmp/list_$i.txt
  done
  ffmpeg -y -loglevel error -f concat -safe 0 -i /tmp/list_$i.txt -c copy \
    tests/fixtures/rushes/rush_$i.mp4
done
```

Les tests se skippent proprement si le dossier est vide.
