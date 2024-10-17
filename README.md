# humit-språkidentifikator

Dette verktøyet identifiserer skriftspråket som nynorsk eller bokmål.

For å installere, kreves et python-miljø. Python 3 (testet på python3.10) anbefales.

    ./setup.sh

For å kjøre en prøveidentifisering kan du gi et filnavn som en parameter til identify.py-skriptet ved å bruke -f-alternativet som følgende:

    python3 identify.py -f norwegian_text.txt

For å kjøre dette på en directory rekursivt:

    python3 identify.py -d path/to/dir

CUDA-basert GPU anbefales. For å stille inn batchstørrelsen, bruk følgende:

    python3 identify.py -b 16 -d path/to/dir 

Dette proseserer 16 filer som er en batch. Directorymoden bruker bare beginnelsen av filer.

# License

[MIT license](https://github.com/textlab/norwegian_ml_tagger/blob/master/LICENSE)


