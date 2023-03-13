cd data

python canary_insertion.py call 4 100
python canary_insertion.py call 6 100
python canary_insertion.py call 6 1000

python build_vocab.py call-4-100
python build_vocab.py call-6-100
python build_vocab.py call-6-1000

python build_glove.py call-4-100
python build_glove.py call-6-100
python build_glove.py call-6-1000

cd ..
cd model

python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100
python main2.py call 4 100

python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100
python main2.py call 6 100

python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000
python main2.py call 6 1000

cd ..

cd data

python canary_insertion.py pin 4 100
python canary_insertion.py pin 4 500
python canary_insertion.py pin 6 100
python canary_insertion.py pin 6 500

python build_vocab.py pin-4-100
python build_vocab.py pin-4-500
python build_vocab.py pin-6-100
python build_vocab.py pin-6-500

python build_glove.py pin-4-100
python build_glove.py pin-4-400
python build_glove.py pin-6-100
python build_glove.py pin-6-500

cd ..
cd model

python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100
python main2.py pin 4 100

python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500
python main2.py pin 4 500

python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100
python main2.py pin 6 100

python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500
python main2.py pin 6 500

cd ..