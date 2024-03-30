docker run -it --rm --name boolean-search -v "$PWD":/work_dir -w /work_dir python:3.12-alpine pip3 install -r requirements.txt && python3 hw_boolean_search.py \
    --queries_file data/queries.numerate.txt \
    --objects_file  data/objects.numerate.txt\
    --docs_file data/docs.txt \
    --submission_file output.csv

#python3 hw_boolean_search.py     --queries_file data/queries.numerate.txt     --objects_file data/objects.numerate.txt    --docs_file data/docs.txt     --submission_file output.csv
