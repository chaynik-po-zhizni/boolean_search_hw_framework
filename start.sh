docker run -it --rm --name boolean-search -v "$PWD":/work_dir -w /work_dir python:3.12-alpine python hw_boolean_search.py \
    --queries_file /work_dir/data/queries.numerate.txt \
    --objects_file  /work_dir/data/objects.numerate.txt\
    --docs_file /work_dir/data/docs.txt \
    --submission_file /work_dir/output.csv
