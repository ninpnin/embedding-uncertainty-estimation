echo "V=100, K=5"
#for DATASET in data/ten_V200_K5/*.json
for R in 4 5 6 7 8 9 10
	for DATALEN in 1000 2000 5000 10000 20000 50000 100000 500000 1000000 2000000
        for FOLDER in $R-gibbs-N-$DATALEN-K-5-V-100-*simulation-*/
            PYTHONPATH="$PYTHONPATH:." python3 scripts/sample_based_coverage.py --sample_folder $FOLDER \
                --data_path data/ten_V100_K5/$R.json;
            end
	end
end
exit
#test
echo "V=200, K=20"
for R in 4 5 6 7 8 9 10
	for DATALEN in 1000 2000 5000 10000 20000 50000 100000 500000 1000000
        PYTHONPATH="$PYTHONPATH:." python3 scripts/sample_based_coverage.py --sample_folder $R-gibbs-N-$DATALEN-D-20-*simulation-*/ \
            --data_path data/ten_V200_K20/$R.json;
	end
end

echo "V=200, K=5"
#for DATASET in data/ten_V100_K10/*.json
for R in 123
    echo "Dataset" $R
	for DATALEN in 1000# 2000 5000 10000 20000 50000 100000 500000 1000000 2000000
        echo "Data length" $DATALEN
        for FOLDER in $R-gibbs-N-$DATALEN-D-5-*simulation-*/
            echo Folder $FOLDER;
            PYTHONPATH="$PYTHONPATH:." python3 scripts/sample_based_coverage.py --sample_folder $FOLDER \
                --data_path data/ten_V200_K5/$R.json;
        end
	end
end

echo "V=100, K=10"
#for DATASET in data/ten_V200_K5/*.json
for R in 123 #4 5 6 7 8 9 10
	for DATALEN in 1000 2000 5000 10000 20000 50000 100000 500000 1000000 2000000
        for FOLDER in $R-gibbs-N-$DATALEN-D-10-*simulation-*/
            PYTHONPATH="$PYTHONPATH:." python3 scripts/sample_based_coverage.py --sample_folder $FOLDER \
                --data_path data/ten_V100_K10/$R.json;
            end
	end
end
