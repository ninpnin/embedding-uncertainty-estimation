
echo "V=200, K=10"
for DATASET in data/ten_V200K10/*.json
	for DATALEN in 500000 1000000 2000000
		PYTHONPATH="$PYTHONPATH:." python3 scripts/gibbs_sampler.py --datapath $DATASET \
			--data_len $DATALEN --samples $SAMPLES --dim 10 --example_word word1 \
			--use_tf True --pg_iter 10 --mvn_method cholesky --prefix simulation
	end
end

echo "V=100, K=5"
set SAMPLES 2000
for DATASET in data/ten_V100K5/*.json
	for DATALEN in 500000 1000000
		PYTHONPATH="$PYTHONPATH:." python3 scripts/gibbs_sampler.py --datapath $DATASET \
			--data_len $DATALEN --samples $SAMPLES --dim 5 --example_word word1 \
			--use_tf True --pg_iter 10  --mvn_method cholesky --prefix simulation
	end
end
