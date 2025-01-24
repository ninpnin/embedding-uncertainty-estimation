#test

for DATALEN in 850000 425000 170000 85000
	PYTHONPATH="$PYTHONPATH:." python3 scripts/gibbs_sampler.py --datapath data/movielens_train.json --data_len $DATALEN --samples 2000 --dim 20 --example_word StarWars1977 --use_tf True --pg_iter 10 --lambda0 1.0 --mvn_method cholesky
end
