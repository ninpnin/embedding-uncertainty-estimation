echo K5 V200
for DATASET in 1 2 3 4 5 6 7 8 9 10;
	for DATASIZE in 1000 2000 5000 10000 20000 50000 100000 500000 1000000;
		python3 scripts/convert_cmdstanmodel.py --path trained/map/V200K5_MAP/$DATASET/stan_fit_map_K5_*_$DATASIZE.pkl \
			 --outpath trained/map/embedding/map_K5_V200_$DATASET"_"$DATASIZE.pkl --format embedding;
		PYTHONPATH="$PYTHONPATH:." python3 scripts/laplace_approximation.py --datapath data/ten_V200_K5/$DATASET.json \
			--data_len $DATASIZE --embedding trained/map/embedding/map_K5_V200_$DATASET"_"$DATASIZE.pkl --samples 2000 | tee -a logs/laplace-K5-V200.txt;
	end
end

echo K10 V200
for DATASET in 1 2 3 4 5 6 7 8 9 10;
	for DATASIZE in 1000 2000 5000 10000 20000 50000 100000 500000 1000000;
		python3 scripts/convert_cmdstanmodel.py --path trained/map/V200K10_MAP/$DATASET/stan_fit_map_K10_*_$DATASIZE.pkl \
			 --outpath trained/map/embedding/map_K10_V200_$DATASET"_"$DATASIZE.pkl --format embedding;
		PYTHONPATH="$PYTHONPATH:." python3 scripts/laplace_approximation.py --datapath data/ten_V200_K10/$DATASET.json \
			--data_len $DATASIZE --embedding trained/map/embedding/map_K10_V200_$DATASET"_"$DATASIZE.pkl --samples 2000 | tee -a logs/laplace-K10-V200.txt;
	end
end

echo K20 V200
for DATASET in 1 2 3 4 5 6 7 8 9 10;
	for DATASIZE in 1000 2000 5000 10000 20000 50000 100000 500000 1000000;
		python3 scripts/convert_cmdstanmodel.py --path trained/map/V200K20_MAP/$DATASET/stan_fit_map_K20_*_$DATASIZE.pkl \
			 --outpath trained/map/embedding/map_K20_V200_$DATASET"_"$DATASIZE.pkl --format embedding;
		PYTHONPATH="$PYTHONPATH:." python3 scripts/laplace_approximation.py --datapath data/ten_V200_K20/$DATASET.json \
			--data_len $DATASIZE --embedding trained/map/embedding/map_K20_V200_$DATASET"_"$DATASIZE.pkl --samples 2000 | tee -a logs/laplace-K20-V200.txt;
	end
end
