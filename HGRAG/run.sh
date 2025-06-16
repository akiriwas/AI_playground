for c in {1..128..12}; do
	for o in {3..7}; do
		for i in {1..50}; do
			python HSFC_test.py -C $c -O $o | grep "RESULTS"
			sleep 1
		done
	done
done
