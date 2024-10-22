declare -i N=20
declare -i N0=5

#
for i in $(seq $N0 $N); do
	mkdir -p "run_$i"
done
#
#
for i in $(seq $N0 $N); do
    	# Clear the folder contents
    	rm -rf "run_$i"/*
done
#
#
for i in $(seq $N0 $N); do
    	# Copy all contents from source into the run_$i folder
    	cp -r source/* "run_$i"
done
#
python3 init_cell_number_maker.py
#
for i in $(seq $N0 $N); do
	cd "run_$i"
	./Organoid_init
	cd ..
done
#
#
for i in $(seq $N0 $N); do
	cd "run_$i"
	./do_all.sh
	cd ..
done
#
python3 org_pp_over_runs.py




#mkdir a3dot20; cd a3dot20; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a3dot40; cd a3dot40; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a3dot60; cd a3dot60; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a3dot80; cd a3dot80; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a4dot00; cd a4dot00; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a6dot00; cd a6dot00; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a8dot00; cd a8dot00; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a10dot00; cd a10dot00; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#mkdir a20dot00; cd a20dot00; for i in $(seq 1 $N); do mkdir run_${i}/; done; cd ..;
#for i in {1..50}; do mkdir run_${i}/; done
#for i in {1..50}; do cp pp_CPM_v5.py run_${i}/; done 
#for i in {16..50}; do cd run_${i}; ./CPM; cd ..; done
