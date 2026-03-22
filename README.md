Based on CircuitNet (BSD-3 License, Peking University)

Modified by: Yash Ganjare
- Added CPU support
- Reduced dataset pipeline
- Visualization notebook
- Synthetic data experiments

🔹 Intuition
High demand → more wires needed ❗
Low capacity → less space ❗
High density → crowded region ❗

👉 All together = congestion hotspot

High demand → more current flow ❗
High density → longer paths / higher resistance ❗
Poor routing (low capacity) → inefficient power delivery ❗

👉 All together = higher voltage drop (IR drop hotspot)

High density → cells too close ❗
High demand → excessive routing pressure ❗
Low capacity → routing overflow ❗

👉 All together = design rule violations (DRC hotspot)

Run this to train :- python train.py --task congestion_gpdl --cpu --max_iters 300 && python train.py --task drc_routenet --cpu --max_iters 300 && python train.py --task irdrop_mavi --cpu --max_iters 300

Testing :- 
python test.py --task congestion_gpdl --pretrained work_dir/congestion_gpdl/model_iters_300.pth --cpu

python test.py --task drc_routenet --pretrained work_dir/drc_routenet/model_iters_300.pth --cpu

python test.py --task irdrop_mavi --pretrained work_dir/irdrop_mavi/model_iters_300.pth --cpu"# CircuitGNN-Graph-Learning-for-VLSI-Congestion-DRC-IR-Drop-Prediction-" 
